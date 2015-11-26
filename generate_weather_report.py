import logging
from ConfigParser import ConfigParser
from calendar import monthrange
from datetime import datetime
from glob import glob
from os import getcwd
from os.path import join, basename, splitext, abspath, dirname
import numpy
import pandas
from scipy.interpolate import griddata
from utils import *

LOG_FILE_NAME = 'process.log'

configure_logging(LOG_FILE_NAME)

DATA_FRAMES_DIR = r'DATA_FRAMES'
DAILY_RESULTS_DIR = r'DAILY'
MONTHLY_RESULTS_DIR = r'MONTHLY'


def create_weather_reports():
    logging.info('Start parsing weather data')
    config = ConfigParser()
    config.read(join(dirname(abspath(__file__)), 'paths_config.cfg'))

    csv_data_path = config.get('paths', 'csv_data_path')
    section_files_path = config.get('paths', 'section_files_path')
    basein_files_path = config.get('paths', 'basein_files_path')

    output_dir_path = config.get('paths', 'output_dir_path')

    data_frames_dir_path = join(output_dir_path, DATA_FRAMES_DIR)
    daily_results_dir_path = join(output_dir_path, DAILY_RESULTS_DIR)
    monthly_results_dir_path = join(output_dir_path, MONTHLY_RESULTS_DIR)

    clear_dir(data_frames_dir_path)
    open(join(getcwd(), LOG_FILE_NAME), 'w').close()
    shape_file_and_correspondent_stations = get_shape_file_and_correspondent_stations(csv_data_path, section_files_path)
    run_process_of_making_data_frames(shape_file_and_correspondent_stations, data_frames_dir_path)

    clear_dir(daily_results_dir_path)
    data_files = get_files_for_getting_daily_metrics(basein_files_path, section_files_path, data_frames_dir_path)
    run_processing_daily_metrics(data_files, daily_results_dir_path)

    clear_dir(monthly_results_dir_path)
    run_processing_monthly_metrics(daily_results_dir_path, monthly_results_dir_path)
    logging.info('Reports created. You can find them inside output folder')


def run_process_of_making_data_frames(shape_file_and_correspondent_stations, output_path):
    logging.info('Starting making data frame tasks')
    tasks_desc = []
    for shape_file, stations in shape_file_and_correspondent_stations.items():
        task_desc = {'target': make_data_frames, 'args': (shape_file, stations, output_path)}
        tasks_desc.append(task_desc)

    run_tasks(tasks_desc)


def make_data_frames(station_shape_file, stations_data, output_path):
    logging.info('Making data frame for shapefile: %s', basename(station_shape_file))
    prcp, tmax, tmin = aggregate_data_frames(stations_data)

    logging.info("joining {0:d} dataframes".format(len(prcp)))
    prcp_df, tmax_df, tmin_df = join_data_aggregated_data_frames(prcp, tmax, tmin)

    output_base = join(output_path, basename(station_shape_file).rstrip('.shp'))
    dump_data_frames(output_base, prcp_df, tmax_df, tmin_df)


def aggregate_data_frames(stations_data):
    prcp, tmin, tmax = [], [], []
    for csv_file_path, stations in stations_data.items():
        logging.info('Reading file: %s', csv_file_path)
        try:
            csv_file = pandas.read_csv(csv_file_path, na_values=-9999,
                                       parse_dates=["DATE"], usecols=[0, 5, 8, 11, 12])
        except ValueError:
            logging.exception('Failed to parse file: %s', csv_file_path)
        else:
            for group_info in csv_file.groupby("STATION").groups.items():
                station, _ = group_info
                if station in stations:
                    logging.info("--processing station: " + station)
                    prcp.append(get_specific_data_frame_for_group(csv_file, group_info, "PRCP"))
                    tmin.append(get_specific_data_frame_for_group(csv_file, group_info, "TMIN"))
                    tmax.append(get_specific_data_frame_for_group(csv_file, group_info, "TMAX"))
    return prcp, tmax, tmin


def get_specific_data_frame_for_group(global_data_frame, group_info, data_name):
    group_id, indexes = group_info
    data_frame = global_data_frame.loc[indexes, ["DATE", data_name]]
    data_frame[group_id] = data_frame.pop(data_name)
    data_frame.index = data_frame.pop("DATE")
    return data_frame


def join_data_aggregated_data_frames(prcp, tmax, tmin):
    prcp_df = join_data_frames(prcp)
    tmin_df = join_data_frames(tmin)
    tmax_df = join_data_frames(tmax)
    return prcp_df, tmax_df, tmin_df


def join_data_frames(data_frames):
    result_df = data_frames[0]
    for df in data_frames[1:]:
        result_df = result_df.merge(df, how="outer", left_index=True, right_index=True)
    return result_df


def dump_data_frames(output_base, prcp_df, tmax_df, tmin_df):
    prcp_df.to_csv(output_base + '_prcp.csv')
    tmin_df.to_csv(output_base + '_tmin.csv')
    tmax_df.to_csv(output_base + '_tmax.csv')


def run_processing_daily_metrics(data_files, output_path):
    logging.info('Start processing daily')
    tasks_desc = []
    for basein_file_path, section_file_path, csv_files_paths in data_files:
        task_desc = {'target': process_daily, 'args': (basein_file_path, section_file_path,
                                                       csv_files_paths, output_path)}
        tasks_desc.append(task_desc)

    run_tasks(tasks_desc)


def process_daily(basein_file_path, section_file_path, csv_files_paths, output_path):
    logging.info('Process daily for %s', basename(basein_file_path))
    x, y, masks_array = load_basein_file(basein_file_path)
    section_stations = load_section_file(section_file_path)

    for csv_file_path in csv_files_paths:
        logging.info('Processing %s', basename(csv_file_path))
        csv_file = pandas.read_csv(csv_file_path, parse_dates=True, index_col=0)
        output_file_name = splitext(basename(csv_file_path))[0] + '_processed.csv'
        output_file_path = join(output_path, output_file_name)

        with open(output_file_path, 'w', 0) as output_file:
            output_file.write(','.join(("date", "area-weighted", "max", "min", "count", "gauge_pairs")) + '\n')

            for date in csv_file.index:
                points, stations, values = prepare_data_raw_data(csv_file, date, section_stations)

                count = len(values)
                if len(values) == 0:
                    awa = numpy.NaN
                    upr = numpy.NaN
                    lwr = numpy.NaN
                else:
                    values = numpy.array(values)
                    points = numpy.array(points)
                    if not numpy.all(values == values[0]):
                        grid = griddata(points, values, (x, y), method="nearest")
                        grid = numpy.ma.masked_where(masks_array == 0, grid)
                        contrib = []
                        unique_vals = numpy.unique(grid)
                        count = 0
                        for station, value in zip(stations, values):
                            if value in unique_vals:
                                mask = numpy.zeros_like(grid)
                                mask = numpy.ma.masked_where(masks_array == 0, mask)
                                mask[numpy.where(grid == value)] = 1
                                unique_vals_count = numpy.sum(mask)
                                contrib.append(unique_vals_count / masks_array.sum())
                                count += 1
                        awa = grid.mean()
                        upr = max(unique_vals)
                        lwr = min(unique_vals)
                        values = contrib
                    else:
                        awa = values[0]
                        upr = values[0]
                        lwr = values[0]

                line = "{0:s},{1:E},{2:E},{3:E},{4:d},".format(str(date), float(awa), float(upr), float(lwr), count)
                for site, val in zip(stations, values):
                    line += ' ' + site + "|{0:E}".format(val)
                line += '\n'
                output_file.write(line)


def prepare_data_raw_data(csv_file, date, section_stations):
    date_df = csv_file.loc[date, :].dropna()
    points = []
    values = []
    stations = []
    if date_df.shape[0] == 1:
        values = date_df.values
    elif date_df.shape[0] != 0:
        for station, value in zip(date_df.index, date_df.values):
            if station in stations:
                raise Exception("Same station: %s found again on date %s" % (str(station), str(date)))

            points.append(section_stations[station])
            values.append(value)
            stations.append(station)
    return points, stations, values


def run_processing_monthly_metrics(data_files_path, output_path):
    tasks_desc = []
    for file_path in glob(join(data_files_path, '*.csv')):
        task_desc = {'target': calculate_monthly_values, 'args': (file_path, output_path)}
        tasks_desc.append(task_desc)

    run_tasks(tasks_desc)


def calculate_monthly_values(data_file_path, output_path):
    logging.info('Calculating monthly metrics for file: %s', data_file_path)
    daily_data_frame = pandas.read_csv(data_file_path, parse_dates=True, index_col=0,
                                       usecols=["date", "area-weighted"], na_values=["NAN"])

    if "_prcp_" in data_file_path.lower():
        groups = daily_data_frame.groupby([lambda x: x.year, lambda x: x.month]).sum()
    else:
        groups = daily_data_frame.groupby([lambda x: x.year, lambda x: x.month]).mean()

    datetimes_collection = []
    for year, month in groups.index:
        datetimes_collection.append(datetime(year=year, month=month, day=monthrange(year, month)[1],
                                             hour=23, minute=59, second=59))
    groups.index = datetimes_collection
    groups.to_csv(join(output_path, basename(data_file_path)), index_label="datetime")


if __name__ == '__main__':
    try:
        create_weather_reports()
    except Exception as e:
        logging.exception('Error on executing main function')
