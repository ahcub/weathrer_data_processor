import logging
from argparse import ArgumentParser
from os import getcwd
from os.path import join, basename
import pandas

from utils import configure_logging, get_shape_file_and_correspondent_stations, run_tasks, clear_dir


LOG_FILE_NAME = 'weather_reports.log'

configure_logging(LOG_FILE_NAME)


CSV_DATA_PATH = r'C:\Users\Alex\Desktop\Upwork_data\NCDC_Climate_Raw Data'
SHAPEFILES_PATH = r'C:\Users\Alex\Desktop\Upwork_data\CRB_NCDC_SectionFiles'

root_dir = getcwd()
RESULT_DIR = r'DATA_FRAMES'
RESULT_DIR_PATH = join(root_dir, RESULT_DIR)


def create_weather_reports():
    logging.info('Start parsing weather data')
    clear_dir(RESULT_DIR_PATH)
    open(join(getcwd(), LOG_FILE_NAME), 'w').close()
    shape_file_and_correspondent_stations = get_shape_file_and_correspondent_stations(CSV_DATA_PATH, SHAPEFILES_PATH)
    run_process_of_making_data_frames(shape_file_and_correspondent_stations)
    logging.info('Work done')


def run_process_of_making_data_frames(shape_file_and_correspondent_stations):
    logging.info('Starting tasks')
    tasks_desc = []
    for shape_file, stations in shape_file_and_correspondent_stations.items():
        task_desc = {'target': make_data_frames, 'args': (shape_file, stations)}
        tasks_desc.append(task_desc)

    run_tasks(tasks_desc)


def make_data_frames(station_shape_file, stations_data):
    logging.info('Making data frame for shapefile: %s', basename(station_shape_file))
    prcp, tmax, tmin = aggregate_data_frames(stations_data)

    logging.info("joining {0:d} dataframes".format(len(prcp)))
    prcp_df, tmax_df, tmin_df = join_data_aggregated_data_frames(prcp, tmax, tmin)

    output_base = join(RESULT_DIR_PATH, basename(station_shape_file).rstrip('.shp'))
    dump_data_frames(output_base, prcp_df, tmax_df, tmin_df)


def aggregate_data_frames(stations_data):
    prcp, tmin, tmax = [], [], []
    for csv_file_path, stations in stations_data.items():
        logging.info('Reading file: %s', csv_file_path)
        csv_file = pandas.read_csv(csv_file_path, na_values=-9999,
                                   parse_dates=["DATE"], usecols=[0, 5, 6, 7, 8])
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


def parse_cmd_line():
    parser = ArgumentParser()
    parser.add_argument('-r', '--raw_data_path', required=True)
    parser.add_argument('-s', '--shape_files_path', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    create_weather_reports()
