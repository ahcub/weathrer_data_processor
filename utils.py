import logging
import re
from distutils.dir_util import remove_tree
from glob import glob
from os.path import join, isdir, isfile, islink, exists, basename
import numpy
import pandas
import os
from multiprocessing import Process
from stat import *
import shapefile
import sys

LOGGING_FORMAT = '%(asctime)-15s %(levelname)s %(message)s'
DATE_FORMAT = '[%Y-%m-%d %H:%M:%S]'

RWXA = S_IRWXU | S_IRWXG | S_IRWXO
READ = S_IRUSR | S_IRGRP | S_IROTH
WRITE = S_IWUSR | S_IWGRP | S_IWOTH


def configure_logging(filename='app.log', level=logging.DEBUG):
    logging.basicConfig(datefmt=DATE_FORMAT, format=LOGGING_FORMAT, level=level, stream=sys.stdout)
    file_handler = logging.FileHandler(filename=join(os.getcwd(), filename))
    file_handler.level = level
    formatter = logging.Formatter(datefmt=DATE_FORMAT, fmt=LOGGING_FORMAT)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)


def get_shape_file_and_correspondent_stations(csv_data_path, shape_files_path):
    logging.info('Getting correspondent station for shape files')
    stations_inside_shapes_data = {}
    for shape_file_path in glob(join(shape_files_path, '*.shp')):
        stations_inside_shapes_data[shape_file_path] = {}

        shape_file_stations = get_station_names_from_shape_file(shape_file_path)
        for file_name, station in get_stations_in_file(csv_data_path):
            if station in shape_file_stations:
                if file_name not in stations_inside_shapes_data[shape_file_path]:
                    stations_inside_shapes_data[shape_file_path][file_name] = set()
                stations_inside_shapes_data[shape_file_path][file_name].add(station)

    return stations_inside_shapes_data


def get_station_names_from_shape_file(shape_file_name):
    shape_file = shapefile.Reader(shape_file_name)
    station_idx = get_field_index(shape_file, 'STATION')
    stations = set([record[station_idx] for record in shape_file.records()])
    return stations


def get_field_index(shape_file, field_name):
    fieldnames = [field[0] for field in shape_file.fields]
    return fieldnames.index(field_name) - 1


def get_stations_in_file(csv_data_path):
    for csv_file_path in glob(join(csv_data_path, '*.csv')):
        csv_file = pandas.read_csv(csv_file_path, usecols=[0])
        for station in csv_file.STATION.unique():
            yield csv_file_path, station


def run_tasks(tasks_description):
    tasks = []
    for task_desc in tasks_description:
        task = Process(target=task_desc['target'], args=task_desc['args'])
        task.start()
        tasks.append(task)

    for task in tasks:
        task.join()


def clear_dir(path):
    delete(path)
    mkpath(path)


def delete(path):
    if exists(path):
        if isdir(path):
            add_permissions_to_dir_rec(path, WRITE)
            remove_tree(path)
        elif isfile(path):
            add_permissions_to_path(path, WRITE)
            os.remove(path)
        elif islink(path):
            add_permissions_to_path(path, WRITE)
            os.unlink(path)


def add_permissions_to_path(path, permissions):
    return os.chmod(path, os.stat(path)[ST_MODE] | permissions)


def add_permissions_to_multiple_paths(root, paths, permissions):
    for path in paths:
        add_permissions_to_path(join(root, path), permissions)


def add_permissions_to_dir_rec(path, permissions):
    for root, dirs, files in os.walk(path):
        add_permissions_to_multiple_paths(root, dirs + files, permissions)


def mkpath(path):
    if not isdir(path):
        os.makedirs(path)


def load_basein_file(shape_file_path):
    shape_file = shapefile.Reader(shape_file_path)
    x_idx, y_idx, mask_idx = get_fields_indexes(shape_file)

    raw_x_array, raw_y_array, raw_masks_array = read_record_values(shape_file, mask_idx, x_idx, y_idx)

    return process_record_raw_arrays(raw_x_array, raw_y_array, raw_masks_array)


def read_record_values(shape_file, mask_idx, x_idx, y_idx):
    x_array = []
    y_array = []
    masks_array = []
    for record_index in range(shape_file.numRecords):
        record = shape_file.record(record_index)
        masks_array.append(int(record[mask_idx] != 'n'))

        if x_idx:
            x, y = float(record[x_idx]), float(record[y_idx])
        else:
            x, y = compute_coordinates_from_record(shape_file, record_index)

        x_array.append(x)
        y_array.append(y)

    return x_array, y_array, masks_array


def get_fields_indexes(shape_file):
    fieldnames = [field[0] for field in shape_file.fields]
    if 'x' in fieldnames and 'y' in fieldnames:
        x_idx = fieldnames.index('x') - 1
        y_idx = fieldnames.index('y') - 1
    else:
        x_idx = y_idx = None
    mask_idx = fieldnames.index("Intersect") - 1
    return x_idx, y_idx, mask_idx


def compute_coordinates_from_record(shape_file, record_index):
    shape = shape_file.shape(record_index)
    xmin, ymin, xmax, ymax = shape.bbox
    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0
    return x, y


def process_record_raw_arrays(raw_x_array, raw_y_array, raw_masks_array):
    x_array = numpy.unique(numpy.array(raw_x_array))
    y_array = numpy.unique(numpy.array(raw_y_array))
    x_array.sort()
    y_array.sort()
    x_size = x_array.shape[0]
    y_size = y_array.shape[0]

    masks_array = numpy.array(raw_masks_array)
    masks_array.resize(x_size, y_size)
    masks_array = numpy.flipud(masks_array.transpose())

    x, y = numpy.meshgrid(x_array, y_array)
    return x, y, masks_array


def load_section_file(shape_file_path):
    shape_file = shapefile.Reader(shape_file_path)
    fieldnames = [field[0] for field in shape_file.fields]
    station_idx = fieldnames.index("STATION") - 1

    stations = {}
    for record_index in range(shape_file.numRecords):
        station = shape_file.record(record_index)[station_idx]
        if station.strip():
            if station in stations:
                raise Exception("duplicate station: %s" % station)

            x, y = shape_file.shape(record_index).points[0]
            stations[station] = (x, y)

    return stations


def get_files_for_getting_daily_metrics(basein_files_root, section_files_root, data_files_root):
    basein_files_registry = create_files_registry(basein_files_root, '*.shp')
    section_files_registry = create_files_registry(section_files_root, '*.shp')
    data_files_registry = create_files_registry(data_files_root, '*.csv')

    files_registry = []
    for key in basein_files_registry.keys():
        if key not in section_files_registry:
            raise Exception('Could not find session %s for sections files in path %s' % (key, section_files_root))
        if key not in data_files_registry:
            raise Exception('Could not find session %s for data files in path %s' % (key, data_files_root))
        files_registry.append((basein_files_registry[key][0], section_files_registry[key][0], data_files_registry[key]))

    return files_registry


def create_files_registry(files_path, pattern):
    registry = {}
    for file_path in glob(join(files_path, pattern)):
        search_result = re.search('\d+', basename(file_path))
        if search_result is None:
            raise Exception('Could not extract session number from file name: %s' % file_path)
        key = int(search_result.group())
        if key not in registry:
            registry[key] = []

        registry[key].append(file_path)

    return registry
