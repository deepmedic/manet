# encoding: utf-8
import ruamel.yaml as yaml
import simplejson as json

import logging
logger = logging.getLogger(__name__)

try:
    yml_loader = yaml.CLoader
except ImportError:
    logger.debug('ruamel.yaml CLoader not available. Falling back to python yaml reader.')
    yml_loader = yaml.Loader


def write_yml(input_dict, yml_path):
    """Writes a dictionary to a yml 2.0 file.

    See: https://stackoverflow.com/a/40227545
    """
    sorted_dict = yaml.comments.CommentedMap()
    for k in sorted(input_dict):
        sorted_dict[k] = input_dict[k]
    yaml.round_trip_dump(sorted_dict, open(yml_path, 'w'))


def read_yml(input_yml):
    """Reads yml 2.0 file and outputs a dictionary.
    """
    out = yaml.load(open(input_yml, 'r'), Loader=yml_loader)
    if not out:
        return dict()
    return out


def write_json(input_dict, json_path):
    """Writes a dictionary to a json file.
    """
    json.dump(input_dict, open(json_path, 'w'),
              sort_keys=True, indent=4)


def read_json(json_path):
    """Reads a json file to dictionary"""
    json_dict = json.load(open(json_path, 'r'))
    return json_dict


def read_list(list_filename):
    """Reads file with caseids, separated by line.
    """
    f = open(list_filename, 'r')
    ids = []
    for line in f:
        ids.append(line.strip())
    f.close()
    return ids
