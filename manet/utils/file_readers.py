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


def write_yml(filename, input_dict):
    """Writes a dictionary to a yml 2.0 file.

    See: https://stackoverflow.com/a/40227545
    """
    sorted_dict = yaml.comments.CommentedMap()
    for k in sorted(input_dict):
        sorted_dict[k] = input_dict[k]
    yaml.round_trip_dump(sorted_dict, open(filename, 'w'))


def read_yml(filename):
    """Reads yml 2.0 file and outputs a dictionary.
    """
    out = yaml.load(open(filename, 'r'), Loader=yml_loader)
    if not out:
        return dict()
    return out


def write_json(filename, input_dict):
    """Writes a dictionary to a json file.
    """
    json.dump(input_dict, open(filename, 'w'),
              sort_keys=True, indent=4)


def read_json(filename):
    """Reads a json file to dictionary"""
    json_dict = json.load(open(filename, 'r'))
    return json_dict


def read_list(filename):
    """Reads file with caseids, separated by line.
    """
    f = open(filename, 'r')
    ids = []
    for line in f:
        ids.append(line.strip())
    f.close()

    # If the first line begins with '===' it is a header.
    if '===BEGIN DESCRIPTION===' == ids[0]:
        for i, _ in enumerate(ids):
            if _ == '===END DESCRIPTION===':
                break
        ids = ids[i + 1:]
    return ids


def write_list(filename, input_list, header=None, append=False):
    """Reads a list of strings and writes the list line by line to a text file."""
    mode = 'a' if append else 'w'
    with open(filename, mode) as f:
        if header and not append:
            header = ['===BEGIN DESCRIPTION==='] + header
            header += ['===END DESCRIPTION===']
            input_list = header + input_list
        for line in input_list:
            f.write(line.strip() + '\n')
