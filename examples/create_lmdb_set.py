"""Tools to write dicom series to a LMDB file."""
import lmdb
import os
from tqdm import tqdm
import simplejson as json
import numpy as np
from manet.utils import read_dcm_series, write_list


def write_kv_to_lmdb(db, key, value):
    """
    Write (key, value) to db.
    """
    success = False
    while not success:
        txn = db.begin(write=True)
        try:
            txn.put(key, value)
            txn.commit()
            success = True
        except lmdb.MapFullError:
            txn.abort()
            # double the map_size
            curr_limit = db.info()['map_size']
            new_limit = curr_limit * 2
            tqdm.write('MapFullError: Doubling LMDB map size to {}MB.'.format(new_limit))
            db.set_mapsize(new_limit)


def write_data_to_lmdb(db, key, image, metadata):
    """Write image data to db."""
    write_kv_to_lmdb(db, key, np.ascontiguousarray(image).tobytes())
    meta_key = key + '_metadata'
    ser_meta = json.dumps(metadata)
    write_kv_to_lmdb(db, meta_key, ser_meta)


def build_db(path, db_name, image_folders, generate_keys=False, dtype='int32'):
    """Build LMDB with images."""
    db = lmdb.open(os.path.join(path, db_name), map_async=True, max_dbs=0)
    if generate_keys:
        keys_filename = os.path.join(path, db_name + '_keys.lst')
        write_list(
            [], keys_filename, header=['LMDB keys for db {}'.format(db_name)])

    for key, folder in tqdm(image_folders):
        try:
            data, metadata = read_dcm_series(folder)
            data = data.astype(dtype)
            # If dataset is written to LMDB,
            # we do not need the filenames anymore.
            metadata.pop('filenames', None)
            series_ids = metadata.pop('series_ids', None)
            if series_ids:
                metadata['series_id'] = series_ids[0]

            metadata['dtype'] = '{}'.format(data.dtype)
            write_data_to_lmdb(db, key, data, metadata)
            if generate_keys:
                write_list([key], keys_filename, append=True)

        except Exception as e:
            tqdm.write('{} failed: {}'.format(path, e))
