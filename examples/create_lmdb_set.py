"""Tools to write dicom series to a LMDB file."""
import lmdb
from tqdm import tqdm
import simplejson as json
import numpy as np
from manet.utils import read_dcm_series


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


def build_db(path, image_folders):
    """Build LMDB with images."""
    db = lmdb.open(path, map_async=True, max_dbs=0)
    for key, folder in tqdm(image_folders):
        try:
            data, metadata = read_dcm_series(folder)
            # If dataset is written to LMDB,
            # we do not need the filenames anymore.
            metadata.pop('filenames', None)
            metadata['dtype'] = '{}'.format(data.dtype)
            write_data_to_lmdb(db, key, data, metadata)
        except Exception as e:
            tqdm.write('{} failed: {}'.format(path, e))
