# encoding: utf-8
import os
import copy
import lmdb
import numpy as np
import simplejson as json
from manet.utils import write_list, read_list


class LmdbDb(object):
    def __init__(self, path, db_name):
        """Load an LMDB database, containing a dataset.

        The dataset should be structured as image_id: binary representing the contiguous block.
        If image_id is available we also need image_id_metadata which is a json parseble dictionary.
        This dictionary should contains the key 'shape' representing the shape and 'dtype'.

        If the keys file is available, the file is loaded, otherwise generated.

        Parameters
        ----------
        path : str
            Path to folder with LMDB db.
        db_name : str
            Name of the database.

        """
        lmdb_path = os.path.join(path, db_name)
        lmdb_keys_path = os.path.join(path, db_name + '_keys.lst')
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(lmdb_path, max_readers=None, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] // 2

        if os.path.isfile(lmdb_keys_path):
            self._keys = read_list(lmdb_keys_path)
        else:
            # The keys file does not exist, we will generate one, but this can take a while.
            with self.env.begin(write=False) as txn:
                keys = [key for key, _ in txn.cursor() if '_metadata' not in key]
                write_list(keys, lmdb_keys_path, header=['LMDB keys for db {}'.format(db_name)])
                self._keys = keys

    def __delitem__(self, key):
        idx = self._keys.index[key]
        self._keys.pop(idx, None)

    def copy(self):
        return copy.deepcopy(self)

    def has_key(self, key):
        return key in self._keys

    def keys(self):
        return self._keys

    def __getitem__(self, key):
        with self.env.begin(buffers=True, write=False) as txn:
            if key not in self._keys:
                raise KeyError(key)
            buf = txn.get(key)
            meta_buf = txn.get(key + '_metadata')

        metadata = json.loads(str(meta_buf))
        dtype = metadata['dtype']
        shape = metadata['shape']
        data = np.ndarray(shape, dtype, buffer=buf)

        out = {}
        out['data'] = data
        out['metadata'] = metadata
        return out

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.lmdb_path + ')'

