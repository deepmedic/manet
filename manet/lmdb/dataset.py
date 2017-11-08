# encoding: utf-8
import os
import os.path
import sys
import lmdb
import numpy as np
import simplejson as json
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class LmdbDb(object):
    def __init__(self, lmdb_path):
        """Load an LMDB database, containing a dataset.

        The dataset should be structured as image_id: binary representing the contiguous block.
        If image_id is available we also need image_id_metadata which is a json parseble dictionary.
        This dictionary should contains the key 'shape' representing the shape and 'dtype'.

        Parameters
        ----------
        lmdb_path : str
            Path to LMDB database.


        """
        self.lmdb_path = os.path.abspath(
            os.path.expanduser(lmdb_path))
        self.env = lmdb.open(lmdb_path, max_readers=None, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        cache_file = '_cache_' + lmdb_path.replace(os.path.sep, '_')
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, 'r'))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor() if '_metadata' not in key]
            pickle.dump(self.keys, open(cache_file, 'w'))

    def __getitem__(self, key):
        with self.env.begin(buffers=True, write=False) as txn:
            if key not in self.keys:
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

