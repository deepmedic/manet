"""An example on how to generate an LMDB set."""

import os
import glob2
from create_lmdb_set import build_db


def get_id(path):
    stripped = os.path.basename(path).split('_')[0]
    return stripped


def main():
    l =  glob2.glob('/breast/EMPIRE/*/*/*slices_E')
    l = [(get_id(_), _) for _ in l]
    build_db('/breast/', 'EMPIRE_LMDB', l, generate_keys=True)


if __name__ == '__main__':
    main()
