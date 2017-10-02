#!/usr/bin/env python

import pickle
import os.path
import glob
import uuid
import sys
import os.path
import numpy as np


def pack(b, x, y):
    name = str(uuid.uuid4())
    pack = os.path.join(b,  name + '.npz')
    with open(pack, 'wb') as f:
        np.savez(f, images=np.stack(x), offsets=np.stack(y))
    print('packed:', pack)


def main():
    if len(sys.argv) < 4:
        print('Usage: bundle_data.py <output dir> <samples per bundle> <input dir1> [input dir2] ...')
        exit(1)
    o = sys.argv[1]
    lim = int(sys.argv[2])
    inputs = sys.argv[3:]
    x = []
    y = []
    for i in inputs:
        for d in glob.glob(os.path.join(i, '*.dat')):
            with open(d, 'rb') as f:
                im, l = pickle.load(f)
            x.append(im)
            y.append(l)
            if len(y) >= lim:
                pack(o, x, y)
                x = []
                y = []
    # Pack any leftovers
    if x:
        pack(o, x, y)


if __name__ == '__main__':
    main()
