#!/usr/bin/env python

import pickle
import os.path
import glob
import uuid
import sys
import os.path
import numpy as np


def main():
    if len(sys.argv) != 4:
        print('Usage: bundle_data.py <input dir> <output dir> <samples per bundle>')
        exit(1)
    p = sys.argv[1]
    b = sys.argv[2]
    lim = int(sys.argv[3])
    x = []
    y = []
    for d in glob.glob(os.path.join(p, '*.dat')):
        with open(d, 'rb') as f:
            im, l = pickle.load(f)
        x.append(im)
        y.append(l)
        if len(y) >= lim:
            name = str(uuid.uuid4())
            pack = os.path.join(b,  name + '.npz')
            with open(pack, 'wb') as f:
                np.savez(f, images=np.stack(x), offsets=np.stack(y))
            print('packed:', pack)
            x = []
            y = []


if __name__ == '__main__':
    main()
