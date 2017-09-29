#!/usr/bin/env python

import subprocess
import os.path
import random
import sys
import tempfile

import numpy as np
import cv2
import pickle

from queue import Queue
from threading import Thread


def generate_points():
    # Choose top-left corner of patch (assume 0,0 is top-left of image)
    # Restrict points to within 24-px from the border
    p = 32
    x, y = (random.randint(56, 136), 56)
    patch = [
        (x, y),
        (x + 128, y),
        (x, y + 128),
        (x + 128, y + 128)
    ]
    # Perturb
    perturbed_patch = [(x + random.randint(-p, p), y + random.randint(-p, p)) for x, y in patch]

    # We need the inverse homography, so reverse the order
    return list(zip(perturbed_patch, patch))


def make_params(points):
    params = []
    for o, p in points:
        # Quick and dirty
        params.append(str(o).strip('()'))
        params.append(str(p).strip('()'))
    return ' '.join(params)


def process_image(image_path, tmp_dir, outdir, num_output=1):
    img_name = os.path.basename(image_path)
    tmp_file = os.path.join(tmp_dir, img_name + '.png')
    # Quick fix to avoid unnecessary work
    if os.path.isfile(tmp_file):
        return
    # Resize, center crop, and convert to grayscale
    subprocess.run(['convert', image_path, '-resize', '320x240^',
                    '-gravity', 'center', '-extent', '320x240',
                    '-colorspace', 'gray',
                    tmp_file])
    args = [
        'convert', tmp_file, '-matte',
        '-virtual-pixel', 'transparent',
        '-distort', 'Perspective'
    ]
    for i in range(num_output):
        p = generate_points()
        # Crop params
        x, y = p[0][0]
        crop_params = '128x128+' + str(x) + '+' + str(y)

        # Crop orig
        out_orig = os.path.join(tmp_dir, img_name + '_' + str(i) + '-a.png')
        subprocess.run(['convert', tmp_file, '-crop', crop_params, out_orig])

        params = make_params(p)
        out_distorted = os.path.join(tmp_dir, img_name + '_' + str(i) + '-b.png')
        i_args = args + [params, '-crop', crop_params, out_distorted]
        print(img_name, params)
        # Distort and crop
        subprocess.run(i_args)

        label = []
        for a, b in p:
            for x in range(2):
                label.append(a[x] - b[x])
        label = np.array(label)

        # Read and normalize images
        im1 = cv2.imread(out_orig, cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(out_distorted, cv2.IMREAD_GRAYSCALE)
        try:
            im = np.stack((im1, im2), axis=-1)
        except ValueError:
            continue

        data = (im, label)
        dat = os.path.join(outdir, img_name + '_' + str(i) + '.dat')
        with open(dat, 'wb') as fdat:
            pickle.dump(data, fdat)


class Worker(Thread):

   def __init__(self, queue):
       Thread.__init__(self)
       self.queue = queue

   def run(self):
       while True:
           # Get the work from the queue and expand the tuple
           img_path, output_dir, samples = self.queue.get()
           process_image(img_path, output_dir, samples)
           self.queue.task_done()


def main():
    if len(sys.argv) != 4:
        print('Usage: generate_data.py <input dir> <output dir> <samples per input>')
        exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    samples = int(sys.argv[3])

    tmp_dir = tempfile.TemporaryDirectory().name
    
    # Create a queue to communicate with the worker threads
    queue = Queue()
    # Create 4 worker threads
    for x in range(4):
        worker = Worker(queue)
        # Setting daemon to True will let the main thread exit even though the workers are blocking
        worker.daemon = True
        worker.start()
    # Put the tasks into the queue as a tuple 
    for i in next(os.walk(input_dir))[-1]:
        img_path = os.path.join(input_dir, i)
        queue.put((img_path, tmp_dir, output_dir, samples))
        
    # Causes the main thread to wait for the queue to finish processing all the tasks
    queue.join()
   


if __name__ == '__main__':
    main()
