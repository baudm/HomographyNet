#!/usr/bin/env python

import sys
import random
import glob
import os.path
import uuid

from queue import Queue, Empty
from threading import Thread

import numpy as np
import cv2


def scale_down(img, target_size):
    src_height, src_width = img.shape
    src_ratio = src_height/src_width
    target_width, target_height = target_size
    if src_ratio < target_height/target_width:
        dst_size = (int(np.round(target_height/src_ratio)), target_height)
    else:
        dst_size = (target_width, int(np.round(target_width*src_ratio)))
    return cv2.resize(img, dst_size, interpolation=cv2.INTER_AREA)


def crop(img, origin, size):
    width, height = size
    x, y = origin
    return img[y:y + height, x:x + width]


def center_crop(img, target_size):
    target_width, target_height = target_size
    # Note the reverse order of width and height
    height, width = img.shape
    x = int(np.round((width - target_width)/2))
    y = int(np.round((height - target_height)/2))
    return crop(img, (x, y), target_size)


def generate_points():
    # Choose top-left corner of patch (assume 0,0 is top-left of image)
    # Restrict points to within 24-px from the border
    p = 32
    x, y = (random.randint(56, 136), 56)
    patch = [
        (x, y),
        (x + 128, y),
        (x + 128, y + 128),
        (x, y + 128)
    ]
    # Perturb
    perturbed_patch = [(x + random.randint(-p, p), y + random.randint(-p, p)) for x, y in patch]
    return np.array(patch), np.array(perturbed_patch)


def warp(img, orig, perturbed, target_size):
    # Get inverse homography matrix
    M = cv2.getPerspectiveTransform(np.float32(perturbed), np.float32(orig))
    return cv2.warpPerspective(img, M, target_size, flags=cv2.INTER_CUBIC)


def process_image(image_path, num_output=1):
    # Read as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img.shape < (240, 320):
        return

    target_size = (320, 240)
    img = scale_down(img, target_size)
    img = center_crop(img, target_size)

    patch_size = (128, 128)
    image_pairs = []
    offsets = []
    #orig_points = []
    #perturbed_points = []
    while len(offsets) < num_output:
        orig, perturbed = generate_points()
        a = crop(img, orig[0], patch_size)
        b = warp(img, orig, perturbed, target_size)
        b = crop(b, orig[0], patch_size)
        try:
            d = np.stack((a, b), axis=-1)
        except ValueError:
            continue
        image_pairs.append(d)
        offset = (perturbed - orig).reshape(-1)
        offsets.append(offset)
        #orig_points.append(orig)
        #perturbed_points.append(perturbed)
    print('done:', image_path)
    return image_pairs, offsets


class Worker(Thread):

   def __init__(self, input_queue, output_queue, num_samples):
       Thread.__init__(self)
       self.input_queue = input_queue
       self.output_queue = output_queue
       self.num_samples = num_samples

   def run(self):
       while True:
           img_path = self.input_queue.get()
           if img_path is None:
               break
           output = process_image(img_path, self.num_samples)
           self.input_queue.task_done()
           if output is not None:
               self.output_queue.put(output)


def pack(outdir, image_pairs, offsets):
    name = str(uuid.uuid4())
    pack = os.path.join(outdir, name + '.npz')
    with open(pack, 'wb') as f:
        np.savez(f, images=np.stack(image_pairs), offsets=np.stack(offsets))
    print('bundled:', name)


def bundle(queue, outdir):
    image_pairs = []
    offsets = []
    #orig_points = []
    #perturbed_points = []
    while True:
        try:
            d, o = queue.get(timeout=10)
        except Empty:
            break
        image_pairs.extend(d)
        offsets.extend(o)
        #orig_points.extend(orig)
        #perturbed_points.extend(perturbed)

        if len(image_pairs) >= 7680:
            pack(outdir, image_pairs, offsets)
            image_pairs = []
            offsets = []
        queue.task_done()

    if image_pairs:
        pack(outdir, image_pairs, offsets)


def main():
    if len(sys.argv) < 4:
        print('Usage: generate.py <output dir> <samples per input> <INPUT DIRS...>')
        exit(1)
    output_dir = sys.argv[1]
    samples = int(sys.argv[2])
    input_dirs = sys.argv[3:]

    # Create a queue to communicate with the worker threads
    input_queue = Queue()
    output_queue = Queue()

    num_workers = 8
    workers = []
    # Create worker threads
    for i in range(num_workers):
        worker = Worker(input_queue, output_queue, samples)
        worker.start()
        workers.append(worker)

    for d in input_dirs:
        for i in glob.iglob(os.path.join(d, '*.jpg')):
            input_queue.put(i)

    bundle(output_queue, output_dir)

    input_queue.join()
    for i in range(num_workers):
        input_queue.put(None)
    for worker in workers:
        worker.join()


if __name__ == '__main__':
    main()
