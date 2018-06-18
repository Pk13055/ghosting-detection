#!/usr/bin/env python3
"""
    @author Pratik K.
    @description Generates a heatmap for the
    the correlation to detect areas of high blur
    Proof of concept implemented without thresholding
    and globalized correlation summation
"""
import argparse
from sys import argv as rd
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
import numpy as np
import cv2


parser = argparse.ArgumentParser(description='Displays a heatmap showing possible blurring locations alongside intensity')
parser.add_argument('-f', '--file', help='file path for the frame', required=True, type=str)
parser.add_argument('-p', '--patch', help='the patch size for localization', type=int, default=50, required=True)
parser.add_argument('-t', '--threshold', help='thresholding for ignoring matched patches', type=float, default=0.8, required=False)


def plot(img, hmap):
    '''
        Plots the image and the blur heatmap
        for the same
        @param img -> np.array: image
        @param hmap -> np.array: heatmap for correlation values
    '''
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(hmap, cmap='hot', interpolation='nearest')
    plt.title('Heat map for possible blurring')
    plt.show(block=True)


def main():
    args = parser.parse_args()
    img = cv2.imread(args.file, cv2.IMREAD_GRAYSCALE)
    patch_size = args.patch
    patches = view_as_windows(img, (patch_size, patch_size), step=patch_size)
    hmap = None
    print("Total %d, %d" % patches.shape[:-2])
    for i, r in enumerate(patches):
        for j, template in enumerate(r):
            print("Processing %d, %d" % (i, j), end="\r")
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            # create a mask and threshold off only local region to the patch
            cut_val = args.threshold * np.max(res)
            res[res < cut_val] = 0
            if hmap is not None: hmap += res
            else: hmap = res
    plot(img, hmap)


if __name__ == "__main__":
    main()
