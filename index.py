from __future__ import print_function
import argparse
import numpy as np
import cv2
import os
import glob
import pickle


def histogram1D(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [10], [0, 256])
    return hist


def histogram2D(img):
    HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([HSV], [0, 1], None, [20, 20], [0, 180, 80, 250])  # 2차원 히스토그램 계산
    return hist


def histogram3D(img):
    hist = cv2.calcHist([img], [0, 1, 2], None, [10, 10, 10], [30, 220, 50, 250, 0, 180])
    return hist


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to the dataset folder")
    ap.add_argument("-i", "--index", required=True, help="path to the index file")
    args = vars(ap.parse_args())

    dataset = args["dataset"]
    index = args["index"]

    fout = open(index, 'wb')

    for imagePath in glob.glob(dataset + "/*"):
        print(imagePath, '처리중...')

        loc1 = imagePath.rfind("\\")
        loc2 = imagePath.rfind(".")
        fileID = imagePath[loc1 + 1:loc2]
        image = cv2.imread(imagePath)

        hist = histogram2D(image)
        pickle.dump(fileID, fout)
        pickle.dump(hist, fout)

