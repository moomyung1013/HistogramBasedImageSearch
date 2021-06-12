from __future__ import print_function
import argparse
import operator

import numpy as np
import cv2
import os
import glob
from matplotlib import pyplot as plt
import pickle

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to the dataset folder")
    ap.add_argument("-i", "--index", required=True, help="path to the index file")
    ap.add_argument("-q", "--query", required=True, help="path to the query file")
    args = vars(ap.parse_args())

    dataset = args["dataset"]
    index = args["index"]
    query = args["query"]
    compareHist_method = cv2.HISTCMP_CORREL

    image = cv2.imread(query)

    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # 1차원 히스토그램 컬러변환
    HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # 2차원 히스토그램 컬러변환

    #hist = cv2.calcHist([gray], [0], None, [10], [0, 256]) # 1차원 히스토그램 계산
    hist = cv2.calcHist([HSV], [0, 1], None, [20, 20], [0, 180, 80, 250]) # 2차원 히스토그램 계산
    #hist = cv2.calcHist([image], [0, 1, 2], None, [10, 10, 10], [30, 220, 50, 250, 0, 180])  # 3차원 히스토그램 계산

    data = dict()
    data_list = []

    fin = open(index, 'rb')
    for imagePath in glob.glob(dataset + "/*"):
        fileID = pickle.load(fin)
        indexData = pickle.load(fin)

        val = cv2.compareHist(hist, indexData, compareHist_method)
        data[fileID] = val

    if compareHist_method == cv2.HISTCMP_BHATTACHARYYA or compareHist_method == cv2.HISTCMP_CHISQR :
        data_list = sorted(data.items(), reverse=False, key=lambda item: item[1])
    else:
        data_list = sorted(data.items(), reverse=True, key=lambda item: item[1])

    '''for i in range(0, 10):
        print(i+1, " : ", data_list[i][0])
    print()'''

    f = plt.figure(figsize=(15, 7))
    plt.subplot(241), plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    plt.title("Query Image"), plt.xticks([]), plt.yticks([])

    for i in range(0, 6):
        data_image = dataset + '/' + data_list[i][0] + '.jpg'
        plt.subplot(242 + i)
        try:
            plt.imshow(cv2.cvtColor(cv2.imread(data_image), cv2.COLOR_BGR2RGB))
        except cv2.error as e:
            data_image = dataset + '/' + data_list[i][0] + '.png'
        finally:
            plt.imshow(cv2.cvtColor(cv2.imread(data_image), cv2.COLOR_BGR2RGB))
            plt.title(data_list[i][0]), plt.xticks([]), plt.yticks([])
    plt.show()
