#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from collections import OrderedDict
import time
from utils.pre_data import read_xlsx

import numpy as np

f = "datasets/"


def loadCsv(path):
    data = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(np.array(row))
    data = np.array(data)
    (n, d) = data.shape
    return data, n, d


def loadPima():
    data, n, d = loadCsv(f + 'pima/pima-indians-diabetes.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != '1'] = '-1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadCodec():
    df = read_xlsx(f + 'codec/codec.xlsx')
    return df
    #x_train, x_test = split_dataset(df)
    #return x_train, x_test


def loadCollections():
    df = read_xlsx(f + 'collections/collections.xlsx')
    return df
    # x_train, x_test = split_dataset(df)
    # return x_train, x_test


def loadIo():
    df = read_xlsx(f + 'io/io.xlsx')
    return df
    # x_train, x_test = split_dataset(df)
    # return x_train, x_test


def loadJsoup():
    df = read_xlsx(f + 'jsoup/jsoup.xlsx')
    return df
    # x_train, x_test = split_dataset(df)
    # return x_train, x_test


def loadJsqlparser():
    df = read_xlsx(f + 'jsqlparser/jsqlparser.xlsx')
    return df
    # x_train, x_test = split_dataset(df)
    # return x_train, x_test


def loadMango():
    df = read_xlsx(f + 'mango/mango.xlsx')
    return df
    # x_train, x_test = split_dataset(df)
    # return x_train, x_test


def loadOrmlite():
    df = read_xlsx(f + 'ormlite/ormlite.xlsx')
    return df
    # x_train, x_test = split_dataset(df)
    # return x_train, x_test


d = OrderedDict()
s = time.time()

# d["pima"] = loadPima()                     #    768  34.90%
d["codec"] = loadCodec()
d["collections"] = loadCollections()
d["io"] = loadIo()
d["jsoup"] = loadJsoup()
d["jsqlparser"] = loadJsqlparser()
d["mango"] = loadMango()
d["ormlite"] = loadOrmlite()

print("Data loaded in {:5.2f}".format(time.time()-s))


