from __future__ import absolute_import
import math
import time
import sys
sys.setrecursionlimit(5000)

import hashlib

import random
from random import randint
import argparse
from collections import namedtuple
from datetime import datetime

from RTree import RTree

parser = argparse.ArgumentParser()

parser.add_argument('-default_ins_strategy', choices=['INS_AREA', 'INS_MARGIN', 'INS_OVERLAP'], help='default insert strategy', default="INS_AREA")
parser.add_argument('-default_spl_strategy', choices=['SPL_MIN_AREA', 'SPL_MIN_MARGIN', 'SPL_MIN_OVERLAP', 'SPL_QUADRATIC', 'SPL_GREENE'], help='default split strategy', default='SPL_MIN_OVERLAP')
parser.add_argument('-max_entry', type=int, help='maximum entry a node can hold', default=50)
parser.add_argument('-data_distribution', choices=['uniform', 'skew', 'gaussian', 'china', 'india'], help='data set distribution', default='gaussian')
parser.add_argument('-data_set_size', type=int, help='data set size', default=20000000)

class Test:
    def __init__(self):
        self.rtree = None
        self.rrstar = None
        self.config = None

    def Initialize(self, config):

        self.config = config

        self.rtree = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
        self.rtree.SetInsertStrategy(config.default_ins_strategy)
        self.rtree.SetSplitStrategy(config.default_spl_strategy)

        self.rrstar = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
        self.rrstar.SetInsertStrategy(config.default_ins_strategy)
        self.rrstar.SetSplitStrategy(config.default_spl_strategy)

if __name__ == '__main__':
    args = parser.parse_args()
    test = Test()
    test.Initialize(args)

    if args.data_distribution == 'uniform':
        open_data_file = "uniform_dataset.txt"
        x_max = 100000
        x_min = 0
        y_max = 100000
        y_min = 0
    elif args.data_distribution == 'skew':
        open_data_file = "skew_dataset.txt"
        x_max = 100000
        x_min = 0
        y_max = 100000
        y_min = 0
    elif args.data_distribution == 'gaussian':
        open_data_file = "gaussian_dataset_0.txt"
        x_max = 100000
        x_min = 0
        y_max = 100000
        y_min = 0
    elif args.data_distribution == 'china':
        open_data_file = "china_locations_shuffle.txt"
        x_max = 55000
        x_min = 7000
        y_max = 141000
        y_min = 70000
    elif args.data_distribution == 'india':
        open_data_file = "india_locations_shuffle.txt"
        x_max = 37500
        x_min = -12400
        y_max = 126000
        y_min = 50000

    model_dataset = []
    if args.data_distribution in ['uniform','skew','gaussian']:
        with open(open_data_file) as input_file:
            n = 0
            for line in input_file:
                if n%2 == 0:
                    model_dataset.append([float(line[:-1]) - 0.0001, float(line[:-1])])
                else:
                    model_dataset[-1].append(float(line[:-1]) - 0.0001)
                    model_dataset[-1].append(float(line[:-1]))
                n += 1

                if len(model_dataset) == int(args.data_set_size) and len(model_dataset[-1]) == 4:
                    break
    elif args.data_distribution in ['china','india']:
        with open(open_data_file) as input_file:
            n = 0
            for line in input_file:
                if n%2 == 0:
                    model_dataset.append([float(line[:-1])*1000 - 10**(-8), float(line[:-1])*1000])
                else:
                    model_dataset[-1].append(float(line[:-1])*1000 - 10**(-8))
                    model_dataset[-1].append(float(line[:-1])*1000)
                n += 1

    """
    construct RTree and RR*Tree for the 3 datasets
    """
    data_set_size = len(model_dataset)
    for j in range(data_set_size):

        if j%10000 == 0:
            print("Building Tree")
            print(j/10000.0)

        insert_rect = model_dataset[j]
        test.rtree.DefaultInsert(insert_rect)
        # test.rrstar.DirectRRInsert(insert_rect)
        # test.rrstar.DirectRRSplit()
        test.rrstar.DirectInsert(insert_rect)
        test.rrstar.DirectSplitWithReinsert()

    """
    Final Test
    """
    test_result = []

    # 0.005% query size
    k = 0
    query_area = (0.005/100) * (x_max - x_min) * (y_max - y_min)
    side = ( query_area**0.5 )/2

    RTree_count = 0
    RRstar_count = 0

    while k < 1000:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:
            k = k + 1
            RTree_count += test.rtree.Query((x - side, x + side, y - side, y + side))
            RRstar_count += test.rrstar.Query((x - side, x + side, y - side, y + side))

    manual_count = 0
    for obj in model_dataset:
        left = max(x-side, obj[0])
        right = min(x+side, obj[1])
        bottom = max(y-side, obj[2])
        top = min(y+side, obj[3])
        if left < right and bottom < top:
            manual_count += 1

    print(["0.005% query", manual_count, test.rrstar.QueryResult(), test.rtree.QueryResult()])
    test_result.append(["0.005% query", RRstar_count, RTree_count])


    # 0.01% query size
    k = 0
    query_area = (0.01/100) * (x_max - x_min) * (y_max - y_min)
    side = ( query_area**0.5 )/2

    RTree_count = 0
    RRstar_count = 0

    while k < 1000:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:
            k = k + 1
            RTree_count += test.rtree.Query((x - side, x + side, y - side, y + side))
            RRstar_count += test.rrstar.Query((x - side, x + side, y - side, y + side))

    manual_count = 0
    for obj in model_dataset:
        left = max(x-side, obj[0])
        right = min(x+side, obj[1])
        bottom = max(y-side, obj[2])
        top = min(y+side, obj[3])
        if left < right and bottom < top:
            manual_count += 1

    print(["0.01% query", manual_count, test.rrstar.QueryResult(), test.rtree.QueryResult()])
    test_result.append(["0.01% query", RRstar_count, RTree_count])


    # 0.05% query size
    k = 0
    query_area = (0.05/100) * (x_max - x_min) * (y_max - y_min)
    side = ( query_area**0.5 )/2

    RTree_count = 0
    RRstar_count = 0

    while k < 1000:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:
            k = k + 1
            RTree_count += test.rtree.Query((x - side, x + side, y - side, y + side))
            RRstar_count += test.rrstar.Query((x - side, x + side, y - side, y + side))

    manual_count = 0
    for obj in model_dataset:
        left = max(x-side, obj[0])
        right = min(x+side, obj[1])
        bottom = max(y-side, obj[2])
        top = min(y+side, obj[3])
        if left < right and bottom < top:
            manual_count += 1

    print(["0.05% query", manual_count, test.rrstar.QueryResult(), test.rtree.QueryResult()])
    test_result.append(["0.05% query", RRstar_count, RTree_count])


    # 0.1% query size
    k = 0
    query_area = (0.1/100) * (x_max - x_min) * (y_max - y_min)
    side = ( query_area**0.5 )/2

    RTree_count = 0
    RRstar_count = 0

    while k < 1000:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:
            k = k + 1
            RTree_count += test.rtree.Query((x - side, x + side, y - side, y + side))
            RRstar_count += test.rrstar.Query((x - side, x + side, y - side, y + side))

    manual_count = 0
    for obj in model_dataset:
        left = max(x-side, obj[0])
        right = min(x+side, obj[1])
        bottom = max(y-side, obj[2])
        top = min(y+side, obj[3])
        if left < right and bottom < top:
            manual_count += 1

    print(["0.1% query", manual_count, test.rrstar.QueryResult(), test.rtree.QueryResult()])
    test_result.append(["0.1% query", RRstar_count, RTree_count])


    # 0.5% query size
    k = 0
    query_area = (0.5/100) * (x_max - x_min) * (y_max - y_min)
    side = ( query_area**0.5 )/2

    RTree_count = 0
    RRstar_count = 0

    while k < 1000:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:
            k = k + 1
            RTree_count += test.rtree.Query((x - side, x + side, y - side, y + side))
            RRstar_count += test.rrstar.Query((x - side, x + side, y - side, y + side))

    manual_count = 0
    for obj in model_dataset:
        left = max(x-side, obj[0])
        right = min(x+side, obj[1])
        bottom = max(y-side, obj[2])
        top = min(y+side, obj[3])
        if left < right and bottom < top:
            manual_count += 1

    print(["0.5% query", manual_count, test.rrstar.QueryResult(), test.rtree.QueryResult()])
    test_result.append(["0.5% query", RRstar_count, RTree_count])


    # 1% query size
    k = 0
    query_area = (1.0/100) * (x_max - x_min) * (y_max - y_min)
    side = ( query_area**0.5 )/2

    RTree_count = 0
    RRstar_count = 0

    while k < 1000:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:
            k = k + 1
            RTree_count += test.rtree.Query((x - side, x + side, y - side, y + side))
            RRstar_count += test.rrstar.Query((x - side, x + side, y - side, y + side))

    manual_count = 0
    for obj in model_dataset:
        left = max(x-side, obj[0])
        right = min(x+side, obj[1])
        bottom = max(y-side, obj[2])
        top = min(y+side, obj[3])
        if left < right and bottom < top:
            manual_count += 1

    print(["1% query", manual_count, test.rrstar.QueryResult(), test.rtree.QueryResult()])
    test_result.append(["1% query", RRstar_count, RTree_count])


    # 2% query size
    k = 0
    query_area = (2.0/100) * (x_max - x_min) * (y_max - y_min)
    side = ( query_area**0.5 )/2

    RTree_count = 0
    RRstar_count = 0

    while k < 1000:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:
            k = k + 1
            RTree_count += test.rtree.Query((x - side, x + side, y - side, y + side))
            RRstar_count += test.rrstar.Query((x - side, x + side, y - side, y + side))

    manual_count = 0
    for obj in model_dataset:
        left = max(x-side, obj[0])
        right = min(x+side, obj[1])
        bottom = max(y-side, obj[2])
        top = min(y+side, obj[3])
        if left < right and bottom < top:
            manual_count += 1

    print(["2% query", manual_count, test.rrstar.QueryResult(), test.rtree.QueryResult()])
    test_result.append(["2% query", RRstar_count, RTree_count])

    print()
    #print("R*-Tree Results")
    print(args.data_distribution)
    print(test_result)