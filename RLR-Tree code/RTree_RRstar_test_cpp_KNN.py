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
    tree_acc_no_1 = 0
    ref_tree_acc_no_1 = 0
    tree_acc_no_2 = 0
    ref_tree_acc_no_2 = 0
    tree_acc_no_3 = 0
    ref_tree_acc_no_3 = 0
    tree_acc_no_4 = 0
    ref_tree_acc_no_4 = 0
    tree_acc_no_5 = 0
    ref_tree_acc_no_5 = 0

    k=0

    while k < 100:
        k+=1
        
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)

        tree_acc_no_1 = tree_acc_no_1 + test.rrstar.KNNQuery(x,y,1)
        ref_tree_acc_no_1 = ref_tree_acc_no_1 + test.rtree.KNNQuery(x,y,1)

        tree_acc_no_2 = tree_acc_no_2 + test.rrstar.KNNQuery(x,y,5)
        ref_tree_acc_no_2 = ref_tree_acc_no_2 + test.rtree.KNNQuery(x,y,5)

        tree_acc_no_3 = tree_acc_no_3 + test.rrstar.KNNQuery(x,y,25)
        ref_tree_acc_no_3 = ref_tree_acc_no_3 + test.rtree.KNNQuery(x,y,25)

        tree_acc_no_4 = tree_acc_no_4 + test.rrstar.KNNQuery(x,y,125)
        ref_tree_acc_no_4 = ref_tree_acc_no_4 + test.rtree.KNNQuery(x,y,125)

        tree_acc_no_5 = tree_acc_no_5 + test.rrstar.KNNQuery(x,y,625)
        ref_tree_acc_no_5 = ref_tree_acc_no_5 + test.rtree.KNNQuery(x,y,625)

    #print("R*-Tree Results")

    print(["k1", tree_acc_no_1, ref_tree_acc_no_1, "k5", tree_acc_no_2, ref_tree_acc_no_2, "k25", tree_acc_no_3, ref_tree_acc_no_3, "k125", tree_acc_no_4, ref_tree_acc_no_4, "k625", tree_acc_no_5, ref_tree_acc_no_5])