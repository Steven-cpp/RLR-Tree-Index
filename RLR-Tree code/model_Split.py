# from tqdm import trange
import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim

import hashlib

import random
from random import randint
import argparse
from collections import namedtuple
import time
from datetime import datetime

from RTree import RTree

parser = argparse.ArgumentParser()

parser.add_argument('-action', choices=['train', 'test', 'test10', 'baseline'], default='train')
parser.add_argument('-objfile', help='data file')
parser.add_argument('-queryfile', help='query file')
parser.add_argument('-epoch', type=int, help='number of epoches', default=15)
parser.add_argument('-reward_query_width', type=float, default=0.01)
parser.add_argument('-reward_query_height', type=float, default=0.01)
parser.add_argument('-default_ins_strategy', help='default insert strategy', default="INS_AREA")
parser.add_argument('-default_spl_strategy', help='default split strategy', default='SPL_MIN_AREA')
parser.add_argument('-reference_tree_ins_strategy', help='default insert strategy for reference tree', default="INS_AREA")
parser.add_argument('-reference_tree_spl_strategy', help='default split strategy for reference tree', default='SPL_MIN_OVERLAP')
parser.add_argument('-action_space', type=int, help='number of possible actions', default=2)
parser.add_argument('-batch_size', type=int, help='batch_size', default=64)
parser.add_argument('-state_dim', type=int, help='input dimension', default=25)
parser.add_argument('-inter_dim', type=int, help='internal dimension', default=64)
parser.add_argument('-memory_cap', type=int, help='memory capacity', default=5000)
parser.add_argument('-lr', type=float, help='learning rate', default=0.01)
parser.add_argument('-gamma', type=float, help='reward discount factor', default=0.8)
parser.add_argument('-model_name', help='name of the model')
parser.add_argument('-target_update', type=int, help='update the parameters for target network every ? steps', default=30)
parser.add_argument('-epsilon', type=float, help='epsilon greedy', default=0.9)
parser.add_argument('-epsilon_decay', type=float, help='how fast to decrease epsilon', default=0.99)
parser.add_argument('-min_epsilon', type=float, help='minimum epsilon', default=0.1)
parser.add_argument('-max_entry', type=int, help='maximum entry a node can hold', default=50)
parser.add_argument('-query_for_reward', type=int, help='number of query used for reward', default=3)
parser.add_argument('-splits_for_update', type=int, help='number of splits for a reward computation', default=5)
parser.add_argument('-parts', type=int, help='number of parts to train', default=15)
parser.add_argument('-network', choices=['strategy', 'spl_loc', 'spl_loc_short', 'sort_spl_loc'], help='which network is used for training', default='sort_spl_loc')
parser.add_argument('-teacher_forcing', type=float, help='the percentage of splits that are with teacher forcing technique', default=0.1)
parser.add_argument('-data_distribution', choices=['uniform', 'skew', 'gaussian', 'china', 'india'], help='data set distribution', default='gaussian')
parser.add_argument('-data_set_size', type=int, help='data set size', default=20000000)
parser.add_argument('-training_set_size', type=int, help='training set size', default=100000)

class DQN(nn.Module):
    def __init__(self, input_dimension, inter_dimension, output_dimension):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_dimension, inter_dimension)
        self.linear2 = nn.Linear(inter_dimension, output_dimension)

    def forward(self, x):
        m = nn.SELU()
        x = self.linear1(x)
        x = m(x)
        x = self.linear2(x)
        # sf = nn.Softmax(dim=0)
        return x


class DQN2(nn.Module):
    def __init__(self, input_dimension=240, inter_dimension=300, output_dimension=48):
        super(DQN2, self).__init__()
        self.linear1 = nn.Linear(input_dimension, inter_dimension)
        self.linear2 = nn.Linear(inter_dimension, output_dimension)
        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        m = nn.SELU()
        x = self.linear1(x)
        x = m(x)
        x = self.linear2(x)
        return x

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory.clear()
        self.position = 0

    def __len__(self):
        return len(self.memory)

class SplitLearner:
    def __init__(self):
        self.tree = None
        self.reference_tree = None
        self.network = None
        self.target_network = None
        self.memory = None
        self.obj_input = None
        self.query_input = None
        self.config = None
        self.model_dataset = []
        self.training_dataset = []
        self.data_set_bounds = []
        md5content = "{}".format(datetime.now())
        self.id = hashlib.md5(md5content.encode()).hexdigest()

    def Initialize(self, config):
        self.config = config
        # if config.objfile:
        #     try:
        #         self.obj_input = open(config.objfile, 'r')
        #     except:
        #         print('object file does not exist.')
        # if config.queryfile:
        #     try:
        #         self.query_input = open(config.queryfile, 'r')
        #     except:
        #         print('query file does not exist.')

        self.tree = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
        self.tree.SetInsertStrategy(config.default_ins_strategy)
        self.tree.SetSplitStrategy(config.default_spl_strategy)
        self.reference_tree = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
        self.reference_tree.SetInsertStrategy(config.reference_tree_ins_strategy)
        self.reference_tree.SetSplitStrategy(config.reference_tree_spl_strategy)

        if self.config.network == 'strategy':
            self.network = DQN(self.config.state_dim, self.config.inter_dim, self.config.action_space)
            self.target_network = DQN(self.config.state_dim, self.config.inter_dim, self.config.action_space)
        if self.config.network == 'spl_loc':
            self.network = DQN2()
            self.target_network = DQN2()
        if self.config.network == 'spl_loc_short':
            self.network = DQN2(60, 60, 12)
            self.target_network = DQN2(60, 60, 12)
        if self.config.network == 'sort_spl_loc': # the final one!
            self.network = DQN2(self.config.action_space * 4, self.config.inter_dim, self.config.action_space)
            self.target_network = DQN2(self.config.action_space * 4, self.config.inter_dim, self.config.action_space)




        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        self.memory = ReplayMemory(self.config.memory_cap)
        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.config.lr)


    def NextObj(self):
        line = self.obj_input.readline()
        if not line:
            return None
        boundary = [float(e) for e in line.strip().split()]
        return boundary

    def ResetObjLoader(self):
        self.obj_input.seek(0)

    def NextQuery(self):
        line = self.query_input.readline()
        if not line:
            return None
        boundary = [float(e) for e in line.strip().split()]
        return boundary

    def ResetQueryLoader(self):
        self.query_input.seek(0)

    def EpsilonGreedy(self, q_values):
        p = random.random()
        if p < self.config.epsilon:
            action = randint(0, self.config.action_space - 1)
            return action
        else:
            values = list(q_values)
            action = values.index(max(values))
            return action
            # return np.argmax(q_values)

    def Optimize(self):
        if len(self.memory) < self.config.batch_size:
            return None

        transitions = self.memory.sample(self.config.batch_size) # this is a list
        list0 = []
        list1 = []
        list2 = []
        list3 = []
        for i in transitions:
            list0.append(list(i[0]))
            list1.append(i[1])
            list2.append(i[2])
            list3.append(list(i[3]))

        Qpred = self.network.forward(torch.tensor(list0, dtype=torch.float32).to(self.network.device))
        Qnext = self.target_network.forward(torch.tensor(list3, dtype=torch.float32).to(self.network.device))

        max_value, maxA = torch.max(Qnext, 1)
        actions = torch.tensor(list1, dtype=torch.int64)
        rewards = torch.tensor(list2, dtype=torch.float32)
        Qtarget = Qpred.clone()

        for i in range(len(maxA)):
            temp = rewards[i] + self.config.gamma*(max_value[i])
            Qtarget[i, actions[i]] = temp

        output = self.loss(Qpred, Qtarget.detach()).to(self.network.device)

        l = output.item()
        self.optimizer.zero_grad()
        output.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return l

        # batch = Transition(*zip(*transitions))

        # state_batch = torch.stack(batch.state).to(self.network.device)
        # reward_batch = torch.stack(batch.reward).to(self.network.device)
        # action_batch = torch.unsqueeze(torch.stack(batch.action), 1).to(self.network.device)
        

        # real_q_values = self.network(state_batch).to(self.network.device)

        # state_action_values = torch.gather(real_q_values, 1, action_batch).to(self.network.device)

        # mask = []
        # non_final_next_state = []
        # for s in batch.next_state:
        #     if s is not None:
        #         mask.append(1)
        #         non_final_next_state.append(s)
        #     else:
        #         mask.append(0)
        # next_state_values = torch.zeros(self.config.batch_size, 1).to(self.network.device)
        # if non_final_next_state:
        #     next_state_mask = torch.nonzero(torch.tensor(mask, dtype=torch.int64), as_tuple=False).squeeze(1)
        #     next_state_batch = torch.stack(non_final_next_state)    
        #     y, _ = self.target_network(next_state_batch).to(self.network.device).max(1, keepdim=True)
        #     next_state_values[next_state_mask] = y
        # expected_state_action_values = reward_batch + (next_state_values * self.config.gamma)

        # output = self.loss(state_action_values, expected_state_action_values).to(self.network.device)
        # l = output.item()
        # self.optimizer.zero_grad()
        # output.backward()
        # for param in self.network.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # self.optimizer.step()
        # return l

    def ComputeReward(self):
        access_rate_avg = 0
        for i in range(self.config.query_for_reward):
            query = self.tree.UniformRandomQuery(self.config.reward_query_width, self.config.reward_query_height)
            access_rate_avg += self.reference_tree.AccessRate(query) - self.tree.AccessRate(query)
        return access_rate_avg / self.config.query_for_reward

    def ComputeDenseRewardForList(self, obj_list):
        access_rate_avg = 0
        for obj in obj_list:
            for i in range(3):
                query = self.tree.UniformDenseRandomQuery(self.config.reward_query_width, self.config.reward_query_height, obj)
                #print(query)
                reference_rate = self.reference_tree.AccessRate(query)
                #print(reference_rate)
                tree_rate = self.tree.AccessRate(query)
                #print(tree_rate)
                access_rate_avg += reference_rate - tree_rate
                #print(access_rate_avg)
        return access_rate_avg / len(obj_list) / 3

    def ComputeDenseReward(self, object_boundary):
        access_rate_avg = 0
        for i in range(self.config.query_for_reward):
            query = self.tree.UniformDenseRandomQuery(self.config.reward_query_width, self.config.reward_query_height, object_boundary)
            access_rate_avg += self.reference_tree.AccessRate(query) - self.tree.AccessRate(query)
        return access_rate_avg / self.config.query_for_reward

    def Test3(self):
        self.network.load_state_dict(torch.load(str(self.config.data_distribution)+'_'+'k'+str(self.config.action_space)+'_testing'+'.mdl'))
        # self.network.eval()
        # self.ResetObjLoader()
        self.tree.Clear()
        self.reference_tree.Clear() #CONTINUE FROM HERE
        # object_boundary = self.NextObj()
        action_count = [0]*self.config.action_space
        obj_cnt = 0
        for i in range(len(self.model_dataset)):
            obj_cnt += 1
            self.reference_tree.DefaultInsert(self.model_dataset[i])
            self.tree.DirectInsert(self.model_dataset[i])
            if self.tree.NeedSplit():
                while True:
                    num_of_zero_ovlp_splits = self.tree.GetNumberOfNonOverlapSplitLocs()
                    if num_of_zero_ovlp_splits == None:
                        break

                    if num_of_zero_ovlp_splits <= 1:
                        self.tree.SplitInMinOverlap()
                    else:
                        states = self.tree.RetrieveZeroOVLPSplitSortedByPerimeterState()
                        # states = self.tree.RetrieveZeroOVLPSplitSortedByWeightedPerimeterState()
                        states = torch.tensor(states, dtype=torch.float32).to(self.network.device)
                        q_values = self.network(states).to(self.network.device)
                        action = torch.argmax(q_values).item()
                        # action = 0
                        action_count[action] += 1
                        self.tree.SplitWithCandidateAction(action)
        # print('RLR Tree average tree node area: ', self.tree.AverageNodeArea())
        # print('RLR Tree average tree node children: ', self.tree.AverageNodeChildren())
        # print('RLR Tree total tree nodes: ', self.tree.TotalTreeNodeNum())

        # print('Ref Tree average tree node area: ', self.reference_tree.AverageNodeArea())
        # print('Ref Tree average tree node children: ', self.reference_tree.AverageNodeChildren())
        # print('Ref Tree total tree nodes: ', self.reference_tree.TotalTreeNodeNum())

        test_result = []

        # 2% query size
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  2.0/100 * ((self.data_set_bounds[1] - self.data_set_bounds[0])*(self.data_set_bounds[3] - self.data_set_bounds[2]))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(self.data_set_bounds[0], self.data_set_bounds[1])
            y = random.uniform(self.data_set_bounds[2], self.data_set_bounds[3])
            if x - side > self.data_set_bounds[0] and y - side > self.data_set_bounds[2] and x + side < self.data_set_bounds[1] and y + side < self.data_set_bounds[3]:

                tree_access = self.tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = self.reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                k = k + 1

        manual_count = 0
        for obj in self.model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        #print(["2% query", manual_count, self.tree.QueryResult(), self.reference_tree.QueryResult()])
                
        test_result.append(["2% query", tree_acc_no, ref_tree_acc_no])

        # 1% query size
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  1.0/100 * ((self.data_set_bounds[1] - self.data_set_bounds[0])*(self.data_set_bounds[3] - self.data_set_bounds[2]))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(self.data_set_bounds[0], self.data_set_bounds[1])
            y = random.uniform(self.data_set_bounds[2], self.data_set_bounds[3])
            if x - side > self.data_set_bounds[0] and y - side > self.data_set_bounds[2] and x + side < self.data_set_bounds[1] and y + side < self.data_set_bounds[3]:

                tree_access = self.tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = self.reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                k = k + 1

        manual_count = 0
        for obj in self.model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        #print(["1% query", manual_count, self.tree.QueryResult(), self.reference_tree.QueryResult()])
                
        test_result.append(["1% query", tree_acc_no, ref_tree_acc_no])

        # 0.5% query size
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  0.5/100 * ((self.data_set_bounds[1] - self.data_set_bounds[0])*(self.data_set_bounds[3] - self.data_set_bounds[2]))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(self.data_set_bounds[0], self.data_set_bounds[1])
            y = random.uniform(self.data_set_bounds[2], self.data_set_bounds[3])
            if x - side > self.data_set_bounds[0] and y - side > self.data_set_bounds[2] and x + side < self.data_set_bounds[1] and y + side < self.data_set_bounds[3]:

                tree_access = self.tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = self.reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                k = k + 1

        manual_count = 0
        for obj in self.model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        #print(["0.5% query", manual_count, self.tree.QueryResult(), self.reference_tree.QueryResult()])
                
        test_result.append(["0.5% query", tree_acc_no, ref_tree_acc_no])

        # 0.1% query size
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  0.1/100 * ((self.data_set_bounds[1] - self.data_set_bounds[0])*(self.data_set_bounds[3] - self.data_set_bounds[2]))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(self.data_set_bounds[0], self.data_set_bounds[1])
            y = random.uniform(self.data_set_bounds[2], self.data_set_bounds[3])
            if x - side > self.data_set_bounds[0] and y - side > self.data_set_bounds[2] and x + side < self.data_set_bounds[1] and y + side < self.data_set_bounds[3]:

                tree_access = self.tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = self.reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                k = k + 1

        manual_count = 0
        for obj in self.model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        #print(["0.1% query", manual_count, self.tree.QueryResult(), self.reference_tree.QueryResult()])
                
        test_result.append(["0.1% query", tree_acc_no, ref_tree_acc_no])

        # 0.05% query size
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  0.05/100 * ((self.data_set_bounds[1] - self.data_set_bounds[0])*(self.data_set_bounds[3] - self.data_set_bounds[2]))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(self.data_set_bounds[0], self.data_set_bounds[1])
            y = random.uniform(self.data_set_bounds[2], self.data_set_bounds[3])
            if x - side > self.data_set_bounds[0] and y - side > self.data_set_bounds[2] and x + side < self.data_set_bounds[1] and y + side < self.data_set_bounds[3]:

                tree_access = self.tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = self.reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                k = k + 1

        manual_count = 0
        for obj in self.model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        #print(["0.05% query", manual_count, self.tree.QueryResult(), self.reference_tree.QueryResult()])
                
        test_result.append(["0.05% query", tree_acc_no, ref_tree_acc_no])

        # 0.01% query size
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  0.01/100 * ((self.data_set_bounds[1] - self.data_set_bounds[0])*(self.data_set_bounds[3] - self.data_set_bounds[2]))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(self.data_set_bounds[0], self.data_set_bounds[1])
            y = random.uniform(self.data_set_bounds[2], self.data_set_bounds[3])
            if x - side > self.data_set_bounds[0] and y - side > self.data_set_bounds[2] and x + side < self.data_set_bounds[1] and y + side < self.data_set_bounds[3]:

                tree_access = self.tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = self.reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                k = k + 1

        manual_count = 0
        for obj in self.model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        #print(["0.01% query", manual_count, self.tree.QueryResult(), self.reference_tree.QueryResult()])
                
        test_result.append(["0.01% query", tree_acc_no, ref_tree_acc_no])

        # 0.005% query size
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  0.005/100 * ((self.data_set_bounds[1] - self.data_set_bounds[0])*(self.data_set_bounds[3] - self.data_set_bounds[2]))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(self.data_set_bounds[0], self.data_set_bounds[1])
            y = random.uniform(self.data_set_bounds[2], self.data_set_bounds[3])
            if x - side > self.data_set_bounds[0] and y - side > self.data_set_bounds[2] and x + side < self.data_set_bounds[1] and y + side < self.data_set_bounds[3]:

                tree_access = self.tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = self.reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                k = k + 1

        manual_count = 0
        for obj in self.model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        #print(["0.005% query", manual_count, self.tree.QueryResult(), self.reference_tree.QueryResult()])
                
        test_result.append(["0.005% query", tree_acc_no, ref_tree_acc_no])

        print(test_result)
        #print("action count", action_count)

    def Test2(self):
        self.network.load_state_dict(torch.load(str(self.config.data_distribution)+'_'+'k'+str(self.config.action_space)+'.mdl'))
        # self.network.eval()
        # self.ResetObjLoader()
        self.tree.Clear()
        self.reference_tree.Clear() #CONTINUE FROM HERE
        # object_boundary = self.NextObj()
        action_count = [0]*self.config.action_space
        obj_cnt = 0
        for i in range(len(self.model_dataset)):
            obj_cnt += 1
            self.reference_tree.DefaultInsert(self.model_dataset[i])
            self.tree.DirectInsert(self.model_dataset[i])
            if self.config.network == 'spl_loc':
                states = self.tree.RetrieveSpecialSplitStates()
            elif self.config.network == 'spl_loc_short':
                states = self.tree.RetrieveShortSplitStates()
            elif self.config.network == 'sort_spl_loc':
                states = self.tree.RetrieveSortedSplitStates(self.config.action_space)
            while states is not None:
                print(i)
                states = torch.tensor(states, dtype=torch.float32).to(self.network.device)
                q_values = self.network(states).to(self.network.device)
                action = torch.argmax(q_values).item()
                action_count[action] += 1
                if self.config.network == 'sort_spl_loc':
                    self.tree.SplitWithSortedLoc(action)
                else:
                    self.tree.SplitWithLoc(action)
                if self.config.network == 'spl_loc':
                    states = self.tree.RetrieveSpecialSplitStates()
                elif self.config.network == 'spl_loc_short':
                    states = self.tree.RetrieveShortSplitStates()
                elif self.config.network == 'sort_spl_loc':
                    states = self.tree.RetrieveSortedSplitStates(self.config.action_space)
            # object_boundary = self.NextObj()
        #self.tree.PrintEntryNum()
        print('RLR Tree average tree node area: ', self.tree.AverageNodeArea())
        print('RLR Tree average tree node children: ', self.tree.AverageNodeChildren())
        print('RLR Tree total tree nodes: ', self.tree.TotalTreeNodeNum())

        print('Ref Tree average tree node area: ', self.reference_tree.AverageNodeArea())
        print('Ref Tree average tree node children: ', self.reference_tree.AverageNodeChildren())
        print('Ref Tree total tree nodes: ', self.reference_tree.TotalTreeNodeNum())

        test_result = []

        # 2% query size
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  2.0/100 * ((100000 - 0)*(100000 - 0))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(0, 100000)
            y = random.uniform(0, 100000)
            if x - side > 0 and y - side > 0 and x + side < 100000 and y + side < 100000:

                tree_access = self.tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = self.reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                k = k + 1

        manual_count = 0
        for obj in self.model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        print(["2% query", manual_count, self.tree.QueryResult(), self.reference_tree.QueryResult()])
                
        test_result.append(["2% query", tree_acc_no, ref_tree_acc_no])

        # 1% query size
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  1.0/100 * ((100000 - 0)*(100000 - 0))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(0, 100000)
            y = random.uniform(0, 100000)
            if x - side > 0 and y - side > 0 and x + side < 100000 and y + side < 100000:

                tree_access = self.tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = self.reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                k = k + 1

        manual_count = 0
        for obj in self.model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        print(["1% query", manual_count, self.tree.QueryResult(), self.reference_tree.QueryResult()])
                
        test_result.append(["1% query", tree_acc_no, ref_tree_acc_no])

        # 0.5% query size
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  0.5/100 * ((100000 - 0)*(100000 - 0))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(0, 100000)
            y = random.uniform(0, 100000)
            if x - side > 0 and y - side > 0 and x + side < 100000 and y + side < 100000:

                tree_access = self.tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = self.reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                k = k + 1

        manual_count = 0
        for obj in self.model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        print(["0.5% query", manual_count, self.tree.QueryResult(), self.reference_tree.QueryResult()])
                
        test_result.append(["0.5% query", tree_acc_no, ref_tree_acc_no])

        # 0.1% query size
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  0.1/100 * ((100000 - 0)*(100000 - 0))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(0, 100000)
            y = random.uniform(0, 100000)
            if x - side > 0 and y - side > 0 and x + side < 100000 and y + side < 100000:

                tree_access = self.tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = self.reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                k = k + 1

        manual_count = 0
        for obj in self.model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        print(["0.1% query", manual_count, self.tree.QueryResult(), self.reference_tree.QueryResult()])
                
        test_result.append(["0.1% query", tree_acc_no, ref_tree_acc_no])

        # 0.05% query size
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  0.05/100 * ((100000 - 0)*(100000 - 0))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(0, 100000)
            y = random.uniform(0, 100000)
            if x - side > 0 and y - side > 0 and x + side < 100000 and y + side < 100000:

                tree_access = self.tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = self.reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                k = k + 1

        manual_count = 0
        for obj in self.model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        print(["0.05% query", manual_count, self.tree.QueryResult(), self.reference_tree.QueryResult()])
                
        test_result.append(["0.05% query", tree_acc_no, ref_tree_acc_no])

        # 0.01% query size
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  0.01/100 * ((100000 - 0)*(100000 - 0))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(0, 100000)
            y = random.uniform(0, 100000)
            if x - side > 0 and y - side > 0 and x + side < 100000 and y + side < 100000:

                tree_access = self.tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = self.reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                k = k + 1

        manual_count = 0
        for obj in self.model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        print(["0.01% query", manual_count, self.tree.QueryResult(), self.reference_tree.QueryResult()])
                
        test_result.append(["0.01% query", tree_acc_no, ref_tree_acc_no])

        # 0.005% query size
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  0.005/100 * ((100000 - 0)*(100000 - 0))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(0, 100000)
            y = random.uniform(0, 100000)
            if x - side > 0 and y - side > 0 and x + side < 100000 and y + side < 100000:

                tree_access = self.tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = self.reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                k = k + 1

        manual_count = 0
        for obj in self.model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        print(["0.005% query", manual_count, self.tree.QueryResult(), self.reference_tree.QueryResult()])
                
        test_result.append(["0.005% query", tree_acc_no, ref_tree_acc_no])

        print(test_result)
        print("action count", action_count)

        # node_access = 0
        # query_num = 0
        # query = self.NextQuery()
        # f = open('debug.result.log', 'w')
        # while query is not None:
        #     node_access += self.tree.Query(query)
        #     f.write('{}\n'.format(self.tree.QueryResult()))
        #     query_num += 1
        #     query = self.NextQuery()
        # print('average node access is ', node_access / query_num)
        # f.close()
        # return 1.0 * node_access / query_num

    def Baseline(self):
        object_boundary = self.NextObj()
        while object_boundary is not None:
            self.tree.DefaultInsert(object_boundary)
            object_boundary = self.NextObj()

        node_access = 0
        query_num = 0

        query = self.NextQuery()
        while query is not None:
            node_access += self.tree.Query(query)
            query_num += 1
            query = self.NextQuery()
        return 1.0 * node_access / query_num

    def Train5(self):
        #with teacher forcing
        start_time = time.time()
        # loss_log = open("./log/{}.loss".format(self.id), 'w')
        # reward_log = open("./log/{}.reward".format(self.id), "w")
        steps = []
        object_num = len(self.training_dataset)
        # self.ResetObjLoader()
        # object_boundary = self.NextObj()
        cache_tree = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
        cache_tree.SetInsertStrategy(self.config.default_ins_strategy)
        cache_tree.SetSplitStrategy(self.config.default_spl_strategy)

        # while object_boundary is not None:
        #     object_num += 1
        #     object_boundary = self.NextObj()

        for epoch in range(self.config.epoch):
            e = 0
            # part_trange = trange(self.config.parts, desc='With parts')
            for part in range(self.config.parts):
                

                # Part 1: Default insert
                ratio_for_tree_construction = 1.0 * (part + 1) / (self.config.parts + 1)
                # self.ResetObjLoader()
                self.tree.Clear()
                self.reference_tree.Clear()
                cache_tree.Clear()
                for i in range(int(object_num * ratio_for_tree_construction)):
                    # object_boundary = self.NextObj()
                    self.tree.DefaultInsert(self.training_dataset[i])
#                print('{} nodes are added into the tree'.format(int(object_num * ratio_for_tree_construction)))
                # part_trange.set_description('With parts: {} nodes are added into the tree'.format(int(object_num * ratio_for_tree_construction)))
                # part_trange.refresh()

                # Part 2: identify objects that will trigger splitting to the tree from part 1
                # fo = open('train_object.tmp', 'w')
                train_obj = []
                # object_boundary = self.NextObj()
                #print('filling leaf nodes')
                objects_for_fill = 0
                objects_for_train = 0
                cnt = 0
                for i in range(int(object_num * ratio_for_tree_construction), object_num):
                    cnt += 1
                    is_success = self.tree.TryInsert(self.training_dataset[i])
                    if not is_success:
                        objects_for_train += 1
                        train_obj.append(self.training_dataset[i])
                        # fo.write('{:.5f} {:.5f} {:.5f} {:.5f}\n'.format(object_boundary[0], object_boundary[1], object_boundary[2], object_boundary[3]))
                    else:
                        objects_for_fill += 1
                    # object_boundary = self.NextObj()
                # fo.close()
                cache_tree.CopyTree(self.tree.tree)
                self.reference_tree.CopyTree(cache_tree.tree)
                self.tree.CopyTree(cache_tree.tree)

                # Part 3: Training
                # fin = open('train_object.tmp', 'r')
                period = 0
                obj_list_for_reward = []
                accum_loss = 0
                accum_loss_cnt = 0
                # split_trange = trange(objects_for_train, desc="Training", leave=False)
                # print("No. of obj to train: ", objects_for_train)
                for training_id in range(objects_for_train):
                    print(["Epoch", epoch, "Parts", part, "Training obj ID", training_id, "Total training obj", objects_for_train])
                    # line = fin.readline()
                    # object_boundary = [float(v) for v in line.strip().split()]
                    self.reference_tree.DefaultInsert(train_obj[training_id])
                    self.tree.DirectInsert(train_obj[training_id])
                    state = None
                    if self.config.network == 'spl_loc':
                        states = self.tree.RetrieveSpecialSplitStates()
                    elif self.config.network == 'spl_loc_short':
                        states = self.tree.RetrieveShortSplitStates()
                    elif self.config.network == 'sort_spl_loc': # the final one
                        states = self.tree.RetrieveSortedSplitStates(self.config.action_space)
                    trigger_split = False
                    while states is not None:
                        trigger_split = True
                        states = torch.tensor(states, dtype=torch.float32).to(self.network.device)
                        action = 0
                        if self.config.teacher_forcing is not None:
                            
                            # use teacher forcing
                            splits_with_tf = int(self.config.teacher_forcing * objects_for_train)
                            if training_id < splits_with_tf:
                                min_perimeter = 1000000000000.0
                                perimeters = np.zeros(self.config.action_space)
                                for i in range(self.config.action_space):
                                    perimeters[i] = states[i * 5 + 2] + states[i * 5 + 3]
                                perimeters = torch.tensor(perimeters, dtype=torch.float32)
                                action = np.argmin(perimeters)
                        else:
                            with torch.no_grad():
                                if epoch > 0:
                                    q_values = self.network(states).to(self.network.device)
                                    action = self.EpsilonGreedy(q_values)
                                else:
                                    action = randint(0, self.config.action_space-1)
                        if self.config.network == 'sort_spl_loc':
                            self.tree.SplitWithSortedLoc(action)
                        else:
                            self.tree.SplitWithLoc(action)
                        steps.append((states, action))
                        if self.config.network == 'spl_loc':
                            states = self.tree.RetrieveSpecialSplitStates()
                        elif self.config.network == 'spl_loc_short':
                            states = self.tree.RetrieveShortSplitStates()
                        elif self.config.network == 'sort_spl_loc':
                            states = self.tree.RetrieveSortedSplitStates(self.config.action_space)
                    
                    if trigger_split:
                        steps.append(([0]*5*self.config.action_space, None))
                    period += 1
                    obj_list_for_reward.append(train_obj[training_id])

                    if period == self.config.splits_for_update:
                        reward = self.ComputeDenseRewardForList(obj_list_for_reward)
                        # reward_log.write('{}\n'.format(reward))
                        for i in range(len(steps) - 1):
                            if steps[i][1] is None:
                                continue
                            self.memory.push(steps[i][0], steps[i][1], reward, steps[i+1][0])
                        self.reference_tree.CopyTree(cache_tree.tree)
                        self.tree.CopyTree(cache_tree.tree)
                        period = 0
                        obj_list_for_reward.clear()
                        steps.clear()
                    
                    l = self.Optimize()
                    if l is not None:
                        accum_loss += l
                        accum_loss_cnt += 1
                    # loss_log.write('{}\n'.format(l))
                    if e % 500 == 0:
                        average_loss = None
                        if accum_loss > 0:
                            average_loss = 1.0 * accum_loss / accum_loss_cnt
                        # split_trange.set_description('Training: average loss {}'.format(average_loss))
                        accum_loss = 0
                        accum_loss_cnt = 0
                        # split_trange.refresh()
                        if epoch > 0:
                            self.config.epsilon = max(self.config.epsilon * self.config.epsilon_decay, self.config.min_epsilon)
                    e += 1
                    if e % self.config.target_update == 0:
                        self.target_network.load_state_dict(self.network.state_dict())
                # fin.close()
            # torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.epoch{}'.format(epoch) + '.mdl')
        end_time = time.time()
        torch.save(self.network.state_dict(), str(self.config.data_distribution)+'_'+'k'+str(self.config.action_space)+'.mdl')
        # reward_log.close()
        # loss_log.close()
        # train_log = open('./log/train.log', 'a')
        # train_log.write('{}:\n'.format(datetime.now()))
        # train_log.write('{}\n'.format(self.id))
        # train_log.write('{}\n'.format(self.config))
        # train_log.write('training time: {}\n'.format(end_time-start_time))
        #train_log.write('zero reward: {}, zero reward2: {}\n'.format(reward_is_0, reward2_is_0))
        # train_log.close()
        self.tree.Clear()
        self.reference_tree.Clear()
        cache_tree.Clear()


    def Train6(self): # RL separate partitions with and without overlaps
        #with teacher forcing
        start_time = time.time()
        # loss_log = open("./log/{}.loss".format(self.id), 'w')
        # reward_log = open("./log/{}.reward".format(self.id), "w")
        steps = []
        object_num = len(self.training_dataset)
        # self.ResetObjLoader()
        # object_boundary = self.NextObj()
        cache_tree = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
        cache_tree.SetInsertStrategy(self.config.default_ins_strategy)
        cache_tree.SetSplitStrategy(self.config.default_spl_strategy)

        # while object_boundary is not None:
        #     object_num += 1
        #     object_boundary = self.NextObj()

        for epoch in range(self.config.epoch):
            e = 0
            # part_trange = trange(self.config.parts, desc='With parts')
            for part in range(self.config.parts):
                # Part 1: Default insert
                ratio_for_tree_construction = 1.0 * (part + 1) / (self.config.parts + 1)
                # self.ResetObjLoader()
                self.tree.Clear()
                self.reference_tree.Clear()
                cache_tree.Clear()
                for i in range(int(object_num * ratio_for_tree_construction)):
                    # object_boundary = self.NextObj()
                    self.tree.DefaultInsert(self.training_dataset[i])
#                print('{} nodes are added into the tree'.format(int(object_num * ratio_for_tree_construction)))
                # part_trange.set_description('With parts: {} nodes are added into the tree'.format(int(object_num * ratio_for_tree_construction)))
                # part_trange.refresh()

                # Part 2: identify objects that will trigger splitting to the tree from part 1
                # fo = open('train_object.tmp', 'w')
                train_obj = []
                # object_boundary = self.NextObj()
                #print('filling leaf nodes')
                objects_for_fill = 0
                objects_for_train = 0
                cnt = 0
                for i in range(int(object_num * ratio_for_tree_construction), object_num):
                    cnt += 1
                    is_success = self.tree.TryInsert(self.training_dataset[i])
                    if not is_success:
                        objects_for_train += 1
                        train_obj.append(self.training_dataset[i])
                        # fo.write('{:.5f} {:.5f} {:.5f} {:.5f}\n'.format(object_boundary[0], object_boundary[1], object_boundary[2], object_boundary[3]))
                    else:
                        objects_for_fill += 1
                    # object_boundary = self.NextObj()
                # fo.close()
                cache_tree.CopyTree(self.tree.tree)
                self.reference_tree.CopyTree(cache_tree.tree)
                self.tree.CopyTree(cache_tree.tree)

                # Part 3: Training
                # fin = open('train_object.tmp', 'r')
                period = 0
                obj_list_for_reward = []
                accum_loss = 0
                accum_loss_cnt = 0
                # split_trange = trange(objects_for_train, desc="Training", leave=False)
                # print("No. of obj to train: ", objects_for_train)
                for training_id in range(objects_for_train):
                    print(["Epoch", epoch, "Parts", part, "Training obj ID", training_id, "Total training obj", objects_for_train])
                    # line = fin.readline()
                    # object_boundary = [float(v) for v in line.strip().split()]
                    self.reference_tree.DefaultInsert(train_obj[training_id])
                    self.tree.DirectInsert(train_obj[training_id])
                    triggered = False
                    if self.tree.NeedSplit():
                        triggered = True
                        enter_loop = False
                        while True:
                            num_of_zero_ovlp_splits = self.tree.GetNumberOfNonOverlapSplitLocs()
                            if num_of_zero_ovlp_splits == None:
                                break

                            if num_of_zero_ovlp_splits <= 1:
                                self.tree.SplitInMinOverlap()
                                if enter_loop:
                                    steps.append(([0]*4*self.config.action_space, None))
                                    enter_loop = False
                            else:
                                enter_loop = True
                                states = self.tree.RetrieveZeroOVLPSplitSortedByPerimeterState()
                                # states = self.tree.RetrieveZeroOVLPSplitSortedByWeightedPerimeterState()
                                states = torch.tensor(states, dtype=torch.float32).to(self.network.device)
                                action = -1

                                if self.config.teacher_forcing is not None:
                                    splits_with_tf = int(self.config.teacher_forcing * objects_for_train)
                                    if training_id < splits_with_tf:
                                        min_perimeter = 1000000000000.0
                                        perimeters = np.zeros(self.config.action_space)
                                        for i in range(self.config.action_space):
                                            perimeters[i] = states[i * 4 + 2] + states[i * 4 + 3]
                                        perimeters = torch.tensor(perimeters, dtype=torch.float32)
                                        action = np.argmin(perimeters)
                                if action == -1:
                                    with torch.no_grad():
                                        q_values = self.network(states).to(self.network.device)
                                        action = self.EpsilonGreedy(q_values)
                                        # action = randint(0, self.config.action_space-1)
                                self.tree.SplitWithCandidateAction(action)
                                steps.append((states, action))
                    if triggered and len(steps) > 0 and steps[-1][1] is not None:
                        steps.append(([0]*4*self.config.action_space, None))
                    period += 1
                    obj_list_for_reward.append(train_obj[training_id])

                    if period == self.config.splits_for_update:
                        reward = self.ComputeDenseRewardForList(obj_list_for_reward)
                        # reward_log.write('{}\n'.format(reward))
                        for i in range(len(steps) - 1):
                            if steps[i][1] is None:
                                continue
                            self.memory.push(steps[i][0], steps[i][1], reward, steps[i+1][0])
                        self.reference_tree.CopyTree(cache_tree.tree)
                        self.tree.CopyTree(cache_tree.tree)
                        period = 0
                        obj_list_for_reward.clear()
                        steps.clear()
                    
                    l = self.Optimize()
                    if l is not None:
                        accum_loss += l
                        accum_loss_cnt += 1
                    # loss_log.write('{}\n'.format(l))
                    if e % 500 == 0:
                        average_loss = None
                        if accum_loss > 0:
                            average_loss = 1.0 * accum_loss / accum_loss_cnt
                        # split_trange.set_description('Training: average loss {}'.format(average_loss))
                        accum_loss = 0
                        accum_loss_cnt = 0
                        # split_trange.refresh()
                        if epoch > 0:
                            self.config.epsilon = max(self.config.epsilon * self.config.epsilon_decay, self.config.min_epsilon)
                    e += 1
                    if e % self.config.target_update == 0:
                        self.target_network.load_state_dict(self.network.state_dict())
                # fin.close()
            # torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.epoch{}'.format(epoch) + '.mdl')
        end_time = time.time()
        torch.save(self.network.state_dict(), str(self.config.data_distribution)+'_'+'k'+str(self.config.action_space)+'_testing'+'.mdl')
        # reward_log.close()
        # loss_log.close()
        # train_log = open('./log/train.log', 'a')
        # train_log.write('{}:\n'.format(datetime.now()))
        # train_log.write('{}\n'.format(self.id))
        # train_log.write('{}\n'.format(self.config))
        # train_log.write('training time: {}\n'.format(end_time-start_time))
        #train_log.write('zero reward: {}, zero reward2: {}\n'.format(reward_is_0, reward2_is_0))
        # train_log.close()
        self.tree.Clear()
        self.reference_tree.Clear()
        cache_tree.Clear()

if __name__ == '__main__':
    args = parser.parse_args()
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.action == 'train':
        spl_learner = SplitLearner()
        spl_learner.Initialize(args)
        
        spl_learner.training_dataset = model_dataset[:int(args.training_set_size)]
        if args.data_distribution in ['china','india']:
            spl_learner.model_dataset = model_dataset
        elif args.data_distribution in ['uniform','skew','gaussian']:
            spl_learner.model_dataset = model_dataset[:int(args.data_set_size)]

        spl_learner.data_set_bounds = [x_min, x_max, y_min, y_max]
        spl_learner.Train6()
        spl_learner.Test3()
    if args.action == 'test':
        spl_learner = SplitLearner()
        spl_learner.Initialize(args)

        spl_learner.training_dataset = model_dataset[:int(args.training_set_size)]
        if args.data_distribution in ['china','india']:
            spl_learner.model_dataset = model_dataset
        elif args.data_distribution in ['uniform','skew','gaussian']:
            spl_learner.model_dataset = model_dataset[:int(args.data_set_size)]

        spl_learner.data_set_bounds = [x_min, x_max, y_min, y_max]
        spl_learner.Test3()
