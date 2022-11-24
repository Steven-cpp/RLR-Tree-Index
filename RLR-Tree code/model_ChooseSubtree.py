import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as f
import torch.multiprocessing as mp
import torch.optim as optim
from torch.autograd import Variable

import hashlib

import random
from random import randint
import math
import argparse
import pickle
from collections import namedtuple
from itertools import count
import os
import time
from datetime import datetime

from RTree import RTree

parser = argparse.ArgumentParser()

parser.add_argument('-process', choices=["training", "testing"], default="training")
parser.add_argument('-objfile', help='data file')
parser.add_argument('-queryfile', help='query file')
parser.add_argument('-epoch', type=int, help='number of epoches', default=20)
parser.add_argument('-reward_query_width', type=float, default=0.01)
parser.add_argument('-reward_query_height', type=float, default=0.01)
parser.add_argument('-default_ins_strategy', help='default insert strategy', default="INS_AREA")
parser.add_argument('-default_spl_strategy', help='default split strategy', default='SPL_MIN_AREA')
parser.add_argument('-reference_tree_ins_strategy', help='default insert strategy for reference tree', default="INS_AREA")
parser.add_argument('-reference_tree_spl_strategy', help='default split strategy for reference tree', default='SPL_MIN_AREA')
parser.add_argument('-action_space', type=int, help='number of possible actions', default=5)
parser.add_argument('-batch_size', type=int, help='batch_size', default=64)
parser.add_argument('-state_dim', type=int, help='input dimension', default=25)
parser.add_argument('-inter_dim', type=int, help='internal dimension', default=32)
parser.add_argument('-memory_cap', type=int, help='memory capacity', default=5000)
parser.add_argument('-lr', type=float, help='learning rate', default=0.01)
parser.add_argument('-gamma', type=float, help='reward discount factor', default=0.8)
parser.add_argument('-model_name', help='name of the model')
parser.add_argument('-target_update', type=int, help='update the parameters for target network every ? steps', default=30)
parser.add_argument('-epsilon', type=float, help='epsilon greedy', default=0.9)
parser.add_argument('-epsilon_decay', type=float, help='how fast to decrease epsilon', default=0.99)
parser.add_argument('-min_epsilon', type=float, help='minimum epsilon', default=0.1)
parser.add_argument('-max_entry', type=int, help='maximum entry a node can hold', default=50)
parser.add_argument('-query_for_reward', type=int, help='number of query used for reward', default=5)
parser.add_argument('-splits_for_update', type=int, help='number of splits for a reward computation', default=20)
parser.add_argument('-parts', type=int, help='number of parts to train', default=5)
parser.add_argument('-network', choices=['strategy', 'spl_loc'], help='which network is used for training', default='strategy')
# below options are specially for insertion
parser.add_argument('-data_distribution', choices=['uniform', 'skew', 'gaussian', 'china', 'india'], help='data set distribution', default='gaussian')
parser.add_argument('-training_data_distribution', choices=['nil', 'uniform'], help='training data set distribution if different', default='nil')
parser.add_argument('-data_set_size', type=int, help='data set size', default=20000000)
parser.add_argument('-training_set_size', type=int, help='training set size', default=100000)
parser.add_argument('-reward_comp_freq', type=int, help='insertion reward computation frequency', default=10)
parser.add_argument('-action_space_size', type=int, help='action space size for top k child nodes', default=2)
parser.add_argument('-rl_method', choices=[0,1], help='0: RL for enlargement; 1: RL for no enlargement', default=0)
parser.add_argument('-model_number', type=int, help='which insertion training model', default=10)
parser.add_argument('-reference_tree_type', choices=['rtree', 'rrstar'], help='which reference tree to use', default='rtree')

# one_layer_nn and Agent are for insertion
class one_layer_nn(nn.Module):
    
    def __init__(self, ALPHA, input_size, hidden_size1, output_size):
        super(one_layer_nn , self).__init__()
        
        self.layer1 = nn.Linear(  input_size  , hidden_size1,   bias=False  )
        self.layer2 = nn.Linear(  hidden_size1, output_size,    bias=False  )
        
        self.optimizer = optim.RMSprop(self.parameters(), lr = ALPHA)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        
        # sm      = nn.Softmax(dim=0)
        sm      = nn.SELU()

        # Variable is used for older torch
        x = Variable(x, requires_grad=False)
        y       = self.layer1(x)
        y_hat   = sm(y)
        z       = self.layer2(y_hat)
        scores  = z
        #scores  = F.relu(y)
        
        return scores

class Agent():
    
    def __init__(self, gamma, epsilon, alpha, maxMemorySize, batch_size, action_space_size, epsEnd=0.1, 
                 replace=20):
        # epsEnd was set to be 0.05 initially
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.memSize = maxMemorySize
        self.batch_size = batch_size
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCntr = 0
        self.replace_target_cnt = replace
        self.action_space_size = action_space_size
        self.Q_eval = one_layer_nn(alpha, action_space_size*4, 64, action_space_size)
        self.Q_next = one_layer_nn(alpha, action_space_size*4, 64, action_space_size)
        self.state_action_pairs = []
        
    def storeTransition(self, state, action, reward, state_):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCntr%self.memSize] = [state, action, reward, state_]
        self.memCntr = self.memCntr + 1

    def sample(self):
        if len(self.memory) < self.batch_size:
            return self.memory
        else:
            return random.sample(self.memory, self.batch_size)
    
    def chooseAction(self, observation):
        observation = torch.FloatTensor(observation).to(self.Q_eval.device)
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        if rand < 1 - self.EPSILON:
            # action = torch.argmax(actions).item()
            useless_value, action = torch.max(actions[:self.action_space_size], 0)
            
        else:
            # action = np.random.choice(self.actionSpace)
            ### Need to change this part!!! based on the 4 sorts of child nodes!!!
            action = randint(0, self.action_space_size-1)
        self.steps = self.steps + 1
        return int(action)
            
    def learn(self):        
        if len(self.memory) > 0:
            self.Q_eval.optimizer.zero_grad()
            if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
                self.Q_next.load_state_dict(self.Q_eval.state_dict())
                
            list0 = []
            list1 = []
            list2 = []
            list3 = []
            for i in self.sample():
                list0.append(list(i[0]))
                list1.append(i[1])
                list2.append(i[2])
                list3.append(list(i[3]))
            Qpred = self.Q_eval.forward(torch.FloatTensor(list0).to(self.Q_eval.device))
            Qnext = self.Q_next.forward(torch.FloatTensor(list3).to(self.Q_eval.device))
            
            # maxA = torch.argmax(Qnext, dim=0)
            max_value, maxA = torch.max(Qnext, 1)
            actions = torch.tensor(list1, dtype=torch.int64)
            actions = Variable(actions, requires_grad=False)
            rewards = torch.FloatTensor(list2)
            rewards = Variable(rewards, requires_grad=False)
            Qtarget = Qpred.clone()

            for i in range(len(maxA)):
                temp = rewards[i] + self.GAMMA*(max_value[i])
                Qtarget[i, actions[i]] = temp
                #Qtarget[i, int(maxA[i])] = temp
            # Qtarget[:,maxA] = rewards + self.GAMMA*(T.max(Qnext,1)[0])
            
            if self.steps > 400000:
                if self.EPSILON > self.EPS_END:
                    self.EPSILON = self.EPSILON * 0.99
                else:
                    self.EPSILON = self.EPS_END

            # be careful of the input format
            loss = self.Q_eval.loss(Qpred, Qtarget.detach()).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()
            self.learn_step_counter = self.learn_step_counter + 1

MAXCHILDREN=50
minimum_split_occupancy = int(0.4 * MAXCHILDREN)
maximum_split_occupancy = int(0.6 * MAXCHILDREN + 1)

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
                #sf = nn.Softmax(dim=0)
                return x

class DQN2(nn.Module):
        def __init__(self, input_dimension=240, inter_dimension=300, output_dimension=48):
                super(DQN2, self).__init__()
                self.linear1 = nn.Linear(input_dimension, inter_dimension)
                self.linear2 = nn.Linear(inter_dimension, output_dimension)

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
                md5content = "{}".format(datetime.now())
                self.id = hashlib.md5(md5content.encode()).hexdigest()

        def Initialize(self, config):
                self.config = config
                if config.objfile:
                        try:
                                self.obj_input = open(config.objfile, 'r')
                        except:
                                print('object file does not exist.')
                if config.queryfile:
                        try:
                                self.query_input = open(config.queryfile, 'r')
                        except:
                                print('query file does not exist.')

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
                        return torch.tensor(random.randint(0, self.config.action_space - 1), dtype=torch.int64)
                else:
                        return np.argmax(q_values)

        def Optimize(self):
                if len(self.memory) < self.config.batch_size:
                        return

                transitions = self.memory.sample(self.config.batch_size)
                batch = Transition(*zip(*transitions))

                state_batch = torch.stack(batch.state)
                reward_batch = torch.stack(batch.reward)
                action_batch = torch.unsqueeze(torch.stack(batch.action), 1)
                

                real_q_values = self.network(state_batch)

                state_action_values = torch.gather(real_q_values, 1, action_batch)

                mask = []
                non_final_next_state = []
                for s in batch.next_state:
                        if s is not None:
                                mask.append(1)
                                non_final_next_state.append(s)
                        else:
                                mask.append(0)
                next_state_values = torch.zeros(self.config.batch_size, 1)
                if non_final_next_state:
                        next_state_mask = torch.nonzero(torch.tensor(mask, dtype=torch.int64)).squeeze(1)
                        next_state_batch = torch.stack(non_final_next_state)    
                        y, _ = self.target_network(next_state_batch).max(1, keepdim=True)
                        next_state_values[next_state_mask] = y
                expected_state_action_values = reward_batch + (next_state_values * self.config.gamma)

                output = self.loss(state_action_values, expected_state_action_values)
                l = output.item()
                self.optimizer.zero_grad()
                output.backward()
                for param in self.network.parameters():
                        param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
                return l

        def ComputeReward(self):
                access_rate_avg = 0
                for i in range(self.config.query_for_reward):
                        query = self.tree.UniformRandomQuery(self.config.reward_query_width, self.config.reward_query_height)
                        access_rate_avg += self.reference_tree.AccessRate(query) - self.tree.AccessRate(query)
                return access_rate_avg / self.config.query_for_reward

        def ComputeDenseRewardForList(self, obj_list):
                access_rate_avg = 0
                for obj in obj_list:
                        for i in range(5):
                                query = self.tree.UniformDenseRandomQuery(self.config.reward_query_width, self.config.reward_query_height, obj)
                                #print(query)
                                reference_rate = self.reference_tree.AccessRate(query)
                                #print(reference_rate)
                                tree_rate = self.tree.AccessRate(query)
                                #print(tree_rate)
                                access_rate_avg += reference_rate - tree_rate
                                #print(access_rate_avg)
                return access_rate_avg / len(obj_list) / 5

        def ComputeDenseReward(self, object_boundary):
                access_rate_avg = 0
                for i in range(self.config.query_for_reward):
                        query = self.tree.UniformDenseRandomQuery(self.config.reward_query_width, self.config.reward_query_height, object_boundary)
                        access_rate_avg += self.reference_tree.AccessRate(query) - self.tree.AccessRate(query)
                return access_rate_avg / self.config.query_for_reward


        def Test10(self):
                self.network.load_state_dict(torch.load("./model/"+self.config.model_name+".mdl"))
                self.network.eval()
                node_access = 0
                query_num = 0
                for i in range(10):
                        self.tree.Clear()
                        name = "./dataset/{}.test{}.txt".format(self.config.objfile, i)
                        ofin = open(name, 'r')
                        for line in ofin:
                                object_boundary = [float(e) for e in line.strip().split()]
                                self.tree.DirectInsert(object_boundary)
                                states, _ = self.tree.RetrieveSplitStates()
                                while states is not None:
                                        states = torch.tensor(states, dtype=torch.float32)
                                        q_values = self.network(states)
                                        action = torch.argmax(q_values).item()
                                        self.tree.SplitOneStep(action)
                                        states, _ = self.tree.RetrieveSplitStates()
                        ofin.close()
                        query = self.NextQuery()
                        while query is not None:
                                node_access += self.tree.Query(query)
                                query_num += 1
                                query = self.NextQuery()
                        self.ResetQueryLoader()
                print('average node access is ', node_access / query_num)

        def Test10_2(self):
                self.network.load_state_dict(torch.load("./model/"+self.config.model_name+".mdl"))
                self.network.eval()
                node_access = 0
                query_num = 0
                for i in range(10):
                        self.tree.Clear()
                        name = "./dataset/{}.test{}.txt".format(self.config.objfile, i)
                        ofin = open(name, 'r')
                        for line in ofin:
                                object_boundary = [float(e) for e in line.strip().split()]
                                self.tree.DirectInsert(object_boundary)
                                states = self.tree.RetrieveSpecialSplitStates()
                                while states is not None:
                                        states = torch.tensor(states, dtype=torch.float32)
                                        q_values = self.network(states)
                                        action = torch.argmax(q_values).item()
                                        self.tree.SplitWithLoc(action)
                                        states = self.tree.RetrieveSpecialSplitStates()
                        ofin.close()
                        query = self.NextQuery()
                        while query is not None:
                                node_access += self.tree.Query(query)
                                query_num += 1
                                query = self.NextQuery()
                        self.ResetQueryLoader()
                print('average node access is ', node_access / query_num)

        def Test2(self):
                self.network.load_state_dict(torch.load("./model/"+self.config.model_name+".mdl"))
                self.network.eval()
                self.ResetObjLoader()
                self.tree.Clear()
                object_boundary = self.NextObj()
                obj_cnt = 0
                while object_boundary is not None:
                        obj_cnt += 1
                        self.tree.DirectInsert(object_boundary)
                        states = self.tree.RetrieveSpecialSplitStates()
                        while states is not None:
                                states = torch.tensor(states, dtype=torch.float32)
                                q_values = self.network(states)
                                action = torch.argmax(q_values).item()
                                self.tree.SplitWithLoc(action)
                                states = self.tree.RetrieveSpecialSplitStates()
                        object_boundary = self.NextObj()
                #self.tree.PrintEntryNum()
                node_access = 0
                query_num = 0
                query = self.NextQuery()
                f = open('debug.result.log', 'w')
                while query is not None:
                        node_access += self.tree.Query(query)
                        f.write('{}\n'.format(self.tree.QueryResult()))
                        query_num += 1
                        query = self.NextQuery()
                print('average node access is ', node_access / query_num)
                f.close()
                return 1.0 * node_access / query_num

        def Test(self):
                self.network.load_state_dict(torch.load("./model/"+self.config.model_name+".mdl"))
                self.network.eval()
                self.ResetObjLoader()
                self.tree.Clear()
                debug = False
                object_boundary = self.NextObj()
                obj_cnt = 0
                self.tree.debug = False
                while object_boundary is not None:
                        obj_cnt += 1
                        #if obj_cnt > 92300:
                                #debug = True
                        #if obj_cnt % 100 == 0:
                                #print(obj_cnt)
                        if debug:
                                print('to insert', obj_cnt)
                        self.tree.DirectInsert(object_boundary)
                        if debug:
                                print('inserted', obj_cnt)
                        #print('insert')
                        states, _ = self.tree.RetrieveSplitStates()
                        #if debug:
                        #print(states)
                        while states is not None:
                                states = torch.tensor(states, dtype=torch.float32)
                                #print('states', states)
                                #input()
                                q_values = self.network(states)
                                #print('qvalues', qvalues)
                                #input()
                                action = torch.argmax(q_values).item()
                                #print('action', action)
                                #input()
                                #if(debug):
                                #print("to split")
                                self.tree.SplitOneStep(action)
                                #if(debug):
                                #       print('splitted')
                                states, _ = self.tree.RetrieveSplitStates()
                        object_boundary = self.NextObj()
                self.tree.PrintEntryNum()

                node_access = 0
                query_num = 0
                query = self.NextQuery()
                f = open('debug.result.log', 'w')
                while query is not None:
                        node_access += self.tree.Query(query)
                        f.write('{}\n'.format(self.tree.QueryResult()))
                        query_num += 1
                        query = self.NextQuery()
                print('average node access is ', node_access / query_num)
                f.close()
                return 1.0 * node_access / query_num

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

        def Train4(self):
                start_time = time.time()
                loss_log = open("./log/{}.loss".format(self.id), 'w')
                reward_log = open("./log/{}.reward".format(self.id), "w")
                steps = []
                object_num = 0
                self.ResetObjLoader()
                object_boundary = self.NextObj()
                cache_tree = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
                cache_tree.SetInsertStrategy(self.config.default_ins_strategy)
                cache_tree.SetSplitStrategy(self.config.default_spl_strategy)

                while object_boundary is not None:
                        object_num += 1
                        object_boundary = self.NextObj()
                objects_for_train = 0
                for epoch in range(self.config.epoch):
                        e = 0
                        self.ResetObjLoader()
                        self.tree.Clear()
                        self.reference_tree.Clear()
                        print("setup initial tree")
                        ratio_for_tree_construction = epoch % self.config.parts + 1
                        for i in range(int(object_num * ratio_for_tree_construction / (self.config.parts + 1))):
                                object_boundary = self.NextObj()
                                self.tree.DefaultInsert(object_boundary)
                        fo = open('train_object.tmp', 'w')
                        object_boundary = self.NextObj()

                        print('filling leaf nodes')
                        #fill the r-tree so every leaf is full
                        objects_for_fill = 0
                        cnt = 0
                        while object_boundary is not None:
                                if cnt % 100 == 0:
                                        print(cnt)
                                cnt += 1
                                is_success = self.tree.TryInsert(object_boundary)
                                if not is_success:
                                        fo.write('{:.5f} {:.5f} {:.5f} {:.5f}\n'.format(object_boundary[0], object_boundary[1], object_boundary[2], object_boundary[3]))
                                else:
                                        objects_for_fill += 1
                                object_boundary = self.NextObj()
                        fo.close()
                        cache_tree.CopyTree(self.tree.tree)
                        self.reference_tree.CopyTree(cache_tree.tree)
                        self.tree.CopyTree(cache_tree.tree)


                        fin = open('train_object.tmp', 'r')
                        period = 0
                        obj_list_for_reward = []
                        for line in fin:
                                objects_for_train += 1
                                object_boundary = [float(v) for v in line.strip().split()]
                                #print('object', object_boundary)
                                self.reference_tree.DefaultInsert(object_boundary)
                                self.tree.DirectInsert(object_boundary)
                                states = self.tree.RetrieveSpecialSplitStates()
                                triggered = False
                                while states is not None:
                                        triggered = True
                                        states = torch.tensor(states, dtype=torch.float32)
                                        with torch.no_grad():
                                                q_values = self.network(states)
                                                action = self.EpsilonGreedy(q_values)
                                                self.tree.SplitWithLoc(action)
                                                steps.append((states, action))
                                                states = self.tree.RetrieveSpecialSplitStates()


                                if triggered:
                                        steps.append((None, None))
                                period += 1
                                obj_list_for_reward.append(object_boundary)

                                if period == self.config.splits_for_update:
                                        reward = self.ComputeDenseRewardForList(obj_list_for_reward)
                                        #print('reward', reward)
                                        reward_log.write('{}\n'.format(reward))
                                        for i in range(len(steps) - 1):
                                                if steps[i][1] is None:
                                                        continue
                                                self.memory.push(steps[i][0], steps[i][1], torch.tensor([reward]), steps[i+1][0])
                                        self.reference_tree.CopyTree(cache_tree.tree)
                                        self.tree.CopyTree(cache_tree.tree)
                                        period = 0
                                        obj_list_for_reward.clear()
                                        steps.clear()


                                l = self.Optimize()
                                loss_log.write('{}\n'.format(l))
                                if e % 500 == 0:
                                        print('{} objects added, loss is {}\n'.format(e, l))
                                        self.config.epsilon = max(self.config.epsilon * self.config.epsilon_decay, self.config.min_epsilon)
                                e += 1

                                if e % self.config.target_update == 0:
                                        self.target_network.load_state_dict(self.network.state_dict())
                        torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.epoch{}'.format(epoch)+'.mdl')
                        fin.close()
                end_time = time.time()
                torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.'+self.id+'.mdl')
                reward_log.close()
                loss_log.close()
                train_log = open('./log/train.log', 'a')
                train_log.write('{}:\n'.format(datetime.now()))
                train_log.write('{}\n'.format(self.id))
                train_log.write('{}\n'.format(self.config))
                train_log.write('training time: {}\n'.format(end_time-start_time))
                #train_log.write('zero reward: {}, zero reward2: {}\n'.format(reward_is_0, reward2_is_0))
                train_log.close()
                self.tree.Clear()
                self.reference_tree.Clear()
                cache_tree.Clear()


        def Train3(self):
                #Use the R-tree with full leaf nodes to train
                start_time = time.time()
                loss_log = open("./log/{}.loss".format(self.config.model_name), 'w')
                debug_log = open('./reward.log', 'w')
                split_triggered = 0
                useful_split = 0
                unuseful_split = 0
                reward_is_0 = 0
                reward2_is_0 = 0
                steps = []
                object_num = 0
                self.ResetObjLoader()
                object_boundary = self.NextObj()
                cache_tree = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
                cache_tree.SetInsertStrategy(self.config.default_ins_strategy)
                cache_tree.SetSplitStrategy(self.config.default_spl_strategy)
                while object_boundary is not None:
                        object_num += 1
                        object_boundary = self.NextObj()
                objects_for_train = 0
                for epoch in range(self.config.epoch):
                        e = 0
                        self.ResetObjLoader()
                        self.tree.Clear()
                        self.reference_tree.Clear()
                        #construct r-tree with 1/4 datasets
                        print("setup initial rtree")
                        ratio_for_tree_construction = epoch % self.config.parts + 1
                        for i in range(int(object_num * ratio_for_tree_construction / (self.config.parts+1))):
                                object_boundary = self.NextObj()
                                self.tree.DefaultInsert(object_boundary)
                        fo = open('train_object.tmp', 'w')
                        object_boundary = self.NextObj()

                        print('filling leaf nodes')
                        #fill the r-tree so every leaf is full
                        objects_for_fill = 0
                        cnt = 0
                        while object_boundary is not None:
                                if cnt % 100 == 0:
                                        print(cnt)
                                cnt += 1
                                is_success = self.tree.TryInsert(object_boundary)
                                if not is_success:
                                        fo.write('{:.5f} {:.5f} {:.5f} {:.5f}\n'.format(object_boundary[0], object_boundary[1], object_boundary[2], object_boundary[3]))
                                else:
                                        objects_for_fill += 1
                                object_boundary = self.NextObj()
                        fo.close()
                        #print(objects_for_fill, 'objects are used to fill leaf nodes')
                        #start train
                        
                        cache_tree.CopyTree(self.tree.tree)
                        self.reference_tree.CopyTree(cache_tree.tree)
                        self.tree.CopyTree(cache_tree.tree)

                        fin = open('train_object.tmp', 'r')
                        period = 0
                        obj_list_for_reward = []
                        for line in fin:
                                objects_for_train += 1
                                object_boundary = [float(v) for v in line.strip().split()]
                                #print('object', object_boundary)
                                self.reference_tree.DefaultInsert(object_boundary)
                                self.tree.DirectInsert(object_boundary)
                                states, is_valid = self.tree.RetrieveSplitStates()
                                triggered = False
                                while states is not None:
                                        triggered = True
                                        states = torch.tensor(states, dtype=torch.float32)
                                        with torch.no_grad():
                                                q_values = self.network(states)
                                                action = self.EpsilonGreedy(q_values)
                                                self.tree.SplitOneStep(action)
                                                steps.append((states, action, is_valid))
                                                states, is_valid = self.tree.RetrieveSplitStates()

                                if triggered:
                                        steps.append((None, None, False))
                                        split_triggered += 1

                                period += 1
                                obj_list_for_reward.append(object_boundary)

                                if period == self.config.splits_for_update:
                                        reward = self.ComputeDenseRewardForList(obj_list_for_reward)
                                        #print('reward', reward)
                                        debug_log.write('{}\n'.format(reward))
                                        for i in range(len(steps) - 1):
                                                if steps[i][2]:
                                                        useful_split += 1
                                                        self.memory.push(steps[i][0], steps[i][1], torch.tensor([reward]), steps[i+1][0])
                                                else:
                                                        unuseful_split += 1
                                        self.reference_tree.CopyTree(cache_tree.tree)
                                        self.tree.CopyTree(cache_tree.tree)
                                        period = 0
                                        obj_list_for_reward.clear()
                                        steps.clear()


                                
                                if period % 5 == 0:
                                        l = self.Optimize()
                                        loss_log.write('{}\n'.format(l))
                                        if e % 500 == 0:
                                                print('{} objects added, loss is {}\n'.format(e, l))
                                                self.config.epsilon = max(self.config.epsilon * self.config.epsilon_decay, self.config.min_epsilon)
                                        e += 1

                                        if e % self.config.target_update == 0:
                                                self.target_network.load_state_dict(self.network.state_dict())
                        torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.epoch{}'.format(epoch)+'.mdl')
                        fin.close()
                end_time = time.time()
                torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.mdl')
                print('{} objects used for training, {} cause splits\n'.format(objects_for_train, split_triggered))
                debug_log.close()
                loss_log.close()
                train_log = open('./log/train.log', 'a')
                train_log.write('{}:\n'.format(datetime.now()))
                train_log.write('{}\n'.format(self.id))
                train_log.write('{}\n'.format(self.config))
                train_log.write('training time: {}, {} objects triggered splitting, {} useful split, {} unuseful split\n'.format(end_time-start_time, split_triggered, useful_split, unuseful_split))
                train_log.write('zero reward: {}, zero reward2: {}\n'.format(reward_is_0, reward2_is_0))
                train_log.close()
                self.tree.Clear()
                self.reference_tree.Clear()
                cache_tree.Clear()



        def Train2(self):
                #first construct R-tree with 2/3 of the dataset. Train the agent with the remaining objects so that the reward should be close.
                start_time = time.time()
                loss_log = open("./log/{}.loss".format(self.config.model_name), 'w')
                debug_log = open('./reward.log', 'w')
                split_triggered = 0
                useful_split = 0
                unuseful_split = 0
                reward_is_0 = 0
                reward2_is_0 = 0
                steps = []
                object_num = 0
                self.ResetObjLoader()
                object_boundary = self.NextObj()
                while object_boundary is not None:
                        object_num += 1
                        object_boundary = self.NextObj()

                for epoch in range(self.config.epoch):
                        e = 0
                        self.ResetObjLoader()
                        self.tree.Clear()
                        self.reference_tree.Clear()

                        for i in range(int(object_num / 3 * 2)):
                                object_boundary = self.NextObj()
                                self.tree.DefaultInsert(object_boundary)
                        self.reference_tree.CopyTree(self.tree.tree)

                        object_boundary = self.NextObj()
                        trigger_period = 0
                        while object_boundary is not None:

                                #self.reference_tree.CopyTree(self.tree.tree)
                                self.reference_tree.DefaultInsert(object_boundary)
                                self.tree.DirectInsert(object_boundary)
                                states, is_valid = self.tree.RetrieveSplitStates()
                                triggered = False
                                while states is not None:
                                        triggered = True
                                        states = torch.tensor(states, dtype=torch.float32)
                                        with torch.no_grad():
                                                q_values = self.network(states)
                                        action = self.EpsilonGreedy(q_values)
                                        self.tree.SplitOneStep(action)
                                        steps.append((states, action, is_valid))
                                        states, is_valid = self.tree.RetrieveSplitStates()

                                if triggered:
                                        split_triggered += 1
                                        trigger_period += 1
                                        #print('triggered', trigger_period, split_triggered)
                                        steps.append((None, None, False))

                                if trigger_period == self.config.splits_for_update:
                                        reward = self.ComputeReward()
                                        debug_log.write('{}\n'.format(reward))
                                        for i in range(len(steps) - 1):
                                                if steps[i][2]:
                                                        useful_split += 1
                                                        self.memory.push(steps[i][0], steps[i][1], torch.tensor([reward]), steps[i+1][0])
                                                else:
                                                        unuseful_split += 1
                                        self.reference_tree.CopyTree(self.tree.tree)
                                        trigger_period = 0


                                l = self.Optimize()
                                loss_log.write('{}\n'.format(l))
                                if e % 100 == 0:
                                        print('{} objects added, loss is {}\n'.format(e, l))
                                        self.config.epsilon = max(self.config.epsilon * self.config.epsilon_decay, self.config.min_epsilon)
                                e += 1

                                object_boundary = self.NextObj()

                                if e % self.config.target_update == 0:
                                        self.target_network.load_state_dict(self.network.state_dict())

                        torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.epoch{}'.format(epoch)+'.mdl')

                torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.mdl')
                end_time = time.time()
                train_log = open('./log/train.log', 'a')
                train_log.write('{}:\n'.format(datetime.now()))
                train_log.write('{}\n'.format(self.id))
                train_log.write('{}\n'.format(self.config))
                train_log.write('training time: {}, {} objects triggered splitting, {} useful split, {} unuseful split\n'.format(end_time-start_time, split_triggered, useful_split, unuseful_split))
                train_log.write('zero reward: {}, zero reward2: {}\n'.format(reward_is_0, reward2_is_0))
                train_log.close()
                loss_log.close()
                debug_log.close()

        def Train(self):
                start_time = time.time()
                loss_log = open("./log/{}.loss".format(self.config.model_name), 'w')
                debug_log = open('./reward.log', 'w')
                split_triggered = 0
                useful_split = 0
                unuseful_split = 0
                reward_is_0 = 0
                reward2_is_0 = 0
                steps = []
                for epoch in range(self.config.epoch):
                        e = 0
                        self.ResetObjLoader()
                        self.tree.Clear()
                        self.reference_tree.Clear()
                        object_boundary = self.NextObj()
                        trigger_period = 0
                        while object_boundary is not None:

                                #self.reference_tree.CopyTree(self.tree.tree)
                                self.reference_tree.DefaultInsert(object_boundary)
                                self.tree.DirectInsert(object_boundary)
                                states, is_valid = self.tree.RetrieveSplitStates()
                                triggered = False
                                while states is not None:
                                        triggered = True
                                        states = torch.tensor(states, dtype=torch.float32)
                                        with torch.no_grad():
                                                q_values = self.network(states)
                                        action = self.EpsilonGreedy(q_values)
                                        self.tree.SplitOneStep(action)
                                        steps.append((states, action, is_valid))
                                        states, is_valid = self.tree.RetrieveSplitStates()

                                if triggered:
                                        split_triggered += 1
                                        trigger_period += 1
                                        #print('triggered', trigger_period, split_triggered)
                                        steps.append((None, None, False))

                                if trigger_period == self.config.splits_for_update:
                                        reward = self.ComputeReward()
                                        debug_log.write('{}\n'.format(reward))
                                        for i in range(len(steps) - 1):
                                                if steps[i][2]:
                                                        useful_split += 1
                                                        self.memory.push(steps[i][0], steps[i][1], torch.tensor([reward]), steps[i+1][0])
                                                else:
                                                        unuseful_split += 1
                                        self.reference_tree.CopyTree(self.tree.tree)
                                        trigger_period = 0


                                l = self.Optimize()
                                loss_log.write('{}\n'.format(l))
                                if e % 100 == 0:
                                        print('{} objects added, loss is {}\n'.format(e, l))
                                        self.config.epsilon = max(self.config.epsilon * self.config.epsilon_decay, self.config.min_epsilon)
                                e += 1

                                object_boundary = self.NextObj()

                                if e % self.config.target_update == 0:
                                        self.target_network.load_state_dict(self.network.state_dict())

                        torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.epoch{}'.format(epoch)+'.mdl')

                torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.mdl')
                end_time = time.time()
                train_log = open('./log/train.log', 'a')
                train_log.write('{}:\n'.format(datetime.now()))
                train_log.write('{}\n'.format(self.config))
                train_log.write('training time: {}, {} objects triggered splitting, {} useful split, {} unuseful split\n'.format(end_time-start_time, split_triggered, useful_split, unuseful_split))
                train_log.write('zero reward: {}, zero reward2: {}\n'.format(reward_is_0, reward2_is_0))
                train_log.close()
                loss_log.close()
                debug_log.close()



# if __name__ == '__main__':
#       args = parser.parse_args()
#       device = torch.device("cpu")

#       if args.action == 'train':
#               spl_learner = SplitLearner()
#               spl_learner.Initialize(args)
#               spl_learner.Train4()
#       if args.action == 'test':
#               spl_learner = SplitLearner()
#               spl_learner.Initialize(args)
#               spl_learner.Test2()
#       if args.action == 'test10':
#               spl_learner = SplitLearner()
#               spl_learner.Initialize(args)
#               spl_learner.Test10_2()

if __name__ == '__main__':
        args = parser.parse_args()
        torch.manual_seed(0)
        np.random.seed(0)
        
        # init agent
        action_space_size = int(args.action_space_size)
        RL_method = int(args.rl_method)
        brain = Agent(gamma=0.95, epsilon=1.0, alpha=0.003, maxMemorySize=5000, batch_size=64, action_space_size=action_space_size)
        brain_final = Agent(gamma=0.95, epsilon=0.0, alpha=0.003, maxMemorySize=5000, batch_size=64, action_space_size=action_space_size)
        # generate data set
        dataset_size = int(args.data_set_size)
        side = 10.0**(-8)
        model_dataset = []
        count = 0
        if args.data_distribution == 'uniform':
                with open("uniform_dataset.txt") as input_file:
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
                x_max = 100000
                x_min = 0
                y_max = 100000
                y_min = 0
        elif args.data_distribution == 'skew':
                with open("skew_dataset.txt") as input_file:
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
                x_max = 100000
                x_min = 0
                y_max = 100000
                y_min = 0
        elif args.data_distribution == 'gaussian':
                with open("gaussian_dataset_0.txt") as input_file:
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
                x_max = 100000
                x_min = 0
                y_max = 100000
                y_min = 0
        elif args.data_distribution == 'india':
                with open("india_locations_shuffle.txt") as input_file:
                    n = 0
                    for line in input_file:
                        if n%2 == 0:
                            model_dataset.append([float(line[:-1])*1000 - 10**(-8), float(line[:-1])*1000])
                        else:
                            model_dataset[-1].append(float(line[:-1])*1000 - 10**(-8))
                            model_dataset[-1].append(float(line[:-1])*1000)
                        n += 1
                dataset_size = len(model_dataset)
                x_max = 37500
                x_min = -12400
                y_max = 126000
                y_min = 50000
        elif args.data_distribution == 'china':
                with open("china_locations_shuffle.txt") as input_file:
                    n = 0
                    for line in input_file:
                        if n%2 == 0:
                            model_dataset.append([float(line[:-1])*1000 - 10**(-8), float(line[:-1])*1000])
                        else:
                            model_dataset[-1].append(float(line[:-1])*1000 - 10**(-8))
                            model_dataset[-1].append(float(line[:-1])*1000)
                        n += 1
                dataset_size = len(model_dataset)
                x_max = 55000
                x_min = 7000
                y_max = 141000
                y_min = 70000

        if args.training_data_distribution == 'nil':
            training_dataset = model_dataset
        elif args.training_data_distribution == 'uniform':
            training_dataset = []
            count = 0
            while count < args.training_set_size:
                x = random.uniform(0, 100000)
                y = random.uniform(0, 100000)
                if min(x,y) - side/2 > 0 and max(x,y) + side/2 < 100000:
                    training_dataset.append([x-side/2,x+side/2,y-side/2,y+side/2])
                    count += 1

        final_output_file = "RL_insertion_" + str(args.data_distribution) + "_" + str(args.data_set_size) + "_" + str(args.reward_comp_freq) + ".txt"
        rf = int(args.reward_comp_freq)

        # model training
        tree = RTree(args.max_entry, int(0.4 * args.max_entry))
        tree.SetInsertStrategy(args.default_ins_strategy) # RL insertion (it does not affect)
        tree.SetSplitStrategy(args.default_spl_strategy)
        reference_tree = RTree(args.max_entry, int(0.4 * args.max_entry))
        reference_tree.SetInsertStrategy(args.reference_tree_ins_strategy)
        reference_tree.SetSplitStrategy(args.reference_tree_spl_strategy) # tree and reference_tree should have the same splitting strategy

        positive_reward_by_epoch = []
        non_negative_reward_by_epoch = []
        training_query_area = 0.05/100 * ( (x_max - x_min)*(y_max - y_min) )

        start_time = time.time()

        if args.process == "training":
            for epo in range(int(args.epoch)):
                    tree.Clear()
                    reference_tree.Clear()
                    positive_reward_counter = 0
                    non_negative_reward_counter = 0

                    for i in range(args.training_set_size):
                            insert_obj = training_dataset[i]
                            if args.reference_tree_type == 'rtree':
                                reference_tree.DefaultInsert(insert_obj) # default insert covers all the way till the end of splitting
                            elif args.reference_tree_type == 'rrstar':
                                reference_tree.DirectRRInsert(insert_obj)
                                reference_tree.DirectRRSplit()
                                #print('inserted into ref tree')

                            tree.PrepareRectangle(insert_obj[0], insert_obj[1] ,insert_obj[2] ,insert_obj[3]) # set the ptr to the root
                            states = tree.RetrieveSortedInsertStates(action_space_size, RL_method) # states for insertion
                            # states = tree.RetrieveSpecialInsertStates4()

                            while states is not None: # we know if ptr points to leaf node when computing states
                                    # states = torch.tensor(states, dtype=torch.float32)
                                    if tree.GetMinAreaContainingChild() is None:
                                        action = brain.chooseAction(states)
                                        brain.state_action_pairs.append([states, action])
                                        # insert_loc = action % tree.CountChildNodes()
                                        # tree.InsertWithSortedLoc(action)
                                        tree.InsertWithSortedLoc(action)
                                    else:
                                        insert_loc = tree.GetMinAreaContainingChild()
                                        tree.InsertWithLoc(insert_loc) # insert at the chosen location
                                    
                                    states = tree.RetrieveSortedInsertStates(action_space_size, RL_method)
                                    # states = tree.RetrieveSpecialInsertStates4()

                            tree.InsertWithLoc(0)
                            #print('inserted into rl tree leaf node')
                            if args.reference_tree_type == 'rtree':
                                tree.DefaultSplit() # autmatically trigger default after insert_obj has reached a leaf node
                            elif args.reference_tree_type == 'rrstar':
                                #print('check rr* splitting')
                                tree.DirectRRSplit()
                                #print('checked rr* splitting')

                            brain.state_action_pairs.append([[0]*action_space_size*4, -1]) # terminal state for leaf node

                            # compute reward every rf insertions
                            if (i+1) > args.max_entry and (i+1) % rf == 0:
                                    avg_access_rate = 0
                                    query_rectangles = []
                                    for k in range(rf):
                                            y_x_ratio = random.uniform(0.1, 10)
                                            y_length = (training_query_area * y_x_ratio)**0.5
                                            x_length = training_query_area / y_length
                                            x_center = (training_dataset[i-k][1] + training_dataset[i-k][0])/2
                                            y_center = (training_dataset[i-k][3] + training_dataset[i-k][2])/2

                                            query_rectangle = [x_center - x_length/2, x_center + x_length/2, y_center - y_length/2, y_center + y_length/2]
                                            reference_rate = reference_tree.AccessRate(query_rectangle)
                                            tree_rate = tree.AccessRate(query_rectangle)
                                            avg_access_rate = avg_access_rate + reference_rate - tree_rate

                                    if avg_access_rate > 0:
                                        positive_reward_counter = positive_reward_counter + 1
                                    if avg_access_rate >= 0:
                                        non_negative_reward_counter = non_negative_reward_counter + 1

                                    ind = 0
                                    while ind < len(brain.state_action_pairs) - 1:
                                            if brain.state_action_pairs[ind][1] == -1:
                                                    ind = ind + 1
                                            else:
                                                    brain.storeTransition(brain.state_action_pairs[ind][0], brain.state_action_pairs[ind][1],
                                                                          avg_access_rate/(rf), brain.state_action_pairs[ind+1][0])
                                                    ind = ind + 1
                                                    # count_steps_diff = count_steps_diff * mcdr
                                                    
                                    print("Epoch number:" + "" + str(epo))
                                    print(i+1)
                                    print(avg_access_rate/(rf))
                                    print("")

                                    brain.learn()
                                    brain.state_action_pairs = []

                                    reference_tree.CopyTree(tree.tree)

                    positive_reward_by_epoch.append(positive_reward_counter)
                    non_negative_reward_by_epoch.append(non_negative_reward_counter)

                    # use the best DQN from the training process
                    if non_negative_reward_counter == max(non_negative_reward_by_epoch):
                        brain_final.Q_eval.load_state_dict(brain.Q_eval.state_dict())
                        best_epoch = epo

            torch.save(brain_final.Q_eval.state_dict(), str(args.model_number)+'_insertion_'+str(args.data_distribution)+'_'+str(args.reference_tree_type)+'_k'+str(args.action_space_size)+'.mdl')
            # print("model saved!")

        print(str(args.training_set_size))
        print("Model training time: ", time.time() - start_time)

        # use the trained model to rebuild the RL-Tree and Reference tree for testing
        brain_final.Q_eval.load_state_dict(torch.load(str(args.model_number)+'_insertion_'+str(args.data_distribution)+'_'+str(args.reference_tree_type)+'_k'+str(args.action_space_size)+'.mdl'))
        # brain_final.Q_eval.load_state_dict(torch.load(str(args.model_number)+'_insertion_'+str(args.data_distribution)+'_'+'rtree'+'_k'+str(args.action_space_size)+'.mdl'))
        rl_count = 0
        non_rl_count = 0
        action_tracker = [0]*action_space_size
        brain.EPSILON = 0
        tree.Clear()
        reference_tree.Clear()
        for i in range(dataset_size):

                insert_obj = model_dataset[i]
                if args.reference_tree_type == 'rtree':
                    reference_tree.DefaultInsert(insert_obj) # default insert covers all the way till the end of splitting
                elif args.reference_tree_type == 'rrstar':
                    reference_tree.DirectRRInsert(insert_obj)
                    reference_tree.DirectRRSplit()

                tree.PrepareRectangle(insert_obj[0], insert_obj[1] ,insert_obj[2] ,insert_obj[3]) # set the ptr to the root
                states = tree.RetrieveSortedInsertStates(action_space_size, RL_method)

                while states is not None: # we know if ptr points to leaf node when computing states
                        if tree.GetMinAreaContainingChild() is None:
                            rl_count = rl_count + 1
                            action = brain_final.chooseAction(states)
                            action_tracker[action] = action_tracker[action] + 1
                            # brain.state_action_pairs.append([states, action])
                            # insert_loc = action % tree.CountChildNodes()
                            tree.InsertWithSortedLoc(action)
                        else:
                            non_rl_count = non_rl_count + 1
                            insert_loc = tree.GetMinAreaContainingChild()
                            tree.InsertWithLoc(insert_loc) # insert at the chosen location

                        states = tree.RetrieveSortedInsertStates(action_space_size, RL_method)

                tree.InsertWithLoc(0)
                if args.reference_tree_type == 'rtree':
                    tree.DefaultSplit() # autmatically trigger default after insert_obj has reached a leaf node
                elif args.reference_tree_type == 'rrstar':
                    tree.DirectRRSplit()
                
        # testing
        test_result = []

        counter_tree_better_than_ref = 0
        access_ratio = 0
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  2.0/100 * ((x_max - x_min)*(y_max - y_min))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:

                tree_access = tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                if tree_access <= reference_tree_access:
                    counter_tree_better_than_ref = counter_tree_better_than_ref + 1

                k = k + 1

        manual_count = 0
        for obj in model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        #print(["2% query", manual_count, tree.QueryResult(), reference_tree.QueryResult()])
                
        test_result.append(["2% query", tree_acc_no, ref_tree_acc_no])


        counter_tree_better_than_ref = 0
        access_ratio = 0
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  1.0/100 * ((x_max - x_min)*(y_max - y_min))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:

                tree_access = tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                if tree_access <= reference_tree_access:
                    counter_tree_better_than_ref = counter_tree_better_than_ref + 1

                k = k + 1

        manual_count = 0
        for obj in model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        #print(["1% query", manual_count, tree.QueryResult(), reference_tree.QueryResult()])
                
        test_result.append(["1% query", tree_acc_no, ref_tree_acc_no])


        counter_tree_better_than_ref = 0
        access_ratio = 0
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  0.5/100 * ((x_max - x_min)*(y_max - y_min))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:

                tree_access = tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                if tree_access <= reference_tree_access:
                    counter_tree_better_than_ref = counter_tree_better_than_ref + 1

                k = k + 1

        manual_count = 0
        for obj in model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        #print(["0.5% query", manual_count, tree.QueryResult(), reference_tree.QueryResult()])
                
        test_result.append(["0.5% query", tree_acc_no, ref_tree_acc_no])


        counter_tree_better_than_ref = 0
        access_ratio = 0
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  0.1/100 * ((x_max - x_min)*(y_max - y_min))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:

                tree_access = tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                if tree_access <= reference_tree_access:
                    counter_tree_better_than_ref = counter_tree_better_than_ref + 1

                k = k + 1

        manual_count = 0
        for obj in model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        #print(["0.1% query", manual_count, tree.QueryResult(), reference_tree.QueryResult()])
                
        test_result.append(["0.1% query", tree_acc_no, ref_tree_acc_no])


        counter_tree_better_than_ref = 0
        access_ratio = 0
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  0.05/100 * ((x_max - x_min)*(y_max - y_min))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:

                tree_access = tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                if tree_access <= reference_tree_access:
                    counter_tree_better_than_ref = counter_tree_better_than_ref + 1

                k = k + 1

        manual_count = 0
        for obj in model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        #print(["0.05% query", manual_count, tree.QueryResult(), reference_tree.QueryResult()])
                
        test_result.append(["0.05% query", tree_acc_no, ref_tree_acc_no])


        counter_tree_better_than_ref = 0
        access_ratio = 0
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  0.01/100 * ((x_max - x_min)*(y_max - y_min))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:

                tree_access = tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                if tree_access <= reference_tree_access:
                    counter_tree_better_than_ref = counter_tree_better_than_ref + 1

                k = k + 1

        manual_count = 0
        for obj in model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        #print(["0.01% query", manual_count, tree.QueryResult(), reference_tree.QueryResult()])

        test_result.append(["0.01% query", tree_acc_no, ref_tree_acc_no])


        counter_tree_better_than_ref = 0
        access_ratio = 0
        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  0.005/100 * ((x_max - x_min)*(y_max - y_min))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:

                tree_access = tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = reference_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                if tree_access <= reference_tree_access:
                    counter_tree_better_than_ref = counter_tree_better_than_ref + 1

                k = k + 1

        manual_count = 0
        for obj in model_dataset:
            left = max(x-side, obj[0])
            right = min(x+side, obj[1])
            bottom = max(y-side, obj[2])
            top = min(y+side, obj[3])
            if left < right and bottom < top:
                manual_count += 1

        #print(["0.005% query", manual_count, tree.QueryResult(), reference_tree.QueryResult()])

        test_result.append(["0.005% query", tree_acc_no, ref_tree_acc_no])

        final_output_file = "RL_no_cf_skew9_ins_100k_100k_5e_kaiyu.txt"
        print("")
        print(test_result)

        # if args.process == "training":
        #     print("best epoch", best_epoch)

        # print("dataset size", dataset_size)
        # print("positive reward", positive_reward_by_epoch)
        # print("non negative reward", non_negative_reward_by_epoch)
        # print(action_tracker)
        # print("RL count", rl_count)
        # print("non RL count", non_rl_count)

        # k = 0
        # testing_query_area =  0.05/100 * (100000**2)
        # side = (testing_query_area**0.5)/2
        # while k < 10:
        #     x = random.uniform(0, 1)*100000
        #     y = ((random.uniform(0, 1))**1)*100000
        #     if x - side > 0 and y - side > 0 and x + side < 100000 and y + side < 100000:

        #         tree_access = tree.Query((x - side, x + side, y - side, y + side))
        #         print(tree.QueryResult())
        #         reference_tree_access = reference_tree.Query((x - side, x + side, y - side, y + side))
        #         print(reference_tree.QueryResult())
        #         obj_num = 0
        #         for i in model_dataset:
        #             nx,ny = max(i[0], x-side),max(i[2], y-side)
        #             nx2,ny2 = min(i[1], x+side),min(i[3], y+side)
        #             w,h = nx2-nx, ny2-ny
        #             if w>=0 and h>=0:
        #                 obj_num += 1
        #         print(obj_num)
        #         print()
        #         k = k + 1

"""
        for i in final_output_file:
            for j in i:
                    print(i, file=open(final_output_file, "a"))
                    print("", file=open(final_output_file, "a"))

"""
