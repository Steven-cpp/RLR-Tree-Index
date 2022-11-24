# from tqdm import trange
import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import hashlib

import random
from random import randint
import argparse
from collections import namedtuple
import time
from datetime import datetime

from RTree import RTree

parser = argparse.ArgumentParser()

parser.add_argument('-action', choices=['train', 'test', 'test10', 'baseline'], default='test')
parser.add_argument('-objfile', help='data file')
parser.add_argument('-queryfile', help='query file')
parser.add_argument('-epoch', type=int, help='number of epoches', default=20)
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
parser.add_argument('-parts', type=int, help='number of parts to train', default=1)
parser.add_argument('-network', choices=['strategy', 'spl_loc', 'spl_loc_short', 'sort_spl_loc'], help='which network is used for training', default='sort_spl_loc')
parser.add_argument('-teacher_forcing', type=float, help='the percentage of splits that are with teacher forcing technique', default=0.1)
parser.add_argument('-data_distribution', choices=['uniform', 'skew', 'gaussian', 'china', 'india'], help='data set distribution', default='gaussian')
parser.add_argument('-data_set_size', type=int, help='data set size', default=20000000)
parser.add_argument('-training_set_size', type=int, help='training set size', default=100000)
parser.add_argument('-model_number', type=int, help='which insertion training model', default=0)
parser.add_argument('-reference_tree_type', choices=['rtree', 'rrstar'], help='which reference tree to use', default='rtree')
parser.add_argument('-reward_comp_freq', type=int, help='insertion reward computation frequency', default=10)

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
        self.insertion_network = None
        self.target_network = None
        self.temp_network = None
        self.temp_network_final = None
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
            self.insertion_network = Agent(gamma=0.95, epsilon=0.0, alpha=0.003, maxMemorySize=5000, batch_size=64, action_space_size=self.config.action_space)
            self.temp_network = Agent(gamma=0.95, epsilon=0.1, alpha=0.003, maxMemorySize=5000, batch_size=64, action_space_size=self.config.action_space)
            self.temp_network_final = Agent(gamma=0.95, epsilon=0.1, alpha=0.003, maxMemorySize=5000, batch_size=64, action_space_size=self.config.action_space)



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

    def TimeAndSize(self):
        self.network.load_state_dict(torch.load(str(self.config.data_distribution)+'_'+'k'+str(self.config.action_space)+'.mdl'))
        # self.insertion_network.Q_eval.load_state_dict(torch.load(str(self.config.model_number)+'_insertion_'+str(self.config.data_distribution)+'_'+str(self.config.reference_tree_type)+'_k'+str(self.config.action_space)+'.mdl'))
        if self.config.action == 'test':
            # self.network.load_state_dict(torch.load(str(self.config.data_distribution)+'_'+'k'+str(self.config.action_space)+'.mdl'))
            self.insertion_network.Q_eval.load_state_dict(torch.load(str(self.config.model_number)+'_insertion_'+str(self.config.data_distribution)+'_'+str(self.config.reference_tree_type)+'_k'+str(self.config.action_space)+'.mdl'))
        elif self.config.action == 'train':
            self.insertion_network.Q_eval.load_state_dict(torch.load(str(self.config.model_number)+'_insertion_'+str(self.config.data_distribution)+'_'+str(self.config.reference_tree_type)+'_k'+str(self.config.action_space)+'_modified'+'.mdl'))
        self.insertion_network.EPSILON = 0.0
        # self.network.eval()
        # self.ResetObjLoader()
        self.tree.Clear()
        self.reference_tree.Clear() #CONTINUE FROM HERE
        # object_boundary = self.NextObj()
        action_count = [0]*self.config.action_space
        action_tracker = [0]*self.config.action_space
        obj_cnt = 0

        self.tree.SetStartTimestamp()
        for i in range(len(self.model_dataset)):
            obj_cnt += 1
            # self.tree.DirectInsert(self.model_dataset[i])
            # add RL insertion here
            insert_obj = self.model_dataset[i]
            self.tree.PrepareRectangle(insert_obj[0], insert_obj[1] ,insert_obj[2] ,insert_obj[3]) # set the ptr to the root
            states = self.tree.RetrieveSortedInsertStates(2, 0)

            while states is not None:
                if self.tree.GetMinAreaContainingChild() is None:
                    action = self.insertion_network.chooseAction(states)
                    action_tracker[action] = action_tracker[action] + 1
                    self.tree.InsertWithSortedLoc(action)
                else:
                    insert_loc = self.tree.GetMinAreaContainingChild()
                    self.tree.InsertWithLoc(insert_loc) # insert at the chosen location
                states = self.tree.RetrieveSortedInsertStates(2, 0)
            self.tree.InsertWithLoc(0)

            # RL splitting
            if self.tree.NeedSplit():
                while True:
                    num_of_zero_ovlp_splits = self.tree.GetNumberOfNonOverlapSplitLocs()
                    if num_of_zero_ovlp_splits == None:
                        break

                    if num_of_zero_ovlp_splits <= 1:
                        self.tree.SplitInMinOverlap()
                    else:
                        states = self.tree.RetrieveZeroOVLPSplitSortedByPerimeterState()
                        states = torch.tensor(states, dtype=torch.float32).to(self.network.device)
                        q_values = self.network(states).to(self.network.device)
                        action = torch.argmax(q_values).item()
                        action_count[action] += 1
                        self.tree.SplitWithCandidateAction(action)
        self.tree.SetEndTimestamp()

        self.reference_tree.SetStartTimestamp()
        for i in range(len(self.model_dataset)):
            self.reference_tree.DefaultInsert(self.model_dataset[i])
        self.reference_tree.SetEndTimestamp()

        print("Reference Tree")
        print([self.reference_tree.GetDurationInSeconds(), self.reference_tree.GetIndexSizeInMB()])
        print("RLR Tree")
        print([self.tree.GetDurationInSeconds(), self.tree.GetIndexSizeInMB()])

    def TestKNN(self):
        self.network.load_state_dict(torch.load(str(self.config.data_distribution)+'_'+'k'+str(self.config.action_space)+'.mdl'))
        # self.insertion_network.Q_eval.load_state_dict(torch.load(str(self.config.model_number)+'_insertion_'+str(self.config.data_distribution)+'_'+str(self.config.reference_tree_type)+'_k'+str(self.config.action_space)+'.mdl'))
        if self.config.action == 'test':
            # self.network.load_state_dict(torch.load(str(self.config.data_distribution)+'_'+'k'+str(self.config.action_space)+'.mdl'))
            self.insertion_network.Q_eval.load_state_dict(torch.load(str(self.config.model_number)+'_insertion_'+str(self.config.data_distribution)+'_'+str(self.config.reference_tree_type)+'_k'+str(self.config.action_space)+'.mdl'))
        elif self.config.action == 'train':
            self.insertion_network.Q_eval.load_state_dict(torch.load(str(self.config.model_number)+'_insertion_'+str(self.config.data_distribution)+'_'+str(self.config.reference_tree_type)+'_k'+str(self.config.action_space)+'_modified'+'.mdl'))
        self.insertion_network.EPSILON = 0.0
        # self.network.eval()
        # self.ResetObjLoader()
        self.tree.Clear()
        self.reference_tree.Clear() #CONTINUE FROM HERE
        # object_boundary = self.NextObj()
        action_count = [0]*self.config.action_space
        action_tracker = [0]*self.config.action_space
        obj_cnt = 0

        self.reference_tree.SetStartTimestamp()
        for i in range(len(self.model_dataset)):
            self.reference_tree.DefaultInsert(self.model_dataset[i])
        self.reference_tree.SetEndTimestamp()

        self.tree.SetStartTimestamp()
        for i in range(len(self.model_dataset)):
            obj_cnt += 1
            # self.tree.DirectInsert(self.model_dataset[i])
            # add RL insertion here
            insert_obj = self.model_dataset[i]
            self.tree.PrepareRectangle(insert_obj[0], insert_obj[1] ,insert_obj[2] ,insert_obj[3]) # set the ptr to the root
            states = self.tree.RetrieveSortedInsertStates(2, 0)

            while states is not None:
                if self.tree.GetMinAreaContainingChild() is None:
                    action = self.insertion_network.chooseAction(states)
                    action_tracker[action] = action_tracker[action] + 1
                    self.tree.InsertWithSortedLoc(action)
                else:
                    insert_loc = self.tree.GetMinAreaContainingChild()
                    self.tree.InsertWithLoc(insert_loc) # insert at the chosen location
                states = self.tree.RetrieveSortedInsertStates(2, 0)
            self.tree.InsertWithLoc(0)

            # RL splitting
            if self.tree.NeedSplit():
                while True:
                    num_of_zero_ovlp_splits = self.tree.GetNumberOfNonOverlapSplitLocs()
                    if num_of_zero_ovlp_splits == None:
                        break

                    if num_of_zero_ovlp_splits <= 1:
                        self.tree.SplitInMinOverlap()
                    else:
                        states = self.tree.RetrieveZeroOVLPSplitSortedByPerimeterState()
                        states = torch.tensor(states, dtype=torch.float32).to(self.network.device)
                        q_values = self.network(states).to(self.network.device)
                        action = torch.argmax(q_values).item()
                        action_count[action] += 1
                        self.tree.SplitWithCandidateAction(action)
        self.tree.SetEndTimestamp()

        print("Reference Tree")
        print([self.reference_tree.GetDurationInSeconds(), self.reference_tree.GetIndexSizeInMB()])
        print("RLR Tree")
        print([self.tree.GetDurationInSeconds(), self.tree.GetIndexSizeInMB()])

        # print('RLR Tree average tree node area: ', self.tree.AverageNodeArea())
        # print('RLR Tree average tree node children: ', self.tree.AverageNodeChildren())
        # print('RLR Tree total tree nodes: ', self.tree.TotalTreeNodeNum())

        # print('Ref Tree average tree node area: ', self.reference_tree.AverageNodeArea())
        # print('Ref Tree average tree node children: ', self.reference_tree.AverageNodeChildren())
        # print('Ref Tree total tree nodes: ', self.reference_tree.TotalTreeNodeNum())

        test_result = []

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

        k = 0

        while k < 100:
            k += 1

            x = random.uniform(self.data_set_bounds[0], self.data_set_bounds[1])
            y = random.uniform(self.data_set_bounds[2], self.data_set_bounds[3])

            tree_acc_no_1 = tree_acc_no_1 + self.tree.KNNQuery(x,y,1)
            ref_tree_acc_no_1 = ref_tree_acc_no_1 + self.reference_tree.KNNQuery(x,y,1)

            tree_acc_no_2 = tree_acc_no_2 + self.tree.KNNQuery(x,y,5)
            ref_tree_acc_no_2 = ref_tree_acc_no_2 + self.reference_tree.KNNQuery(x,y,5)

            tree_acc_no_3 = tree_acc_no_3 + self.tree.KNNQuery(x,y,25)
            ref_tree_acc_no_3 = ref_tree_acc_no_3 + self.reference_tree.KNNQuery(x,y,25)

            tree_acc_no_4 = tree_acc_no_4 + self.tree.KNNQuery(x,y,125)
            ref_tree_acc_no_4 = ref_tree_acc_no_4 + self.reference_tree.KNNQuery(x,y,125)

            tree_acc_no_5 = tree_acc_no_5 + self.tree.KNNQuery(x,y,625)
            ref_tree_acc_no_5 = ref_tree_acc_no_5 + self.reference_tree.KNNQuery(x,y,625)

        print(["k1", tree_acc_no_1, ref_tree_acc_no_1, "k5", tree_acc_no_2, ref_tree_acc_no_2, "k25", tree_acc_no_3, ref_tree_acc_no_3, "k125", tree_acc_no_4, ref_tree_acc_no_4, "k625", tree_acc_no_5, ref_tree_acc_no_5])


    def Test3(self):
        self.network.load_state_dict(torch.load(str(self.config.data_distribution)+'_'+'k'+str(self.config.action_space)+'.mdl'))
        # self.insertion_network.Q_eval.load_state_dict(torch.load(str(self.config.model_number)+'_insertion_'+str(self.config.data_distribution)+'_'+str(self.config.reference_tree_type)+'_k'+str(self.config.action_space)+'.mdl'))
        if self.config.action == 'test':
            # self.network.load_state_dict(torch.load(str(self.config.data_distribution)+'_'+'k'+str(self.config.action_space)+'.mdl'))
            self.insertion_network.Q_eval.load_state_dict(torch.load(str(self.config.model_number)+'_insertion_'+str(self.config.data_distribution)+'_'+str(self.config.reference_tree_type)+'_k'+str(self.config.action_space)+'.mdl'))
        elif self.config.action == 'train':
            self.insertion_network.Q_eval.load_state_dict(torch.load(str(self.config.model_number)+'_insertion_'+str(self.config.data_distribution)+'_'+str(self.config.reference_tree_type)+'_k'+str(self.config.action_space)+'_modified'+'.mdl'))
        self.insertion_network.EPSILON = 0.0
        # self.network.eval()
        # self.ResetObjLoader()
        self.tree.Clear()
        self.reference_tree.Clear() #CONTINUE FROM HERE
        # object_boundary = self.NextObj()
        action_count = [0]*self.config.action_space
        action_tracker = [0]*self.config.action_space
        obj_cnt = 0
        for i in range(len(self.model_dataset)):
            obj_cnt += 1
            self.reference_tree.DefaultInsert(self.model_dataset[i])
            # self.tree.DirectInsert(self.model_dataset[i])
            # add RL insertion here
            insert_obj = self.model_dataset[i]
            self.tree.PrepareRectangle(insert_obj[0], insert_obj[1] ,insert_obj[2] ,insert_obj[3]) # set the ptr to the root
            states = self.tree.RetrieveSortedInsertStates(2, 0)

            while states is not None:
                if self.tree.GetMinAreaContainingChild() is None:
                    action = self.insertion_network.chooseAction(states)
                    action_tracker[action] = action_tracker[action] + 1
                    self.tree.InsertWithSortedLoc(action)
                else:
                    insert_loc = self.tree.GetMinAreaContainingChild()
                    self.tree.InsertWithLoc(insert_loc) # insert at the chosen location
                states = self.tree.RetrieveSortedInsertStates(2, 0)
            self.tree.InsertWithLoc(0)

            # RL splitting
            if self.tree.NeedSplit():
                while True:
                    num_of_zero_ovlp_splits = self.tree.GetNumberOfNonOverlapSplitLocs()
                    if num_of_zero_ovlp_splits == None:
                        break

                    if num_of_zero_ovlp_splits <= 1:
                        self.tree.SplitInMinOverlap()
                    else:
                        states = self.tree.RetrieveZeroOVLPSplitSortedByPerimeterState()
                        states = torch.tensor(states, dtype=torch.float32).to(self.network.device)
                        q_values = self.network(states).to(self.network.device)
                        action = torch.argmax(q_values).item()
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
        #print("splitting action count", action_count)
        #print("insertion action count", action_tracker)

    def Test3_RealData(self):
        self.network.load_state_dict(torch.load(str(self.config.data_distribution)+'_'+'k'+str(self.config.action_space)+'.mdl'))
        # self.insertion_network.Q_eval.load_state_dict(torch.load(str(self.config.model_number)+'_insertion_'+str(self.config.data_distribution)+'_'+str(self.config.reference_tree_type)+'_k'+str(self.config.action_space)+'.mdl'))
        if self.config.action == 'test':
            # self.network.load_state_dict(torch.load(str(self.config.data_distribution)+'_'+'k'+str(self.config.action_space)+'.mdl'))
            self.insertion_network.Q_eval.load_state_dict(torch.load(str(self.config.model_number)+'_insertion_'+str(self.config.data_distribution)+'_'+str(self.config.reference_tree_type)+'_k'+str(self.config.action_space)+'.mdl'))
        elif self.config.action == 'train':
            self.insertion_network.Q_eval.load_state_dict(torch.load(str(self.config.model_number)+'_insertion_'+str(self.config.data_distribution)+'_'+str(self.config.reference_tree_type)+'_k'+str(self.config.action_space)+'_modified'+'.mdl'))
        self.insertion_network.EPSILON = 0.0
        # self.network.eval()
        # self.ResetObjLoader()
        self.tree.Clear()
        self.reference_tree.Clear() #CONTINUE FROM HERE
        # object_boundary = self.NextObj()
        action_count = [0]*self.config.action_space
        action_tracker = [0]*self.config.action_space
        obj_cnt = 0
        for i in range(len(self.model_dataset)):
            if i%10000000 == 0:
                print("Already Inserted: ", i)

            obj_cnt += 1
            self.reference_tree.DefaultInsert(self.model_dataset[i])
            # self.tree.DirectInsert(self.model_dataset[i])
            # add RL insertion here
            insert_obj = self.model_dataset[i]
            self.tree.PrepareRectangle(insert_obj[0], insert_obj[1] ,insert_obj[2] ,insert_obj[3]) # set the ptr to the root
            states = self.tree.RetrieveSortedInsertStates(2, 0)

            while states is not None:
                if self.tree.GetMinAreaContainingChild() is None:
                    action = self.insertion_network.chooseAction(states)
                    action_tracker[action] = action_tracker[action] + 1
                    self.tree.InsertWithSortedLoc(action)
                else:
                    insert_loc = self.tree.GetMinAreaContainingChild()
                    self.tree.InsertWithLoc(insert_loc) # insert at the chosen location
                states = self.tree.RetrieveSortedInsertStates(2, 0)
            self.tree.InsertWithLoc(0)

            # RL splitting
            if self.tree.NeedSplit():
                while True:
                    num_of_zero_ovlp_splits = self.tree.GetNumberOfNonOverlapSplitLocs()
                    if num_of_zero_ovlp_splits == None:
                        break

                    if num_of_zero_ovlp_splits <= 1:
                        self.tree.SplitInMinOverlap()
                    else:
                        states = self.tree.RetrieveZeroOVLPSplitSortedByPerimeterState()
                        states = torch.tensor(states, dtype=torch.float32).to(self.network.device)
                        q_values = self.network(states).to(self.network.device)
                        action = torch.argmax(q_values).item()
                        action_count[action] += 1
                        self.tree.SplitWithCandidateAction(action)
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

        testing_query_area =  2.0/100 * ((self.data_set_bounds[1] - self.data_set_bounds[0])*(self.data_set_bounds[3] - self.data_set_bounds[2]))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            # x = random.uniform(self.data_set_bounds[0], self.data_set_bounds[1])
            # y = random.uniform(self.data_set_bounds[2], self.data_set_bounds[3])
            i = random.randint(0, 95000000)
            x = (self.model_dataset[i][0] + self.model_dataset[i][1])/2.0
            y = (self.model_dataset[i][2] + self.model_dataset[i][3])/2.0

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

        testing_query_area =  1.0/100 * ((self.data_set_bounds[1] - self.data_set_bounds[0])*(self.data_set_bounds[3] - self.data_set_bounds[2]))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            # x = random.uniform(self.data_set_bounds[0], self.data_set_bounds[1])
            # y = random.uniform(self.data_set_bounds[2], self.data_set_bounds[3])
            i = random.randint(0, 95000000)
            x = (self.model_dataset[i][0] + self.model_dataset[i][1])/2.0
            y = (self.model_dataset[i][2] + self.model_dataset[i][3])/2.0

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

        testing_query_area =  0.5/100 * ((self.data_set_bounds[1] - self.data_set_bounds[0])*(self.data_set_bounds[3] - self.data_set_bounds[2]))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            # x = random.uniform(self.data_set_bounds[0], self.data_set_bounds[1])
            # y = random.uniform(self.data_set_bounds[2], self.data_set_bounds[3])
            i = random.randint(0, 95000000)
            x = (self.model_dataset[i][0] + self.model_dataset[i][1])/2.0
            y = (self.model_dataset[i][2] + self.model_dataset[i][3])/2.0

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

        testing_query_area =  0.1/100 * ((self.data_set_bounds[1] - self.data_set_bounds[0])*(self.data_set_bounds[3] - self.data_set_bounds[2]))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            # x = random.uniform(self.data_set_bounds[0], self.data_set_bounds[1])
            # y = random.uniform(self.data_set_bounds[2], self.data_set_bounds[3])
            i = random.randint(0, 95000000)
            x = (self.model_dataset[i][0] + self.model_dataset[i][1])/2.0
            y = (self.model_dataset[i][2] + self.model_dataset[i][3])/2.0

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

        testing_query_area =  0.05/100 * ((self.data_set_bounds[1] - self.data_set_bounds[0])*(self.data_set_bounds[3] - self.data_set_bounds[2]))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            # x = random.uniform(self.data_set_bounds[0], self.data_set_bounds[1])
            # y = random.uniform(self.data_set_bounds[2], self.data_set_bounds[3])
            i = random.randint(0, 95000000)
            x = (self.model_dataset[i][0] + self.model_dataset[i][1])/2.0
            y = (self.model_dataset[i][2] + self.model_dataset[i][3])/2.0

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

        testing_query_area =  0.01/100 * ((self.data_set_bounds[1] - self.data_set_bounds[0])*(self.data_set_bounds[3] - self.data_set_bounds[2]))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            # x = random.uniform(self.data_set_bounds[0], self.data_set_bounds[1])
            # y = random.uniform(self.data_set_bounds[2], self.data_set_bounds[3])
            i = random.randint(0, 95000000)
            x = (self.model_dataset[i][0] + self.model_dataset[i][1])/2.0
            y = (self.model_dataset[i][2] + self.model_dataset[i][3])/2.0

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

        testing_query_area =  0.005/100 * ((self.data_set_bounds[1] - self.data_set_bounds[0])*(self.data_set_bounds[3] - self.data_set_bounds[2]))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            # x = random.uniform(self.data_set_bounds[0], self.data_set_bounds[1])
            # y = random.uniform(self.data_set_bounds[2], self.data_set_bounds[3])
            i = random.randint(0, 95000000)
            x = (self.model_dataset[i][0] + self.model_dataset[i][1])/2.0
            y = (self.model_dataset[i][2] + self.model_dataset[i][3])/2.0

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
        print("splitting action count", action_count)
        print("insertion action count", action_tracker)

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

    def Train6(self): # RL separate partitions with and without overlaps
        #with teacher forcing
        start_time = time.time()
        # loss_log = open("./log/{}.loss".format(self.id), 'w')
        # reward_log = open("./log/{}.reward".format(self.id), "w")
        steps = []
        object_num = len(self.training_dataset)

        self.network.load_state_dict(torch.load(str(self.config.data_distribution)+'_'+'k'+str(self.config.action_space)+'.mdl'))
        self.insertion_network.Q_eval.load_state_dict(torch.load(str(self.config.model_number)+'_insertion_'+str(self.config.data_distribution)+'_'+str(self.config.reference_tree_type)+'_k'+str(self.config.action_space)+'.mdl'))
        self.insertion_network.EPSILON = 0.0
        self.temp_network.Q_eval.load_state_dict(self.insertion_network.Q_eval.state_dict())
        self.temp_network.EPSILON = 0.1
        self.temp_network_final.Q_eval.load_state_dict(self.insertion_network.Q_eval.state_dict())
        self.temp_network_final.EPSILON = 0.1

        positive_reward_by_epoch = []
        non_negative_reward_by_epoch = []
        training_query_area = 0.05/100 * ( (self.data_set_bounds[1] - self.data_set_bounds[0])*(self.data_set_bounds[3] - self.data_set_bounds[2]) )

        for epo in range(self.config.epoch):
            self.tree.Clear()
            self.reference_tree.Clear()
            positive_reward_counter = 0
            non_negative_reward_counter = 0

            for i in range(len(self.training_dataset)):
                insert_obj = self.training_dataset[i]

                self.reference_tree.DirectInsert(insert_obj)
                if self.reference_tree.NeedSplit():
                    while True:
                        num_of_zero_ovlp_splits = self.reference_tree.GetNumberOfNonOverlapSplitLocs()
                        if num_of_zero_ovlp_splits == None:
                            break

                        if num_of_zero_ovlp_splits <= 1:
                            self.reference_tree.SplitInMinOverlap()
                        else:
                            states = self.reference_tree.RetrieveZeroOVLPSplitSortedByPerimeterState()
                            # states = self.reference_tree.RetrieveZeroOVLPSplitSortedByWeightedPerimeterState()
                            states = torch.tensor(states, dtype=torch.float32).to(self.network.device)
                            q_values = self.network(states).to(self.network.device)
                            action = torch.argmax(q_values).item()
                            self.reference_tree.SplitWithCandidateAction(action)

                self.tree.PrepareRectangle(insert_obj[0], insert_obj[1] ,insert_obj[2] ,insert_obj[3]) # set the ptr to the root
                states = self.tree.RetrieveSortedInsertStates(2, 0)

                while states is not None:
                    if self.tree.GetMinAreaContainingChild() is None:
                        action = self.temp_network.chooseAction(states)
                        self.temp_network.state_action_pairs.append([states, action])
                        self.tree.InsertWithSortedLoc(action)
                    else:
                        insert_loc = self.tree.GetMinAreaContainingChild()
                        self.tree.InsertWithLoc(insert_loc)
                    states = self.tree.RetrieveSortedInsertStates(2,0)
                self.tree.InsertWithLoc(0)

                if self.tree.NeedSplit():
                    while True:
                        num_of_zero_ovlp_splits = self.tree.GetNumberOfNonOverlapSplitLocs()
                        if num_of_zero_ovlp_splits == None:
                            break

                        if num_of_zero_ovlp_splits <= 1:
                            self.tree.SplitInMinOverlap()
                        else:
                            states = self.tree.RetrieveZeroOVLPSplitSortedByPerimeterState()
                            states = torch.tensor(states, dtype=torch.float32).to(self.network.device)
                            q_values = self.network(states).to(self.network.device)
                            action = torch.argmax(q_values).item()
                            self.tree.SplitWithCandidateAction(action)

                self.temp_network.state_action_pairs.append([[0]*self.config.action_space*4, -1]) # terminal state for leaf node
                if (i+1) > self.config.max_entry and (i+1) % self.config.reward_comp_freq == 0:
                    avg_access_rate = 0
                    query_rectangles = []
                    for k in range(self.config.reward_comp_freq):
                        y_x_ratio = random.uniform(0.1, 10)
                        y_length = (training_query_area * y_x_ratio)**0.5
                        x_length = training_query_area / y_length
                        x_center = (self.training_dataset[i-k][1] + self.training_dataset[i-k][0])/2
                        y_center = (self.training_dataset[i-k][3] + self.training_dataset[i-k][2])/2

                        query_rectangle = [x_center - x_length/2, x_center + x_length/2, y_center - y_length/2, y_center + y_length/2]
                        reference_rate = self.reference_tree.AccessRate(query_rectangle)
                        tree_rate = self.tree.AccessRate(query_rectangle)
                        avg_access_rate = avg_access_rate + reference_rate - tree_rate

                    if avg_access_rate > 0:
                        positive_reward_counter = positive_reward_counter + 1
                    if avg_access_rate >= 0:
                        non_negative_reward_counter = non_negative_reward_counter + 1

                    ind = 0
                    while ind < len(self.temp_network.state_action_pairs) - 1:
                        if self.temp_network.state_action_pairs[ind][1] == -1:
                            ind = ind + 1
                        else:
                            self.temp_network.storeTransition(self.temp_network.state_action_pairs[ind][0], self.temp_network.state_action_pairs[ind][1],avg_access_rate/(self.config.reward_comp_freq), self.temp_network.state_action_pairs[ind+1][0])
                            ind = ind + 1

                    print("Epoch number:" + "" + str(epo))
                    print(i+1)
                    print(avg_access_rate/(self.config.reward_comp_freq))
                    self.temp_network.learn()
                    self.temp_network.state_action_pairs = []

                    self.reference_tree.CopyTree(self.tree.tree)

            positive_reward_by_epoch.append(positive_reward_counter)
            non_negative_reward_by_epoch.append(non_negative_reward_counter)

            if non_negative_reward_counter == max(non_negative_reward_by_epoch):
                self.temp_network_final.Q_eval.load_state_dict(self.temp_network.Q_eval.state_dict())
                best_epoch = epo

        # compare the modified network and the initial networt
        self.tree.Clear()
        self.reference_tree.Clear()
        self.temp_network_final.EPSILON = 0.0
        self.insertion_network.EPSILON = 0.0

        modified_tree = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
        modified_tree.SetInsertStrategy(self.config.default_ins_strategy)
        modified_tree.SetSplitStrategy(self.config.default_spl_strategy)
        original_tree = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
        original_tree.SetInsertStrategy(self.config.reference_tree_ins_strategy)
        original_tree.SetSplitStrategy(self.config.reference_tree_spl_strategy)

        for i in range(len(self.training_dataset)):
            insert_obj = self.training_dataset[i]

            # modified network
            modified_tree.PrepareRectangle(insert_obj[0], insert_obj[1] ,insert_obj[2] ,insert_obj[3]) # set the ptr to the root
            states = modified_tree.RetrieveSortedInsertStates(2, 0)

            while states is not None:
                if modified_tree.GetMinAreaContainingChild() is None:
                    action = self.temp_network_final.chooseAction(states)
                    modified_tree.InsertWithSortedLoc(action)
                else:
                    insert_loc = modified_tree.GetMinAreaContainingChild()
                    modified_tree.InsertWithLoc(insert_loc)
                states = modified_tree.RetrieveSortedInsertStates(2, 0)
            modified_tree.InsertWithLoc(0)

            if modified_tree.NeedSplit():
                while True:
                    num_of_zero_ovlp_splits = modified_tree.GetNumberOfNonOverlapSplitLocs()
                    if num_of_zero_ovlp_splits == None:
                        break

                    if num_of_zero_ovlp_splits <= 1:
                        modified_tree.SplitInMinOverlap()
                    else:
                        states = modified_tree.RetrieveZeroOVLPSplitSortedByPerimeterState()
                        states = torch.tensor(states, dtype=torch.float32).to(self.network.device)
                        q_values = self.network(states).to(self.network.device)
                        action = torch.argmax(q_values).item()
                        modified_tree.SplitWithCandidateAction(action)

            # original network
            original_tree.PrepareRectangle(insert_obj[0], insert_obj[1] ,insert_obj[2] ,insert_obj[3]) # set the ptr to the root
            states = original_tree.RetrieveSortedInsertStates(2, 0)

            while states is not None:
                if original_tree.GetMinAreaContainingChild() is None:
                    action = self.insertion_network.chooseAction(states)
                    original_tree.InsertWithSortedLoc(action)
                else:
                    insert_loc = original_tree.GetMinAreaContainingChild()
                    original_tree.InsertWithLoc(insert_loc)
                states = original_tree.RetrieveSortedInsertStates(2, 0)
            original_tree.InsertWithLoc(0)

            if original_tree.NeedSplit():
                while True:
                    num_of_zero_ovlp_splits = original_tree.GetNumberOfNonOverlapSplitLocs()
                    if num_of_zero_ovlp_splits == None:
                        break

                    if num_of_zero_ovlp_splits <= 1:
                        original_tree.SplitInMinOverlap()
                    else:
                        states = original_tree.RetrieveZeroOVLPSplitSortedByPerimeterState()
                        states = torch.tensor(states, dtype=torch.float32).to(self.network.device)
                        q_values = self.network(states).to(self.network.device)
                        action = torch.argmax(q_values).item()
                        original_tree.SplitWithCandidateAction(action)

        tree_acc_no = 0
        ref_tree_acc_no = 0

        testing_query_area =  0.01/100 * ((self.data_set_bounds[1] - self.data_set_bounds[0])*(self.data_set_bounds[3] - self.data_set_bounds[2]))
        side = (testing_query_area**0.5)/2
        k = 0
        while k < 1000:
            x = random.uniform(self.data_set_bounds[0], self.data_set_bounds[1])
            y = random.uniform(self.data_set_bounds[2], self.data_set_bounds[3])
            if x - side > self.data_set_bounds[0] and y - side > self.data_set_bounds[2] and x + side < self.data_set_bounds[1] and y + side < self.data_set_bounds[3]:

                tree_access = modified_tree.Query((x - side, x + side, y - side, y + side))
                reference_tree_access = original_tree.Query((x - side, x + side, y - side, y + side))
                tree_acc_no = tree_acc_no + tree_access
                ref_tree_acc_no = ref_tree_acc_no + reference_tree_access

                k = k + 1


        end_time = time.time()
        print(["modified", tree_acc_no, "original", ref_tree_acc_no])
        if tree_acc_no < ref_tree_acc_no:
            self.insertion_network.Q_eval.load_state_dict(self.temp_network_final.Q_eval.state_dict())
            print('Modified Network Selected!')
            print('Best Epoch', best_epoch)
        else:
            print('Original Network Selected!')
        torch.save(self.insertion_network.Q_eval.state_dict(), str(self.config.model_number)+'_insertion_'+str(self.config.data_distribution)+'_'+str(self.config.reference_tree_type)+'_k'+str(self.config.action_space)+'_modified'+'.mdl')
        self.tree.Clear()
        self.reference_tree.Clear()

if __name__ == '__main__':
    args = parser.parse_args()
    spl_learner = SplitLearner()
    spl_learner.Initialize(args)

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

    if args.action == 'train':
        spl_learner.training_dataset = model_dataset[:int(args.training_set_size)]
        if args.data_distribution in ['china','india']:
            spl_learner.model_dataset = model_dataset
        elif args.data_distribution in ['uniform','skew','gaussian']:
            spl_learner.model_dataset = model_dataset[:int(args.data_set_size)]

        spl_learner.data_set_bounds = [x_min, x_max, y_min, y_max]
        spl_learner.Train6()
        spl_learner.Test3()

    if args.action == 'test':
        spl_learner.training_dataset = model_dataset[:int(args.training_set_size)]
        if args.data_distribution in ['china','india']:
            spl_learner.model_dataset = model_dataset
        elif args.data_distribution in ['uniform','skew','gaussian']:
            spl_learner.model_dataset = model_dataset[:int(args.data_set_size)]

        spl_learner.data_set_bounds = [x_min, x_max, y_min, y_max]
        spl_learner.Test3()
