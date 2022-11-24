import ctypes
import numpy as np
import torch
import random

class RTree:
    def __init__(self, max_entry, min_entry):
        self.lib = ctypes.CDLL('./tree.so')

        self.lib.ConstructTree.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.ConstructTree.restype = ctypes.c_void_p

        self.lib.SetDefaultInsertStrategy.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.SetDefaultInsertStrategy.restype = ctypes.c_void_p

        self.lib.SetDefaultSplitStrategy.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.SetDefaultSplitStrategy.restype = ctypes.c_void_p

        self.lib.QueryRectangle.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        self.lib.QueryRectangle.restype = ctypes.c_int

        self.lib.KNNQuery.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_int]
        self.lib.KNNQuery.restype = ctypes.c_int

        self.lib.GetRoot.argtypes = [ctypes.c_void_p]
        self.lib.GetRoot.restype = ctypes.c_void_p

        self.lib.InsertRec.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        self.lib.InsertRec.restype = ctypes.c_void_p

        self.lib.InsertOneStep.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.lib.InsertOneStep.restype = ctypes.c_void_p

        self.lib.SplitOneStep.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.lib.SplitOneStep.restype = ctypes.c_void_p

        self.lib.SplitWithLoc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.lib.SplitWithLoc.restype = ctypes.c_void_p

        self.lib.InsertWithLoc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.InsertWithLoc.restype = ctypes.c_void_p

        self.lib.IsLeaf.argtypes = [ctypes.c_void_p]
        self.lib.IsLeaf.restype = ctypes.c_int

        self.lib.GetNumberOfNonOverlapSplitLocs.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.GetNumberOfNonOverlapSplitLocs.restype = ctypes.c_int

        self.lib.SplitInMinOverlap.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.SplitInMinOverlap.restype = ctypes.c_void_p

        self.lib.SplitWithCandidateAction.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.lib.SplitWithCandidateAction.restype = ctypes.c_void_p

        self.lib.PrintTreeEntry.argtypes = [ctypes.c_void_p]
        self.lib.PrintTreeEntry.restype = ctypes.c_void_p

        self.lib.IsOverflow.argtypes = [ctypes.c_void_p]
        self.lib.IsOverflow.restype = ctypes.c_int

        self.lib.RetrieveStates.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.RetrieveStates.restype = ctypes.c_int

        self.lib.GetIndexSizeInMB.argtypes = [ctypes.c_void_p]
        self.lib.GetIndexSizeInMB.restype = ctypes.c_double

        self.lib.SetStartTimestamp.argtypes = [ctypes.c_void_p]
        self.lib.SetStartTimestamp.restype = ctypes.c_void_p

        self.lib.SetEndTimestamp.argtypes = [ctypes.c_void_p]
        self.lib.SetEndTimestamp.restype = ctypes.c_void_p

        self.lib.GetDurationInSeconds.argtypes = [ctypes.c_void_p]
        self.lib.GetDurationInSeconds.restype = ctypes.c_double

        self.lib.RetrieveSpecialStates.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.RetrieveSpecialStates.restype = ctypes.c_void_p

        self.lib.RetrieveShortSplitStates.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.RetrieveShortSplitStates.restype = ctypes.c_void_p

        self.lib.RetrieveSpecialInsertStates.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.RetrieveSpecialInsertStates.restype = ctypes.c_void_p

        self.lib.RetrieveSpecialInsertStates3.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.RetrieveSpecialInsertStates3.restype = ctypes.c_void_p

        self.lib.RetrieveSpecialInsertStates4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.RetrieveSpecialInsertStates4.restype = ctypes.c_void_p

        self.lib.RetrieveSpecialInsertStates6.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.RetrieveSpecialInsertStates6.restype = ctypes.c_void_p
        
        self.lib.DirectSplitWithReinsert.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.DirectSplitWithReinsert.restype = ctypes.c_void_p

        self.lib.RetrieveSpecialInsertStates7.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.RetrieveSpecialInsertStates7.restype = ctypes.c_void_p

        self.lib.RetrieveSpecialInsertStates7Fill0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.RetrieveSpecialInsertStates7Fill0.restype = ctypes.c_void_p

        self.lib.RetrieveSortedInsertStates.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        self.lib.RetrieveSortedInsertStates.restype = ctypes.c_void_p

        self.lib.GetActualSplitLocFromSortedPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.lib.GetActualSplitLocFromSortedPos.restype = ctypes.c_int

        self.lib.GetActualSplitDimFromSortedPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.lib.GetActualSplitDimFromSortedPos.restype = ctypes.c_int

        self.lib.RetrieveSortedSplitStates.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        self.lib.RetrieveSortedSplitStates.restype = ctypes.c_void_p

        self.lib.RetrieveZeroOVLPSplitSortedByPerimeterState.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.RetrieveZeroOVLPSplitSortedByPerimeterState.restype = ctypes.c_void_p 

        self.lib.RetrieveZeroOVLPSplitSortedByWeightedPerimeterState.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.RetrieveZeroOVLPSplitSortedByWeightedPerimeterState.restype = ctypes.c_void_p

        self.lib.GetMBR.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.GetMBR.restype = ctypes.c_void_p

        self.lib.GetNodeBoundary.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.GetNodeBoundary.restype = ctypes.c_void_p

        self.lib.GetMinAreaContainingChild.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.GetMinAreaContainingChild.restype = ctypes.c_int

        self.lib.GetMinAreaEnlargementChild.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.GetMinAreaEnlargementChild.restype = ctypes.c_int

        self.lib.GetMinMarginIncrementChild.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.GetMinMarginIncrementChild.restype = ctypes.c_int

        self.lib.GetMinOverlapIncrementChild.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.GetMinOverlapIncrementChild.restype = ctypes.c_int

        self.lib.InsertWithSortedLoc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.InsertWithSortedLoc.restype = ctypes.c_void_p

        self.lib.SplitWithSortedLoc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.lib.SplitWithSortedLoc.restype = ctypes.c_void_p 

        self.lib.GetNumberOfEnlargedChildren.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.GetNumberOfEnlargedChildren.restype = ctypes.c_int


        self.lib.PrintTree.argtypes = [ctypes.c_void_p]
        self.lib.PrintTree.restype = ctypes.c_void_p

        self.lib.PrintEntryNum.argtypes = [ctypes.c_void_p]
        self.lib.PrintEntryNum.restype = ctypes.c_void_p

        self.lib.DefaultInsert.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.DefaultInsert.restype = ctypes.c_void_p

        self.lib.DefaultSplit.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.DefaultSplit.restype = ctypes.c_void_p

        self.lib.DirectInsert.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.DirectInsert.restype = ctypes.c_void_p

        self.lib.RRInsert.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.RRInsert.restype = ctypes.c_void_p

        self.lib.RRSplit.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.RRSplit.restype = ctypes.c_void_p

        self.lib.DirectSplit.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.DirectSplit.restype = ctypes.c_void_p

        self.lib.Clear.argtypes = [ctypes.c_void_p]
        self.lib.Clear.restype = ctypes.c_void_p

        self.lib.PrintSortedSplitLocs.argtypes = [ctypes.c_void_p]
        self.lib.PrintSortedSplitLocs.restype = ctypes.c_void_p

        self.lib.AverageNodeArea.argtypes = [ctypes.c_void_p]
        self.lib.AverageNodeArea.restype = ctypes.c_double

        self.lib.AverageNodeChildren.argtypes = [ctypes.c_void_p]
        self.lib.AverageNodeChildren.restype = ctypes.c_double

        self.lib.TotalTreeNode.argtypes = [ctypes.c_void_p]
        self.lib.TotalTreeNode.restype = ctypes.c_int

        self.lib.CopyTree.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.CopyTree.restype = ctypes.c_void_p

        self.lib.TryInsert.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.TryInsert.restype = ctypes.c_int

        self.lib.TreeHeight.argtypes = [ctypes.c_void_p]
        self.lib.TreeHeight.restype = ctypes.c_int

        self.lib.SetDebug.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.SetDebug.restype = ctypes.c_void_p

        self.lib.GetQueryResult.argtypes = [ctypes.c_void_p]
        self.lib.GetQueryResult.restype = ctypes.c_int

        self.lib.GetChildNum.argtypes = [ctypes.c_void_p]
        self.lib.GetChildNum.restype = ctypes.c_int

        self.tree = self.lib.ConstructTree(max_entry, min_entry)
        self.strategy_map = {"INS_AREA":0, "INS_MARGIN":1, "INS_OVERLAP":2, "INS_RANDOM":3, "SPL_MIN_AREA":0, "SPL_MIN_MARGIN":1, "SPL_MIN_OVERLAP":2, "SPL_QUADRATIC":3, "SPL_GREENE":4}

        self.insert_strategy = None
        self.split_strategy = None
        self.max_entry = max_entry
        self.min_entry = min_entry

        self.debug = False

    def PrintEntry(self):
        self.lib.PrintTreeEntry(self.tree)

    def GetIndexSizeInMB(self):
        return self.lib.GetIndexSizeInMB(self.tree)

    def SetStartTimestamp(self):
        self.lib.SetStartTimestamp(self.tree)

    def SetEndTimestamp(self):
        self.lib.SetEndTimestamp(self.tree)

    def GetDurationInSeconds(self):
        return self.lib.GetDurationInSeconds(self.tree)

    def RandomQuery(self, width, height):
        boundary_c = (ctypes.c_double * 4)()
        self.lib.GetMBR(self.tree, boundary_c)
        boundary = np.ctypeslib.as_array(boundary_c)

    def PrintSortedLocs(self):
        self.lib.PrintSortedSplitLocs(self.tree)

    def GetActualSplitFromSortedPos(self, pos):
        split_loc = self.lib.GetActualSplitLocFromSortedPos(self.tree, self.ptr, pos)
        split_dim = self.lib.GetActualSplitDimFromSortedPos(self.tree, self.ptr, pos)
        return split_dim, split_loc

    def SetInsertStrategy(self, strategy):
        self.insert_strategy = strategy
        if strategy == 'RANDOM':
            return
        if strategy not in self.strategy_map:
            print("Insert strategy {} does not exist.".format(strategy))
            exit()
        self.lib.SetDefaultInsertStrategy(self.tree, self.strategy_map[strategy])


    
    def SetSplitStrategy(self, strategy):
        self.split_strategy = strategy
        if strategy == 'RANDOM':
            return
        if strategy not in self.strategy_map:
            print("Split strategy {} does not exist.".format(strategy))
            exit()
        self.lib.SetDefaultSplitStrategy(self.tree, self.strategy_map[strategy])

    def PrepareRectangle(self, left, right, bottom, top):
        self.rec = self.lib.InsertRec(self.tree, left, right, bottom, top)
        self.ptr = self.lib.GetRoot(self.tree)


    def GetNumberOfNonOverlapSplitLocs(self):
        if self.lib.IsOverflow(self.ptr) == 0:
            return None
        num = self.lib.GetNumberOfNonOverlapSplitLocs(self.tree, self.ptr)
        return num

    def GetNumberOfEnlargedChildren(self):
        if self.lib.IsLeaf(self.ptr):
            return None
        num = self.lib.GetNumberOfEnlargedChildren(self.tree, self.ptr, self.rec)
        return num

    def RetrieveSplitStates(self):
        #if self.debug:
        #    self.lib.SetDebug(self.tree, 1)
        #else:
        #    self.lib.SetDebug(self.tree, 0)
        is_overflow = self.lib.IsOverflow(self.ptr)
        #if self.debug:
        #    print(is_overflow)
        if is_overflow == 0:
            return None, False
        state_length = 25
        state_c = (ctypes.c_double * state_length)()
        is_valid = self.lib.RetrieveStates(self.tree, self.ptr, state_c)
        states = np.ctypeslib.as_array(state_c)
        return states, is_valid

    def RetrieveSortedSplitStates(self, topk):
        is_overflow = self.lib.IsOverflow(self.ptr)
        if is_overflow == 0:
            return None
        state_length =  topk * 5 
        state_c = (ctypes.c_double * state_length)()
        self.lib.RetrieveSortedSplitStates(self.tree, self.ptr, topk, state_c)
        states = np.ctypeslib.as_array(state_c)
        return states

    def NeedSplit(self):
        if self.lib.IsOverflow(self.ptr) == 1:
            return True
        else:
            return False

    def RetrieveShortSplitStates(self):
        is_overflow = self.lib.IsOverflow(self.ptr)
        if is_overflow == 0:
            return None
        state_length = 5 * 12
        state_c = (ctypes.c_double * state_length)()
        self.lib.RetrieveShortSplitStates(self.tree, self.ptr, state_c)
        states = np.ctypeslib.as_array(state_c)
        return states

    def RetrieveZeroOVLPSplitSortedByPerimeterState(self):
        state_length = 4 * 2
        state_c = (ctypes.c_double * state_length)()
        self.lib.RetrieveZeroOVLPSplitSortedByPerimeterState(self.tree, self.ptr, state_c)
        states = np.ctypeslib.as_array(state_c)
        return states

    def RetrieveZeroOVLPSplitSortedByWeightedPerimeterState(self):
        state_length = 4 * 2
        state_c = (ctypes.c_double * state_length)()
        self.lib.RetrieveZeroOVLPSplitSortedByWeightedPerimeterState(self.tree, self.ptr, state_c)
        states = np.ctypeslib.as_array(state_c)
        return states

    def RetrieveSpecialSplitStates(self):

        is_overflow = self.lib.IsOverflow(self.ptr)
        if is_overflow == 0:
            return None
        state_length = 5 * 12 * 4
        state_c     = (ctypes.c_double * state_length)()
        self.lib.RetrieveSpecialStates(self.tree, self.ptr, state_c)
        states = np.ctypeslib.as_array(state_c)
        return states

    def GetMinAreaContainingChild(self):
        if self.lib.IsLeaf(self.ptr):
            return None
        child = self.lib.GetMinAreaContainingChild(self.tree, self.ptr, self.rec)
        if child < 0:
            return None
        else:
            return child

    def GetMinAreaEnlargementChild(self):
        if self.lib.IsLeaf(self.ptr):
            return None
        child = self.lib.GetMinAreaEnlargementChild(self.tree, self.ptr, self.rec)
        return child

    def GetMinMarginIncrementChild(self):
        if self.lib.IsLeaf(self.ptr):
            return None
        child = self.lib.GetMinMarginIncrementChild(self.tree, self.ptr, self.rec)
        return child

    def GetMinOverlapIncrementChild(self):
        if self.lib.IsLeaf(self.ptr):
            return None
        child = self.lib.GetMinOverlapIncrementChild(self.tree, self.ptr, self.rec)
        return child

    def RetrieveSpecialInsertStates(self):
        if self.lib.IsLeaf(self.ptr):
            return None
        state_length = 6 + 9 * self.max_entry
        state_c = (ctypes.c_double * state_length)()
        self.lib.RetrieveSpecialInsertStates(self.tree, self.ptr, self.rec, state_c)
        states = np.ctypeslib.as_array(state_c)
        return states

    def RetrieveSpecialInsertStates3(self):
        if self.lib.IsLeaf(self.ptr):
            return None
        state_length = 3 * self.max_entry
        state_c = (ctypes.c_double * state_length)()
        self.lib.RetrieveSpecialInsertStates3(self.tree, self.ptr, self.rec, state_c)
        states = np.ctypeslib.as_array(state_c)
        return states

    def RetrieveSpecialInsertStates4(self):
        if self.lib.IsLeaf(self.ptr):
            return None
        state_length = 4 * self.max_entry
        state_c = (ctypes.c_double * state_length)()
        self.lib.RetrieveSpecialInsertStates4(self.tree, self.ptr, self.rec, state_c)
        states = np.ctypeslib.as_array(state_c)
        return states

    def RetrieveSpecialInsertStates6(self):
        if self.lib.IsLeaf(self.ptr):
            return None
        state_length = 6 * self.max_entry
        state_c     = (ctypes.c_double * state_length)()
        self.lib.RetrieveSpecialInsertStates6(self.tree, self.ptr, self.rec, state_c)
        states = np.ctypeslib.as_array(state_c)
        return states

    def RetrieveSpecialInsertStates7(self):
        if self.lib.IsLeaf(self.ptr):
            return None
        state_length = 7 * self.max_entry
        state_c     = (ctypes.c_double * state_length)()
        self.lib.RetrieveSpecialInsertStates7(self.tree, self.ptr, self.rec, state_c)
        states = np.ctypeslib.as_array(state_c)
        return states

    def RetrieveSpecialInsertStates7Fill0(self):
        if self.lib.IsLeaf(self.ptr):
            return None
        state_length = 7 * self.max_entry
        state_c = (ctypes.c_double * state_length)()
        self.lib.RetrieveSpecialInsertStates7Fill0(self.tree, self.ptr, self.rec, state_c)
        states = np.ctypeslib.as_array(state_c)
        return states

    def RetrieveSortedInsertStates(self, action_space, rl_type):
        #action_space = 5 or 10
        #rl_type=0: RL for enlarged children and deterministic for non-enlarged children.
        #rl_type=1: RL for non-enlarged children and deterministic for enlarged children.
        if self.lib.IsLeaf(self.ptr):
            return None
        state_length = 4 * action_space
        state_c = (ctypes.c_double * state_length)()
        self.lib.RetrieveSortedInsertStates(self.tree, self.ptr, self.rec, action_space, rl_type, state_c)
        states = np.ctypeslib.as_array(state_c)
        return states



    def SplitInMinOverlap(self):
        self.next_ptr = self.lib.SplitInMinOverlap(self.tree, self.ptr)
        self.ptr = self.next_ptr

    def SplitWithCandidateAction(self, loc):
        self.next_ptr = self.lib.SplitWithCandidateAction(self.tree, self.ptr, loc)
        self.ptr = self.next_ptr


        

    def UniformRandomQuery(self, wr, hr):
        boundary_c = (ctypes.c_double * 4)()
        self.lib.GetMBR(self.tree, boundary_c)
        boundary = np.ctypeslib.as_array(boundary_c)
        width = (boundary[1] - boundary[0]) * wr
        height = (boundary[3] - boundary[2]) * hr
        x = random.uniform(boundary[0], boundary[1])
        y = random.uniform(boundary[2], boundary[3])
        return [x, x+width, y, y+height]

    def UniformDenseRandomQuery(self, wr, hr, object_boundary):
        boundary_c = (ctypes.c_double * 4)()
        self.lib.GetMBR(self.tree, boundary_c)
        boundary = np.ctypeslib.as_array(boundary_c)
        width = (boundary[1] - boundary[0]) * wr
        height = (boundary[3] - boundary[2]) * hr

        x = random.uniform(object_boundary[0] - 2 * width, object_boundary[1] + 2 * width)
        y = random.uniform(object_boundary[2] - 2 * height, object_boundary[3] + 2 * height)
        return [x, x+width, y, y+height]

    def CountChildNodes(self):
        child_num = self.lib.GetChildNum(self.ptr)
        return child_num

    def IsLeaf(self):
        if self.lib.IsLeaf(self.ptr):
            return True
        else:
            return False

    def InsertWithLoc(self, loc):
        self.next_ptr = self.lib.InsertWithLoc(self.tree, self.ptr, loc, self.rec)
        if self.lib.IsLeaf(self.ptr):
            return True
        else:
            self.ptr = self.next_ptr
            return False

    def InsertWithSortedLoc(self, loc):
        self.next_ptr = self.lib.InsertWithSortedLoc(self.tree, self.ptr, loc, self.rec)
        if self.lib.IsLeaf(self.ptr):
            return True
        else:
            self.ptr = self.next_ptr
            return False
        

    def Query(self, boundary):
        node_access = self.lib.QueryRectangle(self.tree, boundary[0], boundary[1], boundary[2], boundary[3])
        return node_access

    def KNNQuery(self, x, y, k):
        node_access = self.lib.KNNQuery(self.tree, x, y, k)
        return node_access
        
    def QueryResult(self):
        return self.lib.GetQueryResult(self.tree)

    def AccessRate(self, boundary):
        node_access = self.lib.QueryRectangle(self.tree, boundary[0], boundary[1], boundary[2], boundary[3])
        #total_node = self.lib.TotalTreeNode(self.tree)
        #print('node_access', node_access)
        height = self.lib.TreeHeight(self.tree)
        if height == 0:
            print('height is 0')
            input()
        return 1.0 * node_access / height #total_node

    def AverageNodeArea(self):
        return self.lib.AverageNodeArea(self.tree)

    def AverageNodeChildren(self):
        return self.lib.AverageNodeChildren(self.tree)

    def TotalTreeNodeNum(self):
        return self.lib.TotalTreeNode(self.tree)

    def DefaultInsert(self, boundary):
        self.PrepareRectangle(boundary[0], boundary[1], boundary[2], boundary[3])
        self.lib.DefaultInsert(self.tree, self.rec)

    def DefaultSplit(self):
        self.lib.DefaultSplit(self.tree, self.ptr)



    def CopyTree(self, tree):
        self.lib.CopyTree(self.tree, tree)

    def TryInsert(self, boundary):
        self.PrepareRectangle(boundary[0], boundary[1], boundary[2], boundary[3])
        is_success = self.lib.TryInsert(self.tree, self.rec)
        if is_success == 0:
            return False
        else:
            return True


    def InsertOneStep(self, strategy_id):
        self.next_ptr = self.lib.InsertOneStep(self.tree, self.rec, self.ptr, strategy_id)
        if self.lib.IsLeaf(self.ptr):
            return True
        else:
            self.ptr = self.next_ptr
            return False

    def SplitOneStep(self, strategy_id):
        if self.lib.IsOverflow(self.ptr):
            self.next_ptr = self.lib.SplitOneStep(self.tree, self.ptr, strategy_id)
            self.ptr = self.next_ptr
            return False
        else:
            return True


    def SplitWithLoc(self, loc):
        if self.lib.IsOverflow(self.ptr):
            self.next_ptr = self.lib.SplitWithLoc(self.tree, self.ptr, loc)
            self.ptr = self.next_ptr
            return False
        else: 
            return True
    
    def SplitWithSortedLoc(self, loc):
        if self.lib.IsOverflow(self.ptr):
            self.next_ptr = self.lib.SplitWithSortedLoc(self.tree, self.ptr, loc)
            self.ptr = self.next_ptr
            return False
        else:
            return True

    def Clear(self):
        self.lib.Clear(self.tree)

    def Print(self):
        self.lib.PrintTree(self.tree)

    def PrintEntryNum(self):
        self.lib.PrintEntryNum(self.tree);

    def DirectInsert(self, boundary):
        self.PrepareRectangle(boundary[0], boundary[1], boundary[2], boundary[3])
        self.ptr = self.lib.DirectInsert(self.tree, self.rec)

    def DirectRRInsert(self, boundary):
        self.PrepareRectangle(boundary[0], boundary[1], boundary[2], boundary[3])
        self.ptr = self.lib.RRInsert(self.tree, self.rec)

    def DirectSplitWithReinsert(self):
        self.lib.DirectSplitWithReinsert(self.tree, self.ptr)

    def RandomDirectInsert(self, boundary):
        self.PrepareRectangle(boundary[0], boundary[1], boundary[2], boundary[3])
        while not self.lib.IsLeaf(self.ptr):
            strategy_id = random.randint(0, 3)
            self.next_ptr = self.lib.InsertOneStep(self.tree, self.rec, self.ptr, strategy_id)
            self.ptr = self.next_ptr
        strategy_id = random.randint(0, 3)
        self.lib.InsertOneStep(self.tree, self.rec, self.ptr, strategy_id)

    def RandomDirectSplit(self):
        while self.lib.IsOverflow(self.ptr):
            strategy_id = random.randint(0, 4)
            self.next_ptr = self.lib.SplitOneStep(self.tree, self.ptr, strategy_id)
            self.ptr = self.next_ptr

    def DirectSplit(self):
        self.lib.DirectSplit(self.tree, self.ptr)

    def DirectRRSplit(self):
        self.lib.RRSplit(self.tree, self.ptr)

    def GetNodeBoundary(self):
        boundary_c = (ctypes.c_double * 4)()
        self.lib.GetNodeBoundary(self.ptr, boundary_c)
        boundary = np.ctypeslib.as_array(boundary_c)
        return boundary


if __name__ == '__main__':
    ls = [0.0, 3.0, 11.0, 8.0, 5.0, 13.0, 15.0, 9.0, 1.0, 13.0, 10.0, 14.0]
    rs = [3.0,6.0,13.0,10.0,7.0,15.0,18.0,12.0,4.0,16.0,13.0,16.0]
    bs = [0.0,6.0,2.0,7.0,4.0,8.0,5.0,1.0,3.0,1.0,9.0,9.0]
    ts = [2.0,8.0,4.0,10.0,7.0,11.0,7.0,3.0,5.0,4.0,11.0,12.0]

    tree = RTree(3, 2)
    tree.SetSplitStrategy('SPL_MIN_AREA')
    tree.SetInsertStrategy('INS_AREA')
    tree.SetStartTimestamp()
    for i in range(len(ls)):
        tree.DirectInsert((ls[i], rs[i], bs[i], ts[i]))
        tree.DirectSplitWithReinsert()
        #tree.DefaultInsert((ls[i], rs[i], bs[i], ts[i]))
        #tree.Print()

    
    tree.SetEndTimestamp();
    print(tree.GetDurationInSeconds(), 'seconds')
    print(tree.GetIndexSizeInMB(), 'MB') 
    l, r, b, t = tree.UniformRandomQuery(2.0, 4.0)
    print(l, r, b, t)
    access = tree.Query((l, r, b, t))
    print("{} nodes are accessed\n".format(access))





