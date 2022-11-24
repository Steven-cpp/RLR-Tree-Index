Key points to take note:
1. run "sh compile.sh" first
2. parameter settings are located near the top of each py file and can be adjusted
3. the 3 synthetic datasets can be manually generated and the 2 real datasets can be obtained from OpenStreetMap
4. For the implementation of LISA, we follow the instruction in the source code shared by the authors. (https://github.com/pfl-cs/LISA)

#################
#               #
# (1) Baselines #
#               #
#################

Experimental results on R*-Tree and RR*-Tree can be reproduced by running RTree_RRstar_test_cpp.py (for range queries) and RTree_RRstar_test_cpp_KNN.py (for KNN queries).
Note that for R*-Tree (RR*-Tree), the ChooseSubtree and Split functions are "DirectInsert" (DirectRRInsert) and "DirectSplitWithReinsert" (DirectRRSplit) respectively.

##############################
#                            #
# (2) Combined model testing #
#                            #
##############################

For the convenience of reproducing results, we provide trained models 0_insertion_gaussian_rtree_k2.mdl (ChooseSubtree) and gaussian_k2.mdl (Split) for the Gaussian dataset.
Experimental results for the combined model can be obtained by running combined_model.py.

#################################
#                               #
# (3) Training RL ChooseSubtree #
#                               #
#################################

RL ChooseSubtree model is trained using model_ChooseSubtree.py

#################################
#                               #
# (4) Training RL Split         #
#                               #
#################################

RL Split model is trained using model_Split.py


libtorch --> 更快,但难度大













