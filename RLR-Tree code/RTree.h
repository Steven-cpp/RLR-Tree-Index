//
// Created by Kaiyu on 2020/7/25.
//

#ifndef RTREE_RTREE_H
#define RTREE_RTREE_H
#include<list>
#include<vector>
#include<algorithm>
#include<cfloat>
#include<cmath>
#include<chrono>
#include<assert.h>
#include<iostream>
#include<queue>
#include<fstream>

using std::list;
using std::vector;
using std::pair;
using std::make_pair;
using std::min;
using std::max;
using std::abs;
using std::sort;
using std::cout;
using std::ceil;
using std::endl;
using std::exp;
using std::ofstream;
using std::chrono::high_resolution_clock;
using std::priority_queue;

enum INSERT_STRATEGY{
    INS_AREA, INS_MARGIN, INS_OVERLAP, INS_RANDOM
};
enum SPLIT_STRATEGY{
    SPL_MIN_AREA, SPL_MIN_MARGIN, SPL_MIN_OVERLAP, SPL_QUADRATIC, SPL_GREENE
};

enum TYPE {
	REC, TREE_NODE
};

enum INSERT_STATE_TYPE{
	RL_FOR_ENLARGED_CHILDREN, RL_FOR_CONTAINING_CHILDREN
};

struct Point{
public:
    double x;
    double y;
public:
    double get_left(){return x;};
    double get_right(){return x;};
    double get_bottom(){return y;};
    double get_top(){return y;};
};



class Rectangle{
public:
    double left_;
    double right_;
    double bottom_;
    double top_;

	unsigned int id_;

public:
	Rectangle();
	Rectangle(const Rectangle& rectangle);
	Rectangle(double l, double r, double b, double t) :left_(l), right_(r), top_(t), bottom_(b) {};

	double Left() const { return left_; };
	double Right() const { return right_; };
	double Top() const { return top_; };
	double Bottom() const { return bottom_; };
	double Area() const { return (top_ - bottom_) * (right_ - left_); };
	double Perimeter() const { return 2 * (top_ - bottom_ + right_ - left_); };


	bool Contains(const Rectangle* rec);
	bool IsOverlap(Rectangle* rec);
	bool IsValid();

    void Set(const Rectangle& rectangle);
	void Set(double l, double r, double b, double t);
    void Include(const Rectangle& rectangle);
	Rectangle Merge(const Rectangle& rectangle);
};


//Rectangle Merge(const Rectangle& rectangle1, const Rectangle& rectangle2);
//double Overlap(const Rectangle& rectangle1, const Rectangle& rectangle2);


//template<class T>
//double SplitArea(T* t1, T* t2);
//
//template<class T>
//double SplitOverlap(T* t1, T* t2);
//
//template<class T> 
//double SplitPerimeter(T* t1, T* t2);

double SplitArea(const Rectangle& rectangle1, const Rectangle& rectangle2);
double SplitOverlap(const Rectangle& rectangle1, const Rectangle& rectangle2);
double SplitPerimeter(const Rectangle& rectangle1, const Rectangle& rectangle2);




bool IsContained(const Rectangle& rectangle1, const Rectangle& rectangle2);
bool IsOverlapped(const Rectangle& rectangle1, const Rectangle& rectangle2);

class TreeNode : public Rectangle{
public:
    int father;
    //vector<TreeNode*> children;
	vector<int> children;

    int entry_num;
    bool is_overflow;
	bool is_leaf;

	double origin_center[2];

    static int maximum_entry;
    static int minimum_entry;
	static double RR_s;

public:
    TreeNode();
	TreeNode(TreeNode* node);
    bool CopyChildren(const vector<TreeNode*>& nodes);
	bool CopyChildren(const vector<int>& node_ids);
    bool AddChildren(TreeNode* node);
	bool AddChildren(int node_id);
};


template<class T>
Rectangle MergeRange(const vector<T*>& entries, const int start_idx, const int end_idx);


template<class T>
int FindMinimumSplit(const vector<T*>& entries, double(*score_func1)(const Rectangle &, const Rectangle &),
	double(*score_func2)(const Rectangle &, const Rectangle &), double& min_value1, double& min_value2, Rectangle& rec1, Rectangle& rec2);

struct Stats {
	int node_access;
	int action_history[5];
	void Reset(); 
	void TakeAction(int action);
};

struct SplitLocation{
	double perimeter1;
	double perimeter2;
	double area1;
	double area2;
	double overlap;
	int location;
	int dimension; //0 x-low, 1, y-low, 2, x-high, 3, y-high
};

class RTree {
public:
    //list<TreeNode*> tree_nodes_;
	//list<Rectangle*> objects_;
	vector<TreeNode*> tree_nodes_;
	vector<Rectangle*> objects_;
    int root_;
	int height_;
	list<int> history;
	high_resolution_clock::time_point start_point;
	high_resolution_clock::time_point end_point;

    //parameters
    INSERT_STRATEGY insert_strategy_;
    SPLIT_STRATEGY split_strategy_;

	Stats stats_;
	int result_count;

	int debug;

	double RR_s;
	double RR_y1;
	double RR_ys;
	int ff_cnt = 1;

	vector<TreeNode*> tmp_sorted_children;
	vector<pair<double, pair<bool, int> > > sorted_split_loc;
	vector<SplitLocation> split_locations;
	vector<int> candidate_split_action;

public:

	RTree();


	void Copy(RTree* tree);
	TreeNode* Root();
	Rectangle* InsertRectangle(double left, double right, double bottom, double top);

	TreeNode* RRInsert(Rectangle* rectangle, TreeNode* tree_node);
	TreeNode* RRSplit(TreeNode* tree_node);

    TreeNode* InsertStepByStep(const Rectangle* rectangle, TreeNode* tree_node, INSERT_STRATEGY strategy);
    TreeNode* InsertStepByStep(const Rectangle* rectangle, TreeNode* tree_node);

	TreeNode* TryInsertStepByStep(const Rectangle* rectangle, TreeNode* tree_node);

	void Recover(RTree* rtree);
	

    TreeNode* SplitStepByStep(TreeNode* tree_node, SPLIT_STRATEGY strategy);
    TreeNode* SplitStepByStep(TreeNode* tree_node);

	TreeNode* SplitInLoc(TreeNode* tree_node, int loc);
	TreeNode* SplitInLoc(TreeNode* tree_node, int loc, int dim);
	TreeNode* SplitWithCandidateAction(TreeNode* tree_node, int loc);
	TreeNode* InsertInLoc(TreeNode* tree_node, int loc, Rectangle* rec);
	TreeNode* InsertInSortedLoc(TreeNode* tree_node, int sorted_loc, Rectangle* rec);
	TreeNode* SplitInSortedLoc(TreeNode* tree_node, int sorted_loc);
	void PrepareSplitLocations(TreeNode* tree_node);

	void RetrieveForReinsert(TreeNode* tree_node, list<int>& candidates);
	void UpdateMBRForReinsert(TreeNode* tree_node);  

    //Rectangle MergeRectangleList(const vector<pair<double, Rectangle> >& rectangle_list, double min_value);

    //void FindMinimumSplit(const vector<TreeNode*>& entries, double(*score_func1)(const Rectangle &, const Rectangle &),
            //double(*score_func2)(const Rectangle &, const Rectangle &), double& value1, double& value2, vector<TreeNode*>& child1, vector<TreeNode*>& child2);
	//void FindMinimumSplit(const vector<Rectangle*> &entries, double(*score_func1)(const Rectangle &, const Rectangle &),
		//double(*score_func2)(const Rectangle &, const Rectangle &), double &min_value1, double &min_value2, vector<Rectangle>& child1, vector<Rectangle>& child2);

	//void FindMinimumSplit(void* entries, double(*score_func1)(const Rectangle &, const Rectangle &),
		//double(*score_func2)(const Rectangle &, const Rectangle &), double& value1, double& value2, int& split_loc);
	
	//Rectangle MergeRange(const vector<TreeNode*>& entries, const int start_idx, const int end_idx);
	//Rectangle MergeRange(const vector<Rectangle*>& entries, const int start_idx, const int end_idx);
    TreeNode* CreateNode();


    void SortChildrenByArea(TreeNode* tree_node);
    void SortChildrenByMarginArea(TreeNode* tree_node, Rectangle* rec);
	void SortSplitLocByPerimeter(TreeNode* tree_node);
	int Query(Rectangle& rectangle);
	int KNNQuery(double x, double y, int k, vector<int>& results);
	double MinDistanceToNode(double x, double y, int tree_node_id);
	double MinDistanceToRec(double x, double y, int rec_id);
	void GetSplitStates(TreeNode* tree_node, double* states);
	void GetShortSplitStates(TreeNode* tree_node, double* states);
	void GetInsertStates(TreeNode* tree_node, Rectangle* rec, double* states);
	void GetInsertStates3(TreeNode* tree_node, Rectangle* rec, double* states);
	void GetInsertStates4(TreeNode* tree_node, Rectangle* rec, double* states);
	void GetInsertStates6(TreeNode* tree_node, Rectangle* rec, double* states);
	void GetInsertStates7(TreeNode* tree_node, Rectangle* rec, double* states);
	void GetInsertStates7Fill0(TreeNode* tree_node, Rectangle* rec, double* states);
	void GetSortedInsertStates(TreeNode* tree_node, Rectangle* rec, double* states, int topk, INSERT_STATE_TYPE state_type);
	void GetSortedSplitStates(TreeNode* tree_node, double* states, int topk);
	int GetNumberOfEnlargedChildren(TreeNode* tree_node, Rectangle* rec);

	int GetMinAreaContainingChild(TreeNode* tree_node, Rectangle* rec);
	int GetMinAreaEnlargementChild(TreeNode* tree_node, Rectangle* rec);
	int GetMinMarginIncrementChild(TreeNode* tree_node, Rectangle* rec);
	int GetMinOverlapIncrementChild(TreeNode* tree_node, Rectangle* rec);

	

	void SplitAREACost(TreeNode* tree_node, vector<double>& values, Rectangle& rec1, Rectangle& rec2);
	void SplitMARGINCost(TreeNode* tree_node, vector<double>& values, Rectangle& rec1, Rectangle& rec2);
	void SplitOVERLAPCost(TreeNode* tree_node, vector<double>& values, Rectangle& rec1, Rectangle& rec2);
	void SplitGREENECost(TreeNode* tree_node, vector<double>& values, Rectangle& rec1, Rectangle& rec2);
	void SplitQUADRATICCost(TreeNode* tree_node, vector<double>& values, Rectangle& rec1, Rectangle& rec2);

	void Print();
	void PrintEntryNum();
};




extern "C"{

	int RetrieveStates(RTree* tree, TreeNode* tree_node, double* states);

	void RetrieveSpecialStates(RTree* tree, TreeNode* tree_node, double* states);
	void RetrieveShortSplitStates(RTree* tree, TreeNode* tree_node, double* states);

	void RetrieveSpecialInsertStates(RTree* tree, TreeNode* tree_node, Rectangle* rec, double* states);
	void RetrieveSpecialInsertStates3(RTree* tree, TreeNode* tree_node, Rectangle* rec, double* states);
	void RetrieveSpecialInsertStates4(RTree* tree, TreeNode* tree_node, Rectangle* rec, double* states);
	void RetrieveSpecialInsertStates6(RTree* tree, TreeNode* tree_node, Rectangle* rec, double* states);
	void RetrieveSpecialInsertStates7(RTree* tree, TreeNode* tree_node, Rectangle* rec, double* states);
	void RetrieveSpecialInsertStates7Fill0(RTree* tree, TreeNode* tree_node, Rectangle* rec, double* states);

	void RetrieveSortedInsertStates(RTree* tree, TreeNode* tree_node, Rectangle* rec, int topk, int state_type, double* states);
	void RetrieveSortedSplitStates(RTree* tree, TreeNode* tree_node, int topk, double* states);
	void RetrieveZeroOVLPSplitSortedByPerimeterState(RTree* tree, TreeNode* tree_noe, double* states);
	void RetrieveZeroOVLPSplitSortedByWeightedPerimeterState(RTree* tree, TreeNode* tree_node, double* states);

	RTree* ConstructTree(int max_entry, int min_entry);

	void SetDefaultInsertStrategy(RTree* rtree, int strategy);
	void SetDefaultSplitStrategy(RTree* rtree, int strategy);
	int QueryRectangle(RTree* rtree, double left, double right, double bottom, double top);
	int KNNQuery(RTree* rtree, double x, double y, int k);

	int GetMinAreaContainingChild(RTree* rtree, TreeNode* tree_node, Rectangle* rec);
	int GetMinAreaEnlargementChild(RTree* rtree, TreeNode* tree_node, Rectangle* rec);
	int GetMinMarginIncrementChild(RTree* rtree, TreeNode* tree_node, Rectangle* rec);
	int GetMinOverlapIncrementChild(RTree* rtree, TreeNode* tree_node, Rectangle* rec);

	int GetNumberOfEnlargedChildren(RTree* rtree, TreeNode* tree_node, Rectangle* rec);

	int GetNumberOfNonOverlapSplitLocs(RTree* rtree, TreeNode* tree_node);

	int GetQueryResult(RTree* rtree);
	
	TreeNode* GetRoot(RTree* rtree);

	void CopyTree(RTree* tree, RTree* from_tree);

	Rectangle* InsertRec(RTree* rtree, double left, double right, double bottom, double top);
	TreeNode* InsertOneStep(RTree* rtree, Rectangle* rec, TreeNode* node, int strategy);

	TreeNode* DirectInsert(RTree* rtree, Rectangle* rec);

	void DirectSplit(RTree* rtree, TreeNode* tree_node);
	void DirectSplitWithReinsert(RTree* rtree, TreeNode* tree_node);

	int TryInsert(RTree* rtree, Rectangle* rec);

	TreeNode* InsertWithLoc(RTree* tree, TreeNode* tree_node, int loc, Rectangle* rec);
	TreeNode* InsertWithSortedLoc(RTree* tree, TreeNode* tree_node, int sorted_loc, Rectangle* rec);
	TreeNode* SplitWithSortedLoc(RTree* rtree, TreeNode* node, int sorted_loc);

	TreeNode* SplitInMinOverlap(RTree* rtree, TreeNode* tree_node);

	int GetActualSplitLocFromSortedPos(RTree* rtree, TreeNode* node, int sorted_loc);
	int GetActualSplitDimFromSortedPos(RTree* rtree, TreeNode* node, int sorted_loc);
	void PrintSortedSplitLocs(RTree* rtree);

	void DefaultInsert(RTree* rtree, Rectangle* rec);

	void DefaultSplit(RTree* rtree, TreeNode* tree_node);


	TreeNode* RRInsert(RTree* rtree, Rectangle* rec);

	void RRSplit(RTree* rtree, TreeNode* node);

	void DefaultInsertWithHistory(RTree* rtree, Rectangle* rec);

	TreeNode* SplitOneStep(RTree* rtree, TreeNode* node, int strategy);

	TreeNode* SplitWithLoc(RTree* rtree, TreeNode* node, int loc);

	TreeNode* SplitWithCandidateAction(RTree* rtree, TreeNode* node, int loc);

	void PrintTreeEntry(RTree* rtree);
	
	double GetIndexSizeInMB(RTree* rtree);

	void SetStartTimestamp(RTree* rtree);
	
	void SetEndTimestamp(RTree* rtree);

	double GetDurationInSeconds(RTree* rtree);

	int IsLeaf(TreeNode* node);
	int IsOverflow(TreeNode* node);

	int GetChildNum(TreeNode* node);

	void GetMBR(RTree* rtree, double* boundary);
	void GetNodeBoundary(TreeNode* node, double* boundary);

	void PrintTree(RTree* rtree);

	void PrintEntryNum(RTree* rtree);

	int TotalTreeNode(RTree* rtree);
	double AverageNodeArea(RTree* rtree);
	double AverageNodeChildren(RTree* rtree);
	int TreeHeight(RTree* rtree);

	void SetDebug(RTree* rtree, int value);
	

	void Clear(RTree* rtree);

	void SetRR_s(double s_value);
}


#endif //RTREE_RTREE_H
