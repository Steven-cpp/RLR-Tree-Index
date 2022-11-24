//
// Created by Kaiyu on 2020/7/25.
//

#include "RTree.h"


int TreeNode::maximum_entry = 100;
int TreeNode::minimum_entry = 40;
double TreeNode::RR_s = 0.5;

Rectangle::Rectangle() {
	left_ = 1;
	right_ = -1;
	bottom_ = 1;
	top_ = -1;
}

Rectangle::Rectangle(const Rectangle& rectangle) {
	left_ = rectangle.Left();
	right_ = rectangle.Right();
	bottom_ = rectangle.Bottom();
	top_ = rectangle.Top();
}

void Rectangle::Include(const Rectangle &rectangle) {
	
	left_ = min(left_, rectangle.Left());
	right_ = max(right_, rectangle.Right());
	bottom_ = min(bottom_, rectangle.Bottom());
	top_ = max(top_, rectangle.Top());
}

Rectangle Rectangle::Merge(const Rectangle& rectangle) {
	double l = min(left_, rectangle.Left());
	double r = max(right_, rectangle.Right());
	double b = min(bottom_, rectangle.Bottom());
	double t = max(top_, rectangle.Top());
	Rectangle rec(l, r, b, t);
	return rec;
}

void Rectangle::Set(const Rectangle &rectangle) {
	left_ = rectangle.Left();
	right_ = rectangle.Right();
	bottom_ = rectangle.Bottom();
	top_ = rectangle.Top();
}

void Rectangle::Set(double l, double r, double b, double t) {
	left_ = l;
	right_ = r;
	bottom_ = b;
	top_ = t;
}

bool Rectangle::IsValid() {
	if (right_ >= left_ && top_ >= bottom_) {
		return true;
	}
	else {
		return false;
	}
}


Rectangle Merge(const Rectangle& rectangle1, const Rectangle& rectangle2) {
	return Rectangle(min(rectangle1.Left(), rectangle2.Left()), max(rectangle1.Right(), rectangle2.Right()), min(rectangle1.Bottom(), rectangle2.Bottom()), max(rectangle1.Top(), rectangle2.Top()));
}

Rectangle Overlap(const Rectangle& rectangle1, const Rectangle& rectangle2) {
	Rectangle ovlp;
	double left = max(rectangle1.left_, rectangle2.left_);
	double right = min(rectangle1.right_, rectangle2.right_);
	double bottom = max(rectangle1.bottom_, rectangle2.bottom_);
	double top = min(rectangle1.top_, rectangle2.top_);
	if (left < right && bottom < top) {
		ovlp.Set(left, right, bottom, top);
	}
	return ovlp;
}

void Stats::Reset() {
	node_access = 0;
	for (int i = 0; i < 5; i++) {
		action_history[i] = 0;
	}
}


void Stats::TakeAction(int action) {
	action_history[action] += 1;
}

/*
double Overlap(const Rectangle& rectangle1, const Rectangle& rectangle2){
	double left = max(rectangle1.left, rectangle2.left);
	double right = min(rectangle1.right, rectangle2.right);
	double bottom = max(rectangle1.bottom, rectangle2.bottom);
	double top = min(rectangle1.top, rectangle2.top);
	if(left < right && bottom < top){
		return (top - bottom) * (right - left);
	}
	else{
		return 0;
	}
}
*/




/*
double Area(const Rectangle& rectangle){
	return (rectangle.top - rectangle.bottom) * (rectangle.right - rectangle.left);
}

double Perimeter(const Rectangle& rectangle){
	return 2 * (rectangle.top - rectangle.bottom + rectangle.right - rectangle.left);
}
*/
double SplitArea(const Rectangle& rectangle1, const Rectangle& rectangle2) {

	return rectangle1.Area() + rectangle2.Area();
}

//template<class T>
//double SplitArea(T* t1, T* t2) {
//	return t1->Area() + t2->Area();
//}


double SplitPerimeter(const Rectangle& rectangle1, const Rectangle& rectangle2) {
	return rectangle1.Perimeter() + rectangle2.Perimeter();
}

//template<class T>
//double SlitPerimeter(T* t1, T* t2) {
//	return t1->Perimeter() + t2->Perimeter();
//}

double SplitOverlap(const Rectangle& rectangle1, const Rectangle& rectangle2) {
	double left = max(rectangle1.Left(), rectangle2.Left());
	double right = min(rectangle1.Right(), rectangle2.Right());
	double bottom = max(rectangle1.Bottom(), rectangle2.Bottom());
	double top = min(rectangle1.Top(), rectangle2.Top());
	if (left < right && bottom < top) {
		return (top - bottom) * (right - left);
	}
	else {
		return 0;
	}
}

double MarginOvlpPerim(const Rectangle* rectangle1, const Rectangle* obj, const Rectangle* rectangle2){
	Rectangle r;
	r.Set(*rectangle1);
	r.Include(*obj);
	double left = max(r.Left(), rectangle2->Left());
	double right = min(r.Right(), rectangle2->Right());
	double bottom = max(r.Bottom(), rectangle2->Bottom());
	double top = min(r.Top(), rectangle2->Top());
	double perim1 = 0, perim2 = 0;
	if(left < right && bottom <top){
		perim1 = (right - left) * 2 + (top - bottom) * 2;
	}
	left = max(rectangle1->Left(), rectangle2->Left());
	right = min(rectangle1->Right(), rectangle2->Right());
	bottom = max(rectangle1->Bottom(), rectangle2->Bottom());
	top = min(rectangle1->Top(), rectangle2->Top());
	if(left < right && bottom < top){
		perim2 = (right - left) * 2 + (top - bottom) * 2;
	}
	return perim1 - perim2;
}

double MarginOvlpArea(const Rectangle* rectangle1, const Rectangle* obj, const Rectangle* rectangle2){
	Rectangle r;
	r.Set(*rectangle1);
	r.Include(*obj);
	double left = max(r.Left(), rectangle2->Left());
	double right = min(r.Right(), rectangle2->Right());
	double bottom = max(r.Bottom(), rectangle2->Bottom());
	double top = min(r.Top(), rectangle2->Top());
	double area1 = 0, area2 = 0;
	if(left < right && bottom < top){
		area1 = (right - left) * (top - bottom);
	}
	left = max(rectangle1->Left(), rectangle2->Left());
	right = min(rectangle1->Right(), rectangle2->Right());
	bottom = max(rectangle1->Bottom(), rectangle2->Bottom());
	top = min(rectangle1->Top(), rectangle2->Top());
	if(left < right && bottom < top){
		area2 = (right - left) * (top - bottom);
	}
	return area1 - area2;
}



//template<class T>
//double SplitOverlap(T* t1, T* t2) {
//	double left = max(t1->Left(), t2->Left());
//	double right = min(t1->Right(), t2->Right());
//	double bottom = max(t1->Bottom(), t2->Bottom()); 
//	double top = min(t1->Top(), t2->Top());
//	if (left < right && bottom < top) {
//		return (top - bottom) * (right - left);
//	}
//	else {
//		return 0;
//	}
//}

bool Rectangle::Contains(const Rectangle* rec) {
	if (left_ <= rec->left_ && rec->right_ <= right_ && bottom_ <= rec->bottom_ && top_ >= rec->top_) {
		return true;
	}
	else {
		return false;
	}
}


//bool IsContained(const Rectangle& rectangle1, const Rectangle& rectangle2){
//    if(rectangle1.right >= rectangle2.right && rectangle1.left <= rectangle2.left &&
//    rectangle1.top >= rectangle2.top && rectangle1.bottom <= rectangle2.bottom){
//        return true;
//    }
//    else{
//        return false;
//    }
//}

bool Rectangle::IsOverlap(Rectangle* rec) {
	double left = max(left_, rec->left_);
	double right = min(right_, rec->right_);
	double bottom = max(bottom_, rec->bottom_);
	double top = min(top_, rec->top_);
	if (left < right && bottom < top) {
		return true;
	}
	else {
		return false;
	}
}

//bool IsOverlapped(const Rectangle& rectangle1, const Rectangle& rectangle2){
//    double left = max(rectangle1.left, rectangle2.left);
//    double right = min(rectangle1.right, rectangle2.right);
//    double bottom = max(rectangle1.bottom, rectangle2.bottom);
//    double top = min(rectangle1.top, rectangle2.top);
//    if(left < right && bottom < top){
//        return true;
//    }
//    else{
//        return false;
//    }
//}



bool SortedByLeft(const Rectangle* rec1, const Rectangle* rec2) {
	if(rec1->left_ < rec2->left_)return true;
	if(rec1->left_ > rec2->left_)return false;
	if(rec1->right_ < rec2->right_)return true;
	if(rec1->right_ > rec2->right_)return false;
	if(rec1->bottom_ < rec2->bottom_)return true;
	if(rec1->bottom_ > rec2->bottom_)return false;
	if(rec1->top_ < rec2->top_)return true;
	if(rec1->top_ > rec2->top_)return false;
	return rec1->id_ < rec2->id_;
}



bool SortedByRight(const Rectangle* rec1, const Rectangle* rec2) {
	if(rec1->right_ < rec2->right_)return true;
	if(rec1->right_ > rec2->right_)return false;
	if(rec1->left_ < rec2->left_)return true;
	if(rec1->left_ > rec2->left_)return false;
	if(rec1->bottom_ < rec2->bottom_)return true;
	if(rec1->bottom_ > rec2->bottom_)return false;
	if(rec1->top_ < rec2->top_)return true;
	if(rec1->top_ > rec2->top_)return false;
	return rec1->id_ < rec2->id_;
}



bool SortedByTop(const Rectangle* rec1, const Rectangle* rec2) {
	if(rec1->top_ < rec2->top_)return true;
	if(rec1->top_ > rec2->top_)return false;
	if(rec1->bottom_ < rec2->bottom_)return true;
	if(rec1->bottom_ > rec2->bottom_)return false;
	if(rec1->left_ < rec2->left_)return true;
	if(rec1->left_ > rec2->left_)return false;
	if(rec1->right_ < rec2->right_)return true;
	if(rec1->right_ > rec2->right_)return false;
	return rec1->id_ < rec2->id_;
}



bool SortedByBottom(const Rectangle* rec1, const Rectangle* rec2) {
	if(rec1->bottom_ < rec2->bottom_)return true;
	if(rec1->bottom_ > rec2->bottom_)return false;
	if(rec1->top_ < rec2->top_)return true;
	if(rec1->top_ > rec2->top_)return false;
	if(rec1->left_ < rec2->left_)return true;
	if(rec1->left_ > rec2->left_)return false;
	if(rec1->right_ < rec2->right_)return true;
	if(rec1->right_ > rec2->right_)return false;
	return rec1->id_ < rec2->id_;
}

bool CompareByArea(const Rectangle* rec1, const Rectangle* rec2){
	return rec1->Area() < rec2->Area();
}


TreeNode::TreeNode() {
	entry_num = 0;
	is_overflow = false;
	children = vector<int>(TreeNode::maximum_entry + 1);
	father = -1;
}

TreeNode::TreeNode(TreeNode* node) {
	entry_num = node->entry_num;
	is_overflow = node->is_overflow;
	is_leaf = node->is_leaf;
	children = vector<int>(TreeNode::maximum_entry + 1);
	for (int i = 0; i < entry_num; i++) {
		children[i] = node->children[i];
	}
	origin_center[0] = node->origin_center[0];
	origin_center[1] = node->origin_center[1];
	father = node->father;
	left_ = node->left_;
	right_ = node->right_;
	bottom_ = node->bottom_;
	top_ = node->top_;
	id_ = node->id_;
}

bool TreeNode::AddChildren(int node_id) {
	children[entry_num] = node_id;
	entry_num += 1;
	if (entry_num > TreeNode::maximum_entry) {
		is_overflow = true;
	}
	return is_overflow;
}

bool TreeNode::AddChildren(TreeNode *node) {
	children[entry_num] = node->id_;
	entry_num += 1;
	if (entry_num > TreeNode::maximum_entry) {
		is_overflow = true;
	}
	return is_overflow;
}

bool TreeNode::CopyChildren(const vector<int>& nodes) {
	if (nodes.size() >= TreeNode::maximum_entry)return false;
	int idx = 0;
	for (int idx = 0; idx < nodes.size(); idx++) {
		children[idx] = nodes[idx];
	}
	entry_num = nodes.size();
	is_overflow = false;
	return true;
}

bool TreeNode::CopyChildren(const vector<TreeNode *> &nodes) {
	if (nodes.size() >= TreeNode::maximum_entry)return false;
	for (int idx = 0; idx < nodes.size(); idx++) {
		children[idx] = nodes[idx]->id_;
	}
	entry_num = nodes.size();
	is_overflow = false;
	return true;
}

Rectangle* RTree::InsertRectangle(double left, double right, double bottom, double top) {
	Rectangle* rectangle = new Rectangle(left, right, bottom, top);
	rectangle->id_ = objects_.size();
	objects_.push_back(rectangle);
	return rectangle;
}

TreeNode* RTree::InsertStepByStep(const Rectangle *rectangle, TreeNode *tree_node) {
	return InsertStepByStep(rectangle, tree_node, insert_strategy_);
}

void RTree::PrintEntryNum() {
	TreeNode* iter = nullptr;
	list<TreeNode*> queue;
	queue.push_back(tree_nodes_[root_]);
	while (!queue.empty()) {
		iter = queue.front();
		queue.pop_front();
		cout << iter->entry_num << " ";
		if (!iter->is_leaf) {
			for (int i = 0; i < iter->entry_num; i++) {
				int child_id = iter->children[i];
				queue.push_back(tree_nodes_[child_id]);
			}
		}
	}
}

void RTree::Print() {
	TreeNode* iter = nullptr;
	list<TreeNode*> queue;
	queue.push_back(tree_nodes_[root_]);
	while (!queue.empty()) {
		iter = queue.front();
		queue.pop_front();
		cout << "node: [" << iter->left_ << ", " << iter->right_ << ", " << iter->bottom_ << ", " << iter->top_ << "]";
		cout << " " << iter->entry_num << " children, is_overflow: "<<iter->is_overflow;
		for (int i = 0; i < iter->entry_num; i++) {
			if (iter->is_leaf) {
				int child_idx = iter->children[i];
				Rectangle* r_iter = objects_[child_idx];
				cout << " object [" << r_iter->left_ << ", " << r_iter->right_ << ", " << r_iter->bottom_ << ", " << r_iter->top_ << "]";
			}
			else {
				int child_idx = iter->children[i];
				TreeNode* t_iter = tree_nodes_[child_idx];
				cout << " node[" << t_iter->left_ << ", " << t_iter->right_ << ", " << t_iter->bottom_ << ", " << t_iter->top_ << "]";
				queue.push_back(t_iter);
			}
		}
		cout << endl;
	}
	cout << "#######################"<<endl;
}

TreeNode* RTree::SplitStepByStep(TreeNode *tree_node) {
	return SplitStepByStep(tree_node, split_strategy_);
}


template<class T>
Rectangle MergeRange(const vector<T*>& entries, const int start_idx, const int end_idx) {
	Rectangle rectangle(*entries[start_idx]);
	for (int idx = start_idx + 1; idx < end_idx; ++idx) {
		rectangle.Include(*entries[idx]);
	}
	return rectangle;
}

template<class T>
pair<double, bool> SplitPerimSum(const vector<T*>& entries){
	Rectangle prefix = MergeRange<T>(entries, 0, TreeNode::minimum_entry - 1);
	Rectangle suffix = MergeRange<T>(entries, TreeNode::maximum_entry - TreeNode::minimum_entry+1, entries.size());
	double perim_sum = 0;
	Rectangle rec_remaining;
	bool is_overlap = true;
	for(int idx = TreeNode::minimum_entry - 1; idx < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; idx++){
		prefix.Include(*entries[idx]);
		rec_remaining.Set(suffix);
		for(int i = idx + 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; i++){
			rec_remaining.Include(*entries[i]);
		}
		if(!prefix.IsOverlap(&rec_remaining)){
			is_overlap = false;
		}
		perim_sum += prefix.Perimeter() + rec_remaining.Perimeter();
	}
	pair<double, bool> result;
	result.first = perim_sum;
	result.second = is_overlap;
	return result;
}

//Rectangle RTree::MergeRange(const vector<Rectangle*>& entries, const int start_idx, const int end_idx) {
//	Rectangle rectangle(*entries[0]);
//	for (int idx = start_idx + 1; idx < end_idx; ++idx) {
//		rectangle.Include(*entries[idx]);
//	}
//	return rectangle;
//}
//Rectangle RTree::MergeRange(const vector<TreeNode*> &entry_list, const int start_idx, const int end_idx) {
//    Rectangle rectangle(*entry_list[start_idx]);
//    for(int idx = start_idx+1; idx < end_idx; ++idx){
//        rectangle.Include(*entry_list[idx]);
//    }
//    return rectangle;
//}

template<class T>
int FindMinimumSplit(const vector<T*>& entries, double(*score_func1)(const Rectangle &, const Rectangle &),
	double(*score_func2)(const Rectangle &, const Rectangle &), double& min_value1, double& min_value2, Rectangle& rec1, Rectangle& rec2) {
	Rectangle rec_prefix = MergeRange<T>(entries, 0, TreeNode::minimum_entry - 1);
	Rectangle rec_suffix = MergeRange<T>(entries, TreeNode::maximum_entry - TreeNode::minimum_entry + 1, entries.size());
	int optimal_split = -1;
	for (int idx = TreeNode::minimum_entry - 1; idx < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; idx++) {
		rec_prefix.Include(*entries[idx]);
		Rectangle rec_remaining(rec_suffix);
		for (int i = idx + 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; i++) {
			rec_remaining.Include(*entries[i]);
		}
		double value1 = score_func1(rec_prefix, rec_remaining);
		double value2 = score_func2(rec_prefix, rec_remaining);
		if (value1 < min_value1 || (value1 == min_value1 && value2 < min_value2)) {
			min_value1 = value1;
			min_value2 = value2;
			rec1.Set(rec_prefix);
			rec2.Set(rec_remaining);
			optimal_split = idx;
		}
	}
	return optimal_split;
}

template<class T>
int FindMinimumSplitRR(const vector<T*>& entries, double ys, double y1, double miu, double delta, double perim_max, Rectangle& rec1, Rectangle& rec2){
	Rectangle rec_prefix = MergeRange<T>(entries, 0, TreeNode::minimum_entry - 1);
	Rectangle rec_suffix = MergeRange<T>(entries, TreeNode::maximum_entry - TreeNode::minimum_entry +1, entries.size());
	int optimal_split = -1;
	Rectangle rec_remaining;
	double min_score = DBL_MAX;
	for(int idx = TreeNode::minimum_entry - 1; idx < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; idx++){
		rec_prefix.Include(*entries[idx]);
		rec_remaining.Set(rec_suffix);
		for(int i = idx + 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; i++){
			rec_remaining.Include(*entries[i]);
		}
		double wg = 0;
		if(rec_prefix.IsOverlap(&rec_remaining)){
			wg = SplitOverlap(rec_prefix, rec_remaining);
		}
		else{
			wg = rec_prefix.Perimeter() + rec_remaining.Perimeter() - perim_max;
		}
		
		double x = (2.0 * idx) / (TreeNode::maximum_entry + 1) - 1;
		double wf = ys * (exp(-1.0 * (x - miu) * (x-miu) / delta /delta) - y1);
		double score = 0;
		if(rec_prefix.IsOverlap(&rec_remaining)){
			score = wg / wf;
		}
		else{
			score = wg * wf;
		}
		if(score < min_score){
			min_score = score;
			optimal_split = idx;
			rec1.Set(rec_prefix);
			rec2.Set(rec_remaining);
		}
	}
	return optimal_split;
}

//void RTree::FindMinimumSplit(const vector<Rectangle*> &entry_list, double(*score_func1)(const Rectangle &, const Rectangle &), double(*score_func2)(const Rectangle &, const Rectangle &), double &min_value1, double &min_value2, vector<Rectangle> &child1, vector<Rectangle> &child2) {
//	Rectangle rec_prefix = MergeRange(entry_list, 0, TreeNode::minimum_entry);
//	Rectangle rec_suffix = MergeRange(entry_list, TreeNode::maximum_entry + 1, entry_list.size());
//	int optimal_split = -1;
//	for (int idx = TreeNode::minimum_entry; idx <= TreeNode::maximum_entry; idx++) {
//		rec_prefix.Include(entry_list[idx]);
//		Rectangle rec_remaining(rec_suffix);
//		for (int i = idx + 1; i < TreeNode::maximum_entry + 1; i++) {
//			rec_remaining.Include(entry_list[i]);
//		}
//		double value1 = score_func1(rec_prefix, rec_remaining);
//		double value2 = score_func2(rec_prefix, rec_remaining);
//		if (value1 < min_value1 || (value1 == min_value1 && value2 < min_value2)) {
//			min_value1 = value1;
//			min_value2 = value2;
//			optimal_split = idx;
//			//child1.assign(entry_list.begin, entry_list.begin(0))
//		}
//	}
//	if (optimal_split > 0) {
//		child1.assign(entry_list.begin(), entry_list.begin() + optimal_split + 1);
//		child2.assign(entry_list.begin() + optimal_split + 1, entry_list.end());
//	}
//}

//void RTree::FindMinimumSplit(const vector<TreeNode *> &entry_list, double (*score_func1)(const Rectangle &, const Rectangle &), double (*score_func2)(const Rectangle &, const Rectangle &), double &min_value1, double &min_value2, vector<TreeNode *> &child1, vector<TreeNode *> &child2) {
//    Rectangle rec_prefix = MergeRange(entry_list, 0, TreeNode::minimum_entry);
//    Rectangle rec_suffix = MergeRange(entry_list, TreeNode::maximum_entry+1, entry_list.size());
//	int optimal_split = -1;
//    for(int idx = TreeNode::minimum_entry; idx <= TreeNode::maximum_entry; idx++){
//        rec_prefix.Include(entry_list[idx]->bounding_box);
//        Rectangle rec_remaining(rec_suffix);
//        for(int i = idx+1; i<TreeNode::maximum_entry+1; i++){
//            rec_remaining.Include(entry_list[i]->bounding_box);
//        }
//        double value1 = score_func1(rec_prefix, rec_remaining);
//        double value2 = score_func2(rec_prefix, rec_remaining);
//        if(value1 < min_value1 || (value1 == min_value1 && value2 < min_value2)){
//            min_value1 = value1;
//            min_value2 = value2;
//			optimal_split = idx;
//            //child1.assign(entry_list.begin(), entry_list.begin() + idx + 1);
//            //child2.assign(entry_list.begin() + idx + 1, entry_list.end());
//        }
//    }
//	if (optimal_split > 0) {
//		child1.assign(entry_list.begin(), entry_list.begin() + optimal_split + 1);
//		child2.assign(entry_list.begin() + optimal_split + 1, entry_list.end());
//	}
//}


RTree::RTree() {
	TreeNode* root = CreateNode();
	height_ = 1;
	root->is_leaf = true;
	root_ = 0;

	RR_s = 0.5;
	RR_y1 = exp(-1 / RR_s / RR_s);
	RR_ys = 1.0 / (1.0 - RR_y1);
}

TreeNode* RTree::Root() {
	return tree_nodes_[root_];
}

void RTree::Recover(RTree* rtree) {
	for (auto it = history.begin(); it != history.end(); ++it) {
		int node_id = *it;
		tree_nodes_[node_id]->Set(*(rtree->tree_nodes_[node_id]));
		tree_nodes_[node_id]->entry_num = rtree->tree_nodes_[node_id]->entry_num;
		tree_nodes_[node_id]->origin_center[0] = rtree->tree_nodes_[node_id]->origin_center[0];
		tree_nodes_[node_id]->origin_center[1] = rtree->tree_nodes_[node_id]->origin_center[1];
		for (int i = 0; i < tree_nodes_[node_id]->entry_num; i++) {
			tree_nodes_[node_id]->children[i] = rtree->tree_nodes_[node_id]->children[i];
			int child = tree_nodes_[node_id]->children[i];
			tree_nodes_[child]->father = node_id;
		}
	}
}

TreeNode* SplitWithSortedLoc(RTree* tree, TreeNode* tree_node, int loc){
	TreeNode* node = tree->SplitInSortedLoc(tree_node, loc);
	return node;
}

TreeNode* SplitWithCandidateAction(RTree* tree, TreeNode* tree_node, int loc){
	TreeNode* next_node = tree->SplitWithCandidateAction(tree_node, loc);
	return next_node;
}

TreeNode* SplitWithLoc(RTree* tree, TreeNode* tree_node, int loc) {
	TreeNode* node = tree->SplitInLoc(tree_node, loc);
	return node;
}

TreeNode* InsertWithLoc(RTree* tree, TreeNode* tree_node, int loc, Rectangle* rec){
	//cout<<"insert with loc invoked"<<endl;
	if(tree_node->entry_num > 0){
		loc = loc % tree_node->entry_num;
	}
	TreeNode* next_node = tree->InsertInLoc(tree_node, loc, rec);
	return next_node;
}

TreeNode* InsertWithSortedLoc(RTree* tree, TreeNode* tree_node, int sorted_loc, Rectangle* rec){
	if(tree->tmp_sorted_children.empty()){
		cout<<"Children haven't been sorted yet."<<endl;
		exit(0);
	}
	if(tree->tmp_sorted_children.size() > 0){
		sorted_loc = sorted_loc % tree->tmp_sorted_children.size();
	}
	TreeNode* next_node = tree->InsertInSortedLoc(tree_node, sorted_loc, rec);
	tree->tmp_sorted_children.clear();
	return next_node;
}

TreeNode* RTree::InsertInSortedLoc(TreeNode *tree_node, int sorted_loc, Rectangle *rec){
	TreeNode* next_node = nullptr;
	if(tree_node->is_leaf){
		if(tree_node->entry_num == 0){
			tree_node->Set(*rec);
			tree_node->origin_center[0] = 0.5 * (rec->Right() + rec->Left());
			tree_node->origin_center[1] = 0.5 * (rec->Bottom() + rec->Top());
		}
		else{
			tree_node->Include(*rec);
		}
		tree_node->AddChildren(rec->id_);
	}
	else{
		tree_node->Include(*rec);
		next_node = tmp_sorted_children[sorted_loc];
	}
	return next_node;
}

TreeNode* RTree::InsertInLoc(TreeNode *tree_node, int loc, Rectangle *rec){
	TreeNode* next_node = nullptr;
	if(tree_node->is_leaf){
		//cout<<"is leaf"<<endl;
		if(tree_node->entry_num == 0){
			tree_node->Set(*rec);
			tree_node->origin_center[0] = 0.5 * (rec->Left() + rec->Right());
			tree_node->origin_center[1] = 0.5 * (rec->Bottom() + rec->Top());
		}
		else{
			tree_node->Include(*rec);
		}
		tree_node->AddChildren(rec->id_);
	}
	else{
		tree_node->Include(*rec);
		int next_node_id = tree_node->children[loc];
		next_node = tree_nodes_[next_node_id];
	}
	return next_node;
}

TreeNode* RTree::SplitInSortedLoc(TreeNode* tree_node, int sorted_loc){
	TreeNode* next_node = nullptr;
	vector<int> new_child1;
	vector<int> new_child2;
	Rectangle bounding_box1;
	Rectangle bounding_box2;
	bool is_horizontal = sorted_split_loc[sorted_loc].second.first;
	int split_loc = sorted_split_loc[sorted_loc].second.second;
	//cout<<"split in loc "<<split_loc<<" is_horizontal: "<<is_horizontal<<endl;
	if(tree_node->is_leaf){
		vector<Rectangle*> recs(tree_node->entry_num);
		for(int i=0; i<tree_node->entry_num; i++){
			int obj_id = tree_node->children[i];
			recs[i] = objects_[obj_id];
		}
		if(is_horizontal){
			sort(recs.begin(), recs.end(), SortedByLeft);
		}
		else{
			sort(recs.begin(), recs.end(), SortedByBottom);
		}
		new_child1.resize(split_loc + 1);
		new_child2.resize(TreeNode::maximum_entry - split_loc);
		for(int i=0; i<split_loc + 1; i++){
			new_child1[i] = recs[i]->id_;
		}
		for(int i=split_loc + 1; i < recs.size(); i++){
			new_child2[i-split_loc-1] = recs[i]->id_;
		}
		bounding_box1.Set(*recs[0]);
		for(int i=1; i<split_loc+1; i++){
			bounding_box1.Include(*recs[i]);
		}
		bounding_box2.Set(*recs[split_loc+1]);
		for(int i=split_loc+2; i<recs.size(); i++){
			bounding_box2.Include(*recs[i]);
		}
	}
	else{
		vector<TreeNode*> nodes(tree_node->entry_num);
		for(int i=0; i<tree_node->entry_num; i++){
			int node_id = tree_node->children[i];
			nodes[i] = tree_nodes_[node_id];
		}
		if(is_horizontal){
			sort(nodes.begin(), nodes.end(), SortedByLeft);
		}
		else{
			sort(nodes.begin(), nodes.end(), SortedByBottom);
		}
		new_child1.resize(split_loc+1);
		new_child2.resize(TreeNode::maximum_entry - split_loc);
		for(int i=0; i<split_loc + 1; i++){
			new_child1[i] = nodes[i]->id_;
		}
		for(int i = split_loc+1; i<nodes.size(); i++){
			new_child2[i - split_loc - 1] = nodes[i]->id_;
		}
		bounding_box1.Set(*nodes[0]);
		for(int i=1; i<split_loc + 1; i++){
			bounding_box1.Include(*nodes[i]);
		}
		bounding_box2.Set(*nodes[split_loc+1]);
		for(int i=split_loc+2; i<nodes.size(); i++){
			bounding_box2.Include(*nodes[i]);
		}
	}
	TreeNode* sibling = CreateNode();
	sibling->is_leaf = tree_node->is_leaf;
	sibling->CopyChildren(new_child2);
	if (!sibling->is_leaf) {
		for (int i = 0; i < new_child2.size(); i++) {
			tree_nodes_[new_child2[i]]->father = sibling->id_;
		}
	}
	sibling->Set(bounding_box2);
	sibling->origin_center[0] = 0.5 * (bounding_box2.Left() + bounding_box2.Right());
	sibling->origin_center[1] = 0.5 * (bounding_box2.Bottom() + bounding_box2.Top());
	
	tree_node->CopyChildren(new_child1);
	if (!tree_node->is_leaf) {
		for (int i = 0; i < new_child1.size(); i++) {
			tree_nodes_[new_child1[i]]->father = tree_node->id_;
		}
	}
	tree_node->Set(bounding_box1);
	tree_node->origin_center[0] = 0.5 * (bounding_box1.Left() + bounding_box1.Right());
	tree_node->origin_center[1] = 0.5 * (bounding_box1.Bottom() + bounding_box1.Top());
	if (tree_node->father >= 0) {
		tree_nodes_[tree_node->father]->AddChildren(sibling);
		tree_nodes_[tree_node->father]->Include(bounding_box2);
		sibling->father = tree_node->father;
	}
	else {
		TreeNode* new_root = CreateNode();
		new_root->is_leaf = false;
		new_root->AddChildren(tree_node);
		new_root->AddChildren(sibling);
		new_root->Set(bounding_box1);
		new_root->Include(bounding_box2);
		new_root->origin_center[0] = 0.5 * (new_root->Left() + new_root->Right());
		new_root->origin_center[1] = 0.5 * (new_root->Bottom() + new_root->Top());
		root_ = new_root->id_;
		tree_node->father = new_root->id_;
		sibling->father = new_root->id_;
		height_ += 1;
	}
	next_node = tree_nodes_[tree_node->father];
	return next_node;
}

TreeNode* RTree::SplitInLoc(TreeNode* tree_node, int loc, int dim){
	TreeNode* next_node = nullptr;
	vector<int> new_child1;
	vector<int> new_child2;
	Rectangle bounding_box1;
	Rectangle bounding_box2;
	if(tree_node->is_leaf){
		vector<Rectangle*> recs(tree_node->entry_num);
		//cout<<"in the original node: ";
		for (int i = 0; i < tree_node->entry_num; i++) {
			int obj_id = tree_node->children[i];
			//cout<<obj_id<<" ";
			recs[i] = objects_[obj_id];
		}
		//cout<<endl;
		switch(dim){
			case 0:{
				sort(recs.begin(), recs.end(), SortedByLeft);
				break;
			}
			case 1:{
				sort(recs.begin(), recs.end(), SortedByBottom);
				break;
			}
		}
		//cout<<"after sorting: ";
		//for(int i=0; i<recs.size(); i++){
		//	cout<<recs[i]->id_<<" ";
		//}
		//cout<<endl;
		new_child1.resize(loc + 1);
		new_child2.resize(TreeNode::maximum_entry - loc);
		for (int i = 0; i < loc + 1; i++) {
			new_child1[i] = recs[i]->id_;
		}
		for (int i = loc + 1; i < recs.size(); i++){
			new_child2[i - loc - 1] = recs[i]->id_;
		}
		//cout<<"child 1: "<<endl;
		//for(int i=0; i<new_child1.size(); i++){
		//	cout<<new_child1[i]<<" ";
		//}
		//cout<<endl;;
		//cout<<"child 2: "<<endl;
		//for(int i=0; i<new_child2.size(); i++){
		//	cout<<new_child2[i]<<" ";
		//}
		//cout<<endl;
		//getchar();
		bounding_box1.Set(*recs[0]);
		for (int i = 1; i < loc + 1; i++) {
			bounding_box1.Include(*recs[i]);
		}
		bounding_box2.Set(*recs[loc + 1]);
		for (int i = loc + 2; i < recs.size(); i++) {
			bounding_box2.Include(*recs[i]);
		}
	}
	else{
		vector<TreeNode*> nodes(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			int node_id = tree_node->children[i];
			nodes[i] = tree_nodes_[node_id];
		}
		switch(dim){
			case 0:{
				sort(nodes.begin(), nodes.end(), SortedByLeft);
				break;
			}
			case 1:{
				sort(nodes.begin(), nodes.end(), SortedByBottom);
				break;
			}
		}
		new_child1.resize(loc + 1);
		new_child2.resize(TreeNode::maximum_entry - loc);
		for (int i = 0; i < loc + 1; i++) {
			new_child1[i] = nodes[i]->id_;
		}
		for (int i = loc + 1; i < nodes.size(); i++){
			new_child2[i - loc - 1] = nodes[i]->id_;
		}
		bounding_box1.Set(*nodes[0]);
		for (int i = 1; i < loc + 1; i++) {
			bounding_box1.Include(*nodes[i]);
		}
		bounding_box2.Set(*nodes[loc + 1]);
		for (int i = loc + 2; i < nodes.size(); i++) {
			bounding_box2.Include(*nodes[i]);
		}
	}
	TreeNode* sibling = CreateNode();
	sibling->is_leaf = tree_node->is_leaf;
	sibling->CopyChildren(new_child2);
	if (!sibling->is_leaf) {
		for (int i = 0; i < new_child2.size(); i++) {
			tree_nodes_[new_child2[i]]->father = sibling->id_;
		}
	}
	sibling->Set(bounding_box2);
	sibling->origin_center[0] = 0.5 * (sibling->Left() + sibling->Right());
	sibling->origin_center[1] = 0.5 * (sibling->Top() + sibling->Bottom());
	tree_node->CopyChildren(new_child1);
	if (!tree_node->is_leaf) {
		for (int i = 0; i < new_child1.size(); i++) {
			tree_nodes_[new_child1[i]]->father = tree_node->id_;
		}
	}
	tree_node->Set(bounding_box1);
	tree_node->origin_center[0] = 0.5 * (tree_node->Left() + tree_node->Right());
	tree_node->origin_center[1] = 0.5 * (tree_node->Bottom() + tree_node->Top());
	if (tree_node->father >= 0) {
		tree_nodes_[tree_node->father]->AddChildren(sibling);
		tree_nodes_[tree_node->father]->Include(bounding_box2);
		sibling->father = tree_node->father;
	}
	else {
		TreeNode* new_root = CreateNode();
		new_root->is_leaf = false;
		new_root->AddChildren(tree_node);
		new_root->AddChildren(sibling);
		new_root->Set(bounding_box1);
		new_root->Include(bounding_box2);
		new_root->origin_center[0] = 0.5 * (new_root->Left() + new_root->Right());
		new_root->origin_center[1] = 0.5 * (new_root->Bottom() + new_root->Top());
		root_ = new_root->id_;
		tree_node->father = new_root->id_;
		sibling->father = new_root->id_;
		height_ += 1;
	}
	next_node = tree_nodes_[tree_node->father];
	return next_node;
}

TreeNode* RTree::SplitWithCandidateAction(TreeNode* tree_node, int loc){
	TreeNode* next_node = nullptr;
	int dim = split_locations[candidate_split_action[loc]].dimension;
	int split_loc = split_locations[candidate_split_action[loc]].location;
	next_node = SplitInLoc(tree_node, split_loc, dim);
	return next_node;
}

TreeNode* RTree::SplitInLoc(TreeNode* tree_node, int loc) {
	TreeNode* next_node = nullptr;
	int size_per_dim = TreeNode::maximum_entry - 2 * TreeNode::minimum_entry + 2;
	int dimension = loc / size_per_dim;
	int idx = loc % size_per_dim + TreeNode::minimum_entry - 1;
	vector<int> new_child1;
	vector<int> new_child2;
	Rectangle bounding_box1;
	Rectangle bounding_box2;
	if (tree_node->is_leaf) {
		vector<Rectangle*> recs(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			int obj_id = tree_node->children[i];
			recs[i] = objects_[obj_id];
		}
		switch (dimension)
		{
		case 0: {
			sort(recs.begin(), recs.end(), SortedByLeft);
			break;
		}
		case 1: {
			sort(recs.begin(), recs.end(), SortedByRight);
			break;
		}
		case 2: {
			sort(recs.begin(), recs.end(), SortedByBottom);
			break;
		}
		case 3: {
			sort(recs.begin(), recs.end(), SortedByTop);
			break;
		}
		default:
			break;
		}
		new_child1.resize(idx + 1);
		new_child2.resize(TreeNode::maximum_entry - idx);
		for (int i = 0; i < idx + 1; i++) {
			new_child1[i] = recs[i]->id_;
		}
		for (int i = idx + 1; i < recs.size(); i++){
			new_child2[i - idx - 1] = recs[i]->id_;
		}
		bounding_box1.Set(*recs[0]);
		for (int i = 1; i < idx + 1; i++) {
			bounding_box1.Include(*recs[i]);
		}
		bounding_box2.Set(*recs[idx + 1]);
		for (int i = idx + 2; i < recs.size(); i++) {
			bounding_box2.Include(*recs[i]);
		}
	}
	else {
		vector<TreeNode*> nodes(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			int node_id = tree_node->children[i];
			nodes[i] = tree_nodes_[node_id];
		}
		switch (dimension)
		{
		case 0: {
			sort(nodes.begin(), nodes.end(), SortedByLeft);
			break;
		}
		case 1: {
			sort(nodes.begin(), nodes.end(), SortedByRight);
			break;
		}
		case 2: {
			sort(nodes.begin(), nodes.end(), SortedByBottom);
			break;
		}
		case 3: {
			sort(nodes.begin(), nodes.end(), SortedByTop);
			break;
		}
		default:
			break;
		}
		new_child1.resize(idx + 1);
		new_child2.resize(TreeNode::maximum_entry - idx);
		for (int i = 0; i < idx + 1; i++) {
			new_child1[i] = nodes[i]->id_;
		}
		for (int i = idx+1; i < nodes.size(); i++) {
			new_child2[i - idx - 1] = nodes[i]->id_;
		}
		bounding_box1.Set(*nodes[0]);
		for (int i = 1; i < idx + 1; i++) {
			bounding_box1.Include(*nodes[i]);
		}
		bounding_box2.Set(*nodes[idx + 1]);
		for (int i = idx + 2; i < nodes.size(); i++) {
			bounding_box2.Include(*nodes[i]);
		}
	}
	TreeNode* sibling = CreateNode();
	sibling->is_leaf = tree_node->is_leaf;
	sibling->CopyChildren(new_child2);
	if (!sibling->is_leaf) {
		for (int i = 0; i < new_child2.size(); i++) {
			tree_nodes_[new_child2[i]]->father = sibling->id_;
		}
	}
	sibling->Set(bounding_box2);
	sibling->origin_center[0] = 0.5 * (bounding_box2.Left() + bounding_box2.Right());
	sibling->origin_center[1] = 0.5 * (bounding_box2.Top() + bounding_box2.Bottom());
	tree_node->CopyChildren(new_child1);
	if (!tree_node->is_leaf) {
		for (int i = 0; i < new_child1.size(); i++) {
			tree_nodes_[new_child1[i]]->father = tree_node->id_;
		}
	}
	tree_node->Set(bounding_box1);
	tree_node->origin_center[0] = 0.5 * (bounding_box1.Left() + bounding_box1.Right());
	tree_node->origin_center[1] = 0.5 * (bounding_box1.Bottom() + bounding_box1.Top());
	if (tree_node->father >= 0) {
		tree_nodes_[tree_node->father]->AddChildren(sibling);
		tree_nodes_[tree_node->father]->Include(bounding_box2);
		sibling->father = tree_node->father;
	}
	else {
		TreeNode* new_root = CreateNode();
		new_root->is_leaf = false;
		new_root->AddChildren(tree_node);
		new_root->AddChildren(sibling);
		new_root->Set(bounding_box1);
		new_root->Include(bounding_box2);
		new_root->origin_center[0] = 0.5 * (new_root->Left() + new_root->Right());
		new_root->origin_center[1] = 0.5 * (new_root->Bottom() + new_root->Top());
		root_ = new_root->id_;
		tree_node->father = new_root->id_;
		sibling->father = new_root->id_;
		height_ += 1;
	}
	next_node = tree_nodes_[tree_node->father];
	return next_node;
}

double RR_wf(int i, double y1, double ys, double miu, double delta) {
	double xi = 2.0 * i / (TreeNode::maximum_entry + 1) - 1;
	double wf = ys * (exp(0 - (xi - miu) * (xi - miu) / delta / delta) - y1);
	return wf;
}

TreeNode* RTree::RRSplit(TreeNode* tree_node) {
	TreeNode* next_node = nullptr;
	if (tree_node->is_overflow) {
		double new_center[2] = { 0.5 * (tree_node->Right() + tree_node->Left()), 0.5 * (tree_node->Top() + tree_node->Bottom()) };
		double length[2] = { tree_node->Right() - tree_node->Left(), tree_node->Top() - tree_node->Bottom() };

		Rectangle bounding_box1;
		Rectangle bounding_box2;
		vector<int> new_child1;
		vector<int> new_child2;
		double perim_max = 2 * (length[0] + length[1]) - min(length[0], length[1]);
		double y1 = exp(-1.0 / TreeNode::RR_s / TreeNode::RR_s);
		double ys = 1.0 / (1.0 - y1);
		double min_w = DBL_MAX;
		int split_dim, split_loc;
		if (tree_node->is_leaf) {
			vector<Rectangle*> children(tree_node->entry_num);
			for (int i = 0; i < tree_node->entry_num; i++) {
				children[i] = objects_[tree_node->children[i]];
			}
			vector<pair<Rectangle, Rectangle> > splits[2];
			splits[0].resize(TreeNode::maximum_entry + 2 - 2 * TreeNode::minimum_entry);
			splits[1].resize(TreeNode::maximum_entry + 2 - 2 * TreeNode::minimum_entry);
			for (int dim = 0; dim < 2; dim++) {
				if (dim == 0) {
					sort(children.begin(), children.end(), SortedByLeft);
				}
				else {
					sort(children.begin(), children.end(), SortedByBottom);
				}
				Rectangle prefix = MergeRange<Rectangle>(children, 0, TreeNode::minimum_entry - 1);
				Rectangle suffix = MergeRange<Rectangle>(children, TreeNode::maximum_entry - TreeNode::minimum_entry + 1, children.size());
				for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; i++) {
					int loc = i - TreeNode::minimum_entry + 1;
					prefix.Include(*children[i]);
					splits[dim][loc].first.Set(prefix);
					splits[dim][loc].second.Set(suffix);
					for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; j++) {
						splits[dim][loc].second.Include(*children[j]);
					}
				}
			}
			double perim_sum[2] = { 0.0, 0.0 };
			for (int dim = 0; dim < 2; dim++) {
				for (int i = 0; i < splits[dim].size(); i++) {
					perim_sum[dim] += splits[dim][i].first.Perimeter() + splits[dim][i].second.Perimeter();
				}
			}
			split_dim = perim_sum[0] < perim_sum[1] ? 0 : 1;
			double asym = 2 * (new_center[split_dim] - tree_node->origin_center[split_dim]) / length[split_dim];
			double miu = (1 - 2.0 * TreeNode::minimum_entry / (TreeNode::maximum_entry + 1)) * asym;
			double delta = TreeNode::RR_s * (1 + abs(miu));
			Rectangle ovlp;
			for (int i = 0; i < splits[split_dim].size(); i++) {
				if (splits[split_dim][i].first.IsOverlap(&splits[split_dim][i].second)) {
					ovlp = Overlap(splits[split_dim][i].first, splits[split_dim][i].second);
					double wg = ovlp.Area();
					double wf = RR_wf(i + TreeNode::minimum_entry - 1, y1, ys, miu, delta);
					double w = wg / wf;
					if (w < min_w) {
						min_w = w;
						split_loc = i + TreeNode::minimum_entry - 1;
						bounding_box1.Set(splits[split_dim][i].first);
						bounding_box2.Set(splits[split_dim][i].second);
					}
				}
				else {
					double wg = splits[split_dim][i].first.Perimeter() + splits[split_dim][i].second.Perimeter() - perim_max;
					double wf = RR_wf(i + TreeNode::minimum_entry - 1, y1, ys, miu, delta);
					double w = wg * wf;
					if (w < min_w) {
						min_w = w;
						split_loc = i + TreeNode::minimum_entry - 1;
						bounding_box1.Set(splits[split_dim][i].first);
						bounding_box2.Set(splits[split_dim][i].second);
					}
				}
			}
			if (split_dim == 0) {
				sort(children.begin(), children.end(), SortedByLeft);
			}
			else {
				sort(children.begin(), children.end(), SortedByBottom);
			}
			new_child1.resize(split_loc + 1);
			new_child2.resize(tree_node->entry_num - split_loc - 1);
			for (int i = 0; i <= split_loc; i++) {
				new_child1[i] = children[i]->id_;
			}
			for (int i = split_loc + 1; i < children.size(); i++) {
				new_child2[i - split_loc - 1] = children[i]->id_;
			}
		}
		else {
			vector<TreeNode*> children(tree_node->entry_num);
			for (int i = 0; i < tree_node->entry_num; i++) {
				children[i] = tree_nodes_[tree_node->children[i]];
			}

			vector<pair<Rectangle, Rectangle> > splits(TreeNode::maximum_entry + 2 - 2 * TreeNode::minimum_entry);
			
			for (int dim = 0; dim < 2; dim++) {
				//processing each dimension in turn.
				double asym = 2 * (new_center[dim] - tree_node->origin_center[dim]) / length[dim];
				double miu = (1 - 2.0 * TreeNode::minimum_entry / (TreeNode::maximum_entry + 1)) * asym;
				double delta = TreeNode::RR_s * (1 + abs(miu));
				if (dim == 0) {
					sort(children.begin(), children.end(), SortedByLeft);
				}
				else {
					sort(children.begin(), children.end(), SortedByBottom);
				}
				Rectangle prefix = MergeRange<TreeNode>(children, 0, TreeNode::minimum_entry - 1);
				Rectangle suffix = MergeRange<TreeNode>(children, TreeNode::maximum_entry - TreeNode::minimum_entry + 1, children.size());
				for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; i++) {
					int loc = i - TreeNode::minimum_entry + 1;
					prefix.Include(*children[i]);
					splits[loc].first.Set(prefix);
					splits[loc].second.Set(suffix);
					for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; j++) {
						splits[loc].second.Include(*children[j]);
					}
				}
				Rectangle ovlp;
				for (int i = 0; i < splits.size(); i++) {
					if (splits[i].first.IsOverlap(&splits[i].second)) {
						ovlp = Overlap(splits[i].first, splits[i].second);
						double wg = ovlp.Area();
						double wf = RR_wf(i + TreeNode::minimum_entry - 1, y1, ys, miu, delta);
						double w = wg / wf;
						if (w < min_w) {
							min_w = w;
							split_dim = dim;
							split_loc = i + TreeNode::minimum_entry - 1;
							bounding_box1.Set(splits[i].first);
							bounding_box2.Set(splits[i].second);
						}
					}
					else {
						double wg = splits[i].first.Perimeter() + splits[i].second.Perimeter() - perim_max;
						double wf = RR_wf(i + TreeNode::minimum_entry - 1, y1, ys, miu, delta);
						double w = wg * wf;
						if (w < min_w) {
							min_w = w;
							split_dim = dim;
							split_loc = i + TreeNode::minimum_entry - 1;
							bounding_box1.Set(splits[i].first);
							bounding_box2.Set(splits[i].second);
						}
					}
				}		 
			}
			if (split_dim == 0) {
				sort(children.begin(), children.end(), SortedByLeft);
			}
			else {
				sort(children.begin(), children.end(), SortedByBottom);
			}
			new_child1.resize(split_loc + 1);
			new_child2.resize(tree_node->entry_num - split_loc - 1);
			for (int i = 0; i <= split_loc; i++) {
				new_child1[i] = children[i]->id_;
			}
			for (int i = split_loc + 1; i < children.size(); i++) {
				new_child2[i - split_loc - 1] = children[i]->id_;
			}
		}
		TreeNode* sibling = CreateNode();
		sibling->is_leaf = tree_node->is_leaf;
		sibling->CopyChildren(new_child2);
		if (!sibling->is_leaf) {
			for (int i = 0; i < new_child2.size(); i++) {
				tree_nodes_[new_child2[i]]->father = sibling->id_;
			}
		}
		sibling->Set(bounding_box2);
		sibling->origin_center[0] = 0.5 * (bounding_box2.Left() + bounding_box2.Right());
		sibling->origin_center[1] = 0.5 * (bounding_box2.Top() + bounding_box2.Bottom());

		tree_node->CopyChildren(new_child1);
		if (!tree_node->is_leaf) {
			for (int i = 0; i < new_child1.size(); i++) {
				tree_nodes_[new_child1[i]]->father = tree_node->id_;
			}
		}
		tree_node->Set(bounding_box1);
		tree_node->origin_center[0] = 0.5 * (bounding_box1.Left() + bounding_box1.Right());
		tree_node->origin_center[1] = 0.5 * (bounding_box1.Bottom() + bounding_box1.Top());

		if (tree_node->father >= 0) {
			tree_nodes_[tree_node->father]->AddChildren(sibling);
			tree_nodes_[tree_node->father]->Include(bounding_box2);
			sibling->father = tree_node->father;
			//tree_node->father->AddChildren(sibling);
			//tree_node->father->Include(bounding_box2);
		}
		else {
			TreeNode* new_root = CreateNode();
			new_root->is_leaf = false;
			new_root->AddChildren(tree_node);
			new_root->AddChildren(sibling);
			new_root->Set(bounding_box1);
			new_root->Include(bounding_box2);
			root_ = new_root->id_;
			tree_node->father = new_root->id_;
			sibling->father = new_root->id_;
			height_ += 1;
			new_root->origin_center[0] = 0.5 * (new_root->left_ + new_root->right_);
			new_root->origin_center[1] = 0.5 * (new_root->bottom_ + new_root->top_);
		}
		next_node = tree_nodes_[tree_node->father];
	}
	return next_node;
}

/*
TreeNode* RTree::RRSplit(TreeNode* tree_node){
	TreeNode* next_node = nullptr;
	if(tree_node->is_overflow){
		//determine split dimension
		int split_dim = 0;
		double delta = 0;
		double new_center[2];
		new_center[0] = 0.5 * (tree_node->Right() + tree_node->Left());
		new_center[1] = 0.5 * (tree_node->Top() + tree_node->Bottom());
		double length[2];
		length[0] = tree_node->Right() - tree_node->Left();
		length[1] = tree_node->Top() - tree_node->Bottom();

		Rectangle bounding_box1;
		Rectangle bounding_box2;
		vector<int> new_child1;
		vector<int> new_child2;

		double perim_max = 2.0 * (length[0] + length[1]) - min(length[0], length[1]);
		double perim_sum[2];
		bool exist_nonoverlap[2];
		if(tree_node->is_leaf){			
			vector<Rectangle*> children(tree_node->entry_num);
			for(int i=0; i<tree_node->entry_num; i++){
				int obj_id = tree_node->children[i];
				children[i] = objects_[obj_id];
			}
			sort(children.begin(), children.end(), SortedByLeft);
			pair<double, bool> split_x = SplitPerimSum<Rectangle>(children);
			perim_sum[0] = split_x.first;
			exist_nonoverlap[0] = split_x.second;
			sort(children.begin(), children.end(), SortedByBottom);
			pair<double, bool> split_y = SplitPerimSum<Rectangle>(children);
			perim_sum[1] = split_y.first;
			exist_nonoverlap[1] = split_y.second;
			split_dim = perim_sum[0] < perim_sum[1] ? 0 : 1;

			cout<<"perim: x: "<<perim_sum[0]<<" y: "<<perim_sum[1]<<" split axis: "<<split_dim<<endl;
			getchar();

			double miu = 2.0 * (new_center[split_dim] - tree_node->origin_center[split_dim]) / length[split_dim] * (1 - 2 * TreeNode::minimum_entry / (TreeNode::maximum_entry + 1));		
			delta = RR_s * (1.0 + abs(miu));
			int split = FindMinimumSplitRR<Rectangle>(children, RR_ys, RR_y1, miu, delta, perim_max, bounding_box1, bounding_box2);
			cout<<"split position: "<<split<<endl;
			getchar();
			vector<Rectangle*> child_rec1;
			vector<Rectangle*> child_rec2;
			child_rec1.assign(children.begin(), children.begin() + split + 1);
			child_rec2.assign(children.begin() + split + 1, children.end());
			new_child1.resize(child_rec1.size());
			new_child2.resize(child_rec2.size());
			for (int i = 0; i < new_child1.size(); i++) {
				new_child1[i] = child_rec1[i]->id_; 
			}
			for (int i = 0; i < new_child2.size(); i++) {
				new_child2[i] = child_rec2[i]->id_;
			}
		}
		else{
			vector<TreeNode*> children(tree_node->entry_num);
			for(int i=0; i<tree_node->entry_num; i++){
				int node_id = tree_node->children[i];
				children[i] = tree_nodes_[node_id];
			}
			sort(children.begin(), children.end(), SortedByLeft);
			pair<double, bool> split_x = SplitPerimSum<TreeNode>(children);
			perim_sum[0] = split_x.first;
			exist_nonoverlap[0] = split_x.second;
			sort(children.begin(), children.end(), SortedByBottom);
			pair<double, bool> split_y = SplitPerimSum<TreeNode>(children);
			perim_sum[1] = split_y.first;
			exist_nonoverlap[1] = split_y.second;
			split_dim = perim_sum[0] < perim_sum[1] ? 0 : 1;
			double miu = 2.0 * (new_center[split_dim] - tree_node->origin_center[split_dim]) / length[split_dim] * (1 - 2 * TreeNode::minimum_entry / (TreeNode::maximum_entry + 1));		
			delta = RR_s * (1.0 + abs(miu));
			int split = FindMinimumSplitRR<TreeNode>(children, RR_ys, RR_y1, miu, delta, perim_max, bounding_box1, bounding_box2);
			vector<TreeNode*> child_rec1;
			vector<TreeNode*> child_rec2;
			child_rec1.assign(children.begin(), children.begin() + split + 1);
			child_rec2.assign(children.begin() + split + 1, children.end());
			new_child1.resize(child_rec1.size());
			new_child2.resize(child_rec2.size());
			for (int i = 0; i < new_child1.size(); i++) {
				new_child1[i] = child_rec1[i]->id_; 
			}
			for (int i = 0; i < new_child2.size(); i++) {
				new_child2[i] = child_rec2[i]->id_;
			}
		}
		TreeNode* sibling = CreateNode();
		sibling->is_leaf = tree_node->is_leaf;
		sibling->CopyChildren(new_child2);
		if(!sibling->is_leaf){
			for(int i=0; i < new_child2.size(); i++){
				tree_nodes_[new_child2[i]]->father = sibling->id_;
			}
		}
		sibling->Set(bounding_box2);
		sibling->origin_center[0] = 0.5 * (bounding_box2.Left() + bounding_box2.Right());
		sibling->origin_center[1] = 0.5 * (bounding_box2.Top() + bounding_box2.Bottom());

		tree_node->CopyChildren(new_child1);
		if(!tree_node->is_leaf){
			for(int i=0; i < new_child1.size(); i++){
				tree_nodes_[new_child1[i]]->father = tree_node->id_;
			}
		}
		tree_node->Set(bounding_box1);
		tree_node->origin_center[0] = 0.5 * (bounding_box1.Left() + bounding_box1.Right());
		tree_node->origin_center[1] = 0.5 * (bounding_box1.Bottom() + bounding_box1.Top());

		if (tree_node->father >= 0) {
			tree_nodes_[tree_node->father]->AddChildren(sibling);
			tree_nodes_[tree_node->father]->Include(bounding_box2);
			sibling->father = tree_node->father;
			//tree_node->father->AddChildren(sibling);
			//tree_node->father->Include(bounding_box2);
		}
		else {
			TreeNode* new_root = CreateNode();
			new_root->is_leaf = false;
			new_root->AddChildren(tree_node);
			new_root->AddChildren(sibling);
			new_root->Set(bounding_box1);
			new_root->Include(bounding_box2);
			root_ = new_root->id_;
			tree_node->father = new_root->id_;
			sibling->father = new_root->id_;
			height_ += 1;
		}
		next_node = tree_nodes_[tree_node->father];
	}
	return next_node;
}
*/

double RTree::MinDistanceToRec(double x, double y, int rec_id) {
	Rectangle* rec = objects_[rec_id];
	double min_distance = 0.0;
	if (x > rec->right_) {
		min_distance += (x - rec->right_) * (x - rec->right_);
	}
	else if (x < rec->left_) {
		min_distance += (rec->left_ - x) * (rec->left_ - x);
	}
	if (y > rec->top_) {
		min_distance += (y - rec->top_) * (y - rec->top_);
	}
	else if (y < rec->bottom_) {
		min_distance += (rec->bottom_ - y) * (rec->bottom_ - y);
	}
	min_distance = sqrt(min_distance);
	return min_distance;
}

double RTree::MinDistanceToNode(double x, double y, int tree_node_id) {
	TreeNode* tree_node = tree_nodes_[tree_node_id];
	double min_distance = 0.0;
	if (x > tree_node->right_) {
		min_distance += (x - tree_node->right_) * (x - tree_node->right_);
	}
	else if (x < tree_node->left_) {
		min_distance += (tree_node->left_ - x) * (tree_node->left_ - x);
	}
	if (y > tree_node->top_) {
		min_distance += (y - tree_node->top_) * (y - tree_node->top_);
	}
	else if (y < tree_node->bottom_) {
		min_distance += (tree_node->bottom_ - y) * (tree_node->bottom_ - y);
	}
	min_distance = sqrt(min_distance);
	return min_distance;
}


TreeNode* RTree::SplitStepByStep(TreeNode *tree_node, SPLIT_STRATEGY strategy) {
	TreeNode* next_node = nullptr;
	if (tree_node->is_overflow) {
		//vector<TreeNode*> new_child1;
		//vector<TreeNode*> new_child2;
		//vector<Rectangle*> new_child1_rec;
		//vector<Rectangle*> new_child2_rec;
		vector<int> new_child1;
		vector<int> new_child2;
		Rectangle bounding_box1;
		Rectangle bounding_box2;
		switch (strategy) {
		case SPL_MIN_AREA: {

			double minimum_area = DBL_MAX;
			double minimum_overlap = DBL_MAX;

			//choose the split with the minimum total area, break the tie by preferring the split with smaller overlap
			if (tree_node->is_leaf) {
				vector<Rectangle*> recs(tree_node->entry_num);
				for (int i = 0; i < tree_node->entry_num; i++) {
					int obj_id = tree_node->children[i];
					recs[i] = objects_[obj_id];
				}
				vector<Rectangle*> child_rec1;
				vector<Rectangle*> child_rec2;
				//sort by left
				sort(recs.begin(), recs.end(), SortedByLeft);
				Rectangle rec1;
				Rectangle rec2;
				int split = FindMinimumSplit<Rectangle>(recs, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}

				//sort by right
				sort(recs.begin(), recs.end(), SortedByRight);
				split = FindMinimumSplit<Rectangle>(recs, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}

				//sort by bottom
				sort(recs.begin(), recs.end(), SortedByBottom);
				split = FindMinimumSplit<Rectangle>(recs, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}

				//sort by top
				sort(recs.begin(), recs.end(), SortedByTop);
				split = FindMinimumSplit<Rectangle>(recs, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				new_child1.resize(child_rec1.size());
				new_child2.resize(child_rec2.size());
				for (int i = 0; i < new_child1.size(); i++) {
					new_child1[i] = child_rec1[i]->id_; 
				}
				for (int i = 0; i < new_child2.size(); i++) {
					new_child2[i] = child_rec2[i]->id_;
				}
			}
			else {
				vector<TreeNode*> children(tree_node->entry_num);
				for (int i = 0; i < tree_node->entry_num; i++) {
					int obj_id = tree_node->children[i];
					children[i] = tree_nodes_[obj_id];
				}
				vector<TreeNode*> child_treenode1;
				vector<TreeNode*> child_treenode2;
				//sort by left
				sort(children.begin(), children.end(), SortedByLeft);
				Rectangle rec1;
				Rectangle rec2;
				int split = FindMinimumSplit<TreeNode>(children, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by right
				sort(children.begin(), children.end(), SortedByRight);
				split = FindMinimumSplit<TreeNode>(children, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by bottom
				sort(children.begin(), children.end(), SortedByBottom);
				split = FindMinimumSplit<TreeNode>(children, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by top
				sort(children.begin(), children.end(), SortedByTop);
				split = FindMinimumSplit<TreeNode>(children, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				new_child1.resize(child_treenode1.size());
				new_child2.resize(child_treenode2.size());
				for (int i = 0; i < child_treenode1.size(); i++) {
					new_child1[i] = child_treenode1[i]->id_;
				}
				for (int i = 0; i < child_treenode2.size(); i++) {
					new_child2[i] = child_treenode2[i]->id_;
				}

			}
			break;
		}
		case SPL_MIN_MARGIN: {
			double minimum_perimeter = DBL_MAX;
			double minimum_overlap = DBL_MAX;
			//choose the split with the minimum total perimeter, break the tie by preferring the split with smaller overlap

			if (tree_node->is_leaf) {
				vector<Rectangle*> recs(tree_node->entry_num);
				for (int i = 0; i < tree_node->entry_num; i++) {
					int obj_id = tree_node->children[i];
					recs[i] = objects_[obj_id];
				}
				vector<Rectangle*> child_rec1;
				vector<Rectangle*> child_rec2;
				Rectangle rec1;
				Rectangle rec2;
				//sort by left
				sort(recs.begin(), recs.end(), SortedByLeft);
				int split = FindMinimumSplit<Rectangle>(recs, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by right
				//sort(recs.begin(), recs.end(), SortedByRight);
				//split = FindMinimumSplit<Rectangle>(recs, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
				//if (split >= 0) {
				//	child_rec1.assign(recs.begin(), recs.begin() + split + 1);
				//	child_rec2.assign(recs.begin() + split + 1, recs.end());
				//	bounding_box1.Set(rec1);
				//	bounding_box2.Set(rec2);
				//}
				//sort by bottom 
				sort(recs.begin(), recs.end(), SortedByBottom);
				split = FindMinimumSplit<Rectangle>(recs, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by top
				//sort(recs.begin(), recs.end(), SortedByTop);
				//split = FindMinimumSplit<Rectangle>(recs, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
				//if (split >= 0) {
				//	child_rec1.assign(recs.begin(), recs.begin() + split + 1);
				//	child_rec2.assign(recs.begin() + split + 1, recs.end());
				//	bounding_box1.Set(rec1);
				//	bounding_box2.Set(rec2);
				//}
                new_child1.resize(child_rec1.size());
                new_child2.resize(child_rec2.size());
                for (int i = 0; i < new_child1.size(); i++) {
					new_child1[i] = child_rec1[i]->id_;
                }
                for (int i = 0; i < new_child2.size(); i++) {
                    new_child2[i] = child_rec2[i]->id_;
                }
			}
			else {
				vector<TreeNode*> children(tree_node->entry_num);
				for (int i = 0; i < tree_node->entry_num; i++) {
					int child_id = tree_node->children[i];
					children[i] = tree_nodes_[child_id];
				}
				vector<TreeNode*> child_treenode1;
				vector<TreeNode*> child_treenode2;
				Rectangle rec1;
				Rectangle rec2;
				//sort by left
				int real_split=-1;
				int real_dim = -1;
				sort(children.begin(), children.end(), SortedByLeft);
				int split = FindMinimumSplit<TreeNode>(children, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by right
				//sort(children.begin(), children.end(), SortedByRight);
				//split = FindMinimumSplit<TreeNode>(children, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
				//if (split >= 0) {
				//	child_treenode1.assign(children.begin(), children.begin() + split + 1);
				//	child_treenode2.assign(children.begin() + split + 1, children.end());
				//	bounding_box1.Set(rec1);
				//	bounding_box2.Set(rec2);
				//}
				//sort by bottom
				sort(children.begin(), children.end(), SortedByBottom);
				split = FindMinimumSplit<TreeNode>(children, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by top
				//sort(children.begin(), children.end(), SortedByTop);
				//split = FindMinimumSplit<TreeNode>(children, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
				//if (split >= 0) {
				//	child_treenode1.assign(children.begin(), children.begin() + split + 1);
				//	child_treenode2.assign(children.begin() + split + 1, children.end());
				//	bounding_box1.Set(rec1);
				//	bounding_box2.Set(rec2);
				//}
				new_child1.resize(child_treenode1.size());
				new_child2.resize(child_treenode2.size());
				for (int i = 0; i < child_treenode1.size(); i++) {
					new_child1[i] = child_treenode1[i]->id_;
				}
				for (int i = 0; i < child_treenode2.size(); i++) {
					new_child2[i] = child_treenode2[i]->id_;
				}
			}
			
			break;
		}
		case SPL_MIN_OVERLAP: {
			double minimum_overlap = DBL_MAX;
			double minimum_area = DBL_MAX;
			//choose the split with the minimum overlap, break the tie by preferring the split with smaller total area

			if (tree_node->is_leaf) {
				vector<Rectangle*> recs(tree_node->entry_num);
				for (int i = 0; i < tree_node->entry_num; i++) {
					int obj_id = tree_node->children[i];
					recs[i] = objects_[obj_id]; 
				}
				vector<Rectangle*> child_rec1;
				vector<Rectangle*> child_rec2;
				Rectangle rec1;
				Rectangle rec2;
				//sort by left
				sort(recs.begin(), recs.end(), SortedByLeft);
				int split = FindMinimumSplit<Rectangle>(recs, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by right
				sort(recs.begin(), recs.end(), SortedByRight);
				split = FindMinimumSplit<Rectangle>(recs, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by bottom
				sort(recs.begin(), recs.end(), SortedByBottom);
				split = FindMinimumSplit<Rectangle>(recs, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by top
				sort(recs.begin(), recs.end(), SortedByTop);
				split = FindMinimumSplit<Rectangle>(recs, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
                new_child1.resize(child_rec1.size());
                new_child2.resize(child_rec2.size());
                for (int i = 0; i < child_rec1.size(); i++) {
					new_child1[i] = child_rec1[i]->id_;
                }
                for (int i = 0; i < child_rec2.size(); i++) {
					new_child2[i] = child_rec2[i]->id_;
                }
			}
			else {
				vector<TreeNode*> children(tree_node->entry_num);
				for (int i = 0; i < tree_node->entry_num; i++) {
					int child_id = tree_node->children[i];
					children[i] = tree_nodes_[child_id];
				}
				vector<TreeNode*> child_treenode1;
				vector<TreeNode*> child_treenode2;
				Rectangle rec1;
				Rectangle rec2;
				//sort by left
				sort(children.begin(), children.end(), SortedByLeft);
				int split = FindMinimumSplit<TreeNode>(children, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by right
				sort(children.begin(), children.end(), SortedByRight);
				split = FindMinimumSplit<TreeNode>(children, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by bottom
				sort(children.begin(), children.end(), SortedByBottom);
				split = FindMinimumSplit<TreeNode>(children, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by top
				sort(children.begin(), children.end(), SortedByTop);
				split = FindMinimumSplit<TreeNode>(children, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				new_child1.resize(child_treenode1.size());
				new_child2.resize(child_treenode2.size());
				for (int i = 0; i < child_treenode1.size(); i++) {
					new_child1[i] = child_treenode1[i]->id_;
				}
				for (int i = 0; i < child_treenode2.size(); i++) {
					new_child2[i] = child_treenode2[i]->id_;
				}
			}
			break;
		}
		case SPL_QUADRATIC: {
			int seed1 = -1;
			int seed2 = -1;
			int seed_idx1 = -1, seed_idx2 = -1;
			double max_waste = -DBL_MAX;
			//find the pair of children that waste the most area were they to be inserted in the same node
			for (int i = 0; i < tree_node->entry_num - 1; i++) {
				for (int j = i + 1; j < tree_node->entry_num; j++) {
					double waste = 0;
					unsigned int id1 = tree_node->children[i];
					unsigned int id2 = tree_node->children[j];
					if (tree_node->is_leaf) {
						waste = objects_[id1]->Merge(*objects_[id2]).Area() - objects_[id1]->Area() - objects_[id2]->Area();
						//waste = ((Rectangle*)tree_node->children[i])->Merge(*((Rectangle*)tree_node->children[j])).Area() - ((Rectangle*)tree_node->children[i])->Area() - ((Rectangle*)tree_node->children[j])->Area();
					}
					else {
						waste = tree_nodes_[id1]->Merge(*tree_nodes_[id2]).Area() - tree_nodes_[id1]->Area() - tree_nodes_[id2]->Area();
						//waste = tree_node->children[i]->Merge(*tree_node->children[j]).Area() - tree_node->children[i]->Area() - tree_node->children[j]->Area();
					}
					if (waste > max_waste) {
						max_waste = waste;
						seed1 = id1;
						seed2 = id2;
						seed_idx1 = i;
						seed_idx2 = j;
					}
				}
			}
			//list<TreeNode*> child1;
			//list<TreeNode*> child2;
			list<int> child1;
			list<int> child2;
			child1.push_back(seed1);
			child2.push_back(seed2);
			if (tree_node->is_leaf) {
				bounding_box1.Set(*objects_[seed1]);
				bounding_box2.Set(*objects_[seed2]);
			}
			else {
				bounding_box1.Set(*tree_nodes_[seed1]);
				bounding_box2.Set(*tree_nodes_[seed2]);
			}
			list<int> unassigned_entry;
			for (int i = 0; i < tree_node->entry_num; i++) {
				if (i == seed_idx1 || i == seed_idx2)continue;
				unassigned_entry.push_back(tree_node->children[i]);
			}
			while (!unassigned_entry.empty()) {
				//make sure the two child nodes are balanced.
				if (unassigned_entry.size() + child1.size() == TreeNode::minimum_entry) {
					for (auto it = unassigned_entry.begin(); it != unassigned_entry.end(); ++it) {
						child1.push_back(*it);
						if (tree_node->is_leaf) {
							Rectangle* rec_ptr = objects_[*it];
							bounding_box1.Include(*rec_ptr);
						}
						else {
							TreeNode* node_ptr = tree_nodes_[*it];
							bounding_box1.Include(*node_ptr);
						}
						
					}
					break;
				}
				if (unassigned_entry.size() + child2.size() == TreeNode::minimum_entry) {
					for (auto it = unassigned_entry.begin(); it != unassigned_entry.end(); ++it) {
						child2.push_back(*it);
						if (tree_node->is_leaf) {
							Rectangle* rec_ptr = objects_[*it];
							bounding_box2.Include(*rec_ptr);
						}
						else {
							TreeNode* node_ptr = tree_nodes_[*it];
							bounding_box2.Include(*node_ptr);
						}
						
					}
					break;
				}
				//pick next: pick an unassigned entry that maximizes the difference between adding into different groups
				double max_difference = - DBL_MAX;
				double new_area1 = 0, new_area2 = 0;
				list<int>::iterator iter;
				int next_entry;
				for (auto it = unassigned_entry.begin(); it != unassigned_entry.end(); ++it) {
					double d1 = 0, d2 = 0;
					if (tree_node->is_leaf) {
						d1 = bounding_box1.Merge(*objects_[*it]).Area();
						d2 = bounding_box2.Merge(*objects_[*it]).Area();
					}
					else {
						d1 = bounding_box1.Merge(*tree_nodes_[*it]).Area();
						d2 = bounding_box2.Merge(*tree_nodes_[*it]).Area();
					}
					double difference = d1 > d2 ? d1 - d2 : d2 - d1;
					if (difference > max_difference) {
						max_difference = difference;
						iter = it;
						next_entry = *it;
						new_area1 = d1;
						new_area2 = d2;
					}
				}
				unassigned_entry.erase(iter);
				//add the entry to the group with smaller area
				Rectangle *chosen_bounding_box = nullptr;
				if (new_area1 < new_area2) {
					child1.push_back(next_entry);
					chosen_bounding_box = &bounding_box1;
				}
				else if (new_area1 > new_area2) {
					child2.push_back(next_entry);
					chosen_bounding_box = &bounding_box2;
				}
				else {
					if (child1.size() < child2.size()) {
						child1.push_back(next_entry);
						chosen_bounding_box = &bounding_box1;
					}
					else {
						child2.push_back(next_entry);
						chosen_bounding_box = &bounding_box2;
					}
				}
				if (tree_node->is_leaf) {
					chosen_bounding_box->Include(*objects_[next_entry]);
				}
				else {
					chosen_bounding_box->Include(*tree_nodes_[next_entry]);
				}
			}
			new_child1.assign(child1.begin(), child1.end());
			new_child2.assign(child2.begin(), child2.end());
			break;
		}

		case SPL_GREENE: {
			int seed1 = -1;
			int seed2 = -1;
			double max_waste = - DBL_MAX;
			for (int i = 0; i < tree_node->entry_num - 1; i++) {
				for (int j = i + 1; j < tree_node->entry_num; j++) {
					double waste = 0;
					int id1 = tree_node->children[i];
					int id2 = tree_node->children[j];
					if (tree_node->is_leaf) {
						waste = objects_[id1]->Merge(*objects_[id2]).Area() - objects_[id1]->Area() - objects_[id2]->Area();
						//waste = ((Rectangle*)tree_node->children[i])->Merge(*((Rectangle*)tree_node->children[j])).Area() - ((Rectangle*)tree_node->children[i])->Area() - ((Rectangle*)tree_node->children[j])->Area();
					}
					else {
						waste = tree_nodes_[id1]->Merge(*tree_nodes_[id2]).Area() - tree_nodes_[id1]->Area() - tree_nodes_[id2]->Area();
						//waste = tree_node->children[i]->Merge(*tree_node->children[j]).Area() - tree_node->children[i]->Area() - tree_node->children[j]->Area();
					}
					if (waste > max_waste) {
						max_waste = waste;
						seed1 = id1;
						seed2 = id2;
					}
				}
			}

			double max_seed_left, min_seed_right, max_seed_bottom, min_seed_top;
			if (tree_node->is_leaf) {
				max_seed_left = max(objects_[seed1]->Left(), objects_[seed2]->Left());
				min_seed_right = min(objects_[seed1]->Right(), objects_[seed2]->Right());
				max_seed_bottom = max(objects_[seed1]->Bottom(), objects_[seed2]->Bottom());
				min_seed_top = min(objects_[seed1]->Top(), objects_[seed2]->Top());
			}
			else {
				max_seed_left = max(tree_nodes_[seed1]->Left(), tree_nodes_[seed2]->Left());
				min_seed_right = min(tree_nodes_[seed1]->Right(), tree_nodes_[seed2]->Right());
				max_seed_bottom = max(tree_nodes_[seed1]->Bottom(), tree_nodes_[seed2]->Bottom());
				min_seed_top = min(tree_nodes_[seed1]->Top(), tree_nodes_[seed2]->Top());
			}
			double x_seperation = min_seed_right > max_seed_left ? (min_seed_right - max_seed_left) : (max_seed_left - min_seed_right);
			double y_seperation = min_seed_top > max_seed_bottom ? (min_seed_top - max_seed_bottom) : (max_seed_bottom - min_seed_top);

			x_seperation = x_seperation / (tree_node->Right() - tree_node->Left());
			y_seperation = y_seperation / (tree_node->Top() - tree_node->Bottom());
			
			vector<Rectangle*>	recs;
			vector<TreeNode*> child_nodes;

			if (tree_node->is_leaf) {
				recs.resize(tree_node->entry_num);
				for (int i = 0; i < tree_node->entry_num; i++) {
					recs[i] = objects_[tree_node->children[i]]; 
				}
			}
			else {
				child_nodes.resize(tree_node->entry_num);
				for (int i = 0; i < tree_node->entry_num; i++) {
					child_nodes[i] = tree_nodes_[tree_node->children[i]];
				}
			}

			if (x_seperation < y_seperation) {
				if (tree_node->is_leaf) {
					sort(recs.begin(), recs.end(), SortedByBottom);
				}
				else {
					sort(child_nodes.begin(), child_nodes.end(), SortedByBottom);
				}
			}
			else {
				if (tree_node->is_leaf) {
					sort(recs.begin(), recs.end(), SortedByLeft);
				}
				else {
					sort(child_nodes.begin(), child_nodes.end(), SortedByLeft);
				}
			}
			if (tree_node->is_leaf) {
				new_child1.resize(tree_node->entry_num / 2);
				new_child2.resize(tree_node->entry_num / 2);
				for (int i = 0; i < tree_node->entry_num / 2; i++) {
					new_child1[i] = recs[i]->id_;
					new_child2[i] = recs[tree_node->entry_num - 1 - i]->id_;
				}
				bounding_box1 = MergeRange<Rectangle>(recs, 0, tree_node->entry_num / 2);
				bounding_box2 = MergeRange<Rectangle>(recs, tree_node->entry_num - tree_node->entry_num / 2, tree_node->entry_num);
				if (tree_node->entry_num % 2 == 1) {
					Rectangle rec1 = bounding_box1.Merge(*recs[tree_node->entry_num / 2]);
					Rectangle rec2 = bounding_box2.Merge(*recs[tree_node->entry_num / 2]);
					double area_increase1 = rec1.Area() - bounding_box1.Area();
					double area_increase2 = rec2.Area() - bounding_box2.Area();
					if (area_increase1 < area_increase2) {
						new_child1.push_back(recs[tree_node->entry_num / 2]->id_);
						bounding_box1 = rec1;
					}
					else {
						new_child2.push_back(recs[tree_node->entry_num / 2]->id_);
						bounding_box2 = rec2;
					}
				}
			}
			else {
				new_child1.resize(tree_node->entry_num / 2);
				new_child2.resize(tree_node->entry_num / 2);
				for (int i = 0; i < tree_node->entry_num / 2; i++) {
					new_child1[i] = child_nodes[i]->id_;
					new_child2[i] = child_nodes[tree_node->entry_num - 1 - i]->id_;
				}
				
				bounding_box1 = MergeRange<TreeNode>(child_nodes, 0, tree_node->entry_num / 2);
				bounding_box2 = MergeRange<TreeNode>(child_nodes, tree_node->entry_num - tree_node->entry_num/2, tree_node->entry_num);
				if (tree_node->entry_num % 2 == 1) {
					Rectangle rec1 = bounding_box1.Merge(*child_nodes[tree_node->entry_num / 2]);
					Rectangle rec2 = bounding_box2.Merge(*child_nodes[tree_node->entry_num / 2]);
					double area_increase1 = rec1.Area() - bounding_box1.Area();
					double area_increase2 = rec2.Area() - bounding_box2.Area();
					if (area_increase1 < area_increase2) {
						new_child1.push_back(child_nodes[tree_node->entry_num / 2]->id_);
						bounding_box1 = rec1;
					}
					else {
						new_child2.push_back(child_nodes[tree_node->entry_num / 2]->id_);
						bounding_box2 = rec2;
					}
				}
			}

			break;
		}
		}
		
		TreeNode* sibling = CreateNode();
		sibling->is_leaf = tree_node->is_leaf;
		if (!sibling->CopyChildren(new_child2)) {
			cout << "Error" << endl;
			exit(0);
		};
		if(!sibling->is_leaf){
            for (int i = 0; i < new_child2.size(); i++) {
				tree_nodes_[new_child2[i]]->father = sibling->id_;
                //new_child2[i]->father = sibling;
            }
		}
		sibling->Set(bounding_box2);
		sibling->origin_center[0] = 0.5 * (sibling->Left() + sibling->Right());
		sibling->origin_center[1] = 0.5 * (sibling->Bottom() + sibling->Top());
		if (!tree_node->CopyChildren(new_child1)) {
			cout << "Error" << endl;
			exit(0);
		};
		if(!tree_node->is_leaf){
            for (int i = 0; i < new_child1.size(); i++) {
				tree_nodes_[new_child1[i]]->father = tree_node->id_;
                //new_child1[i]->father = tree_node;
            }
		}
		tree_node->Set(bounding_box1);
		tree_node->origin_center[0] = 0.5 * (tree_node->Left() + tree_node->Right());
		tree_node->origin_center[1] = 0.5 * (tree_node->Bottom() + tree_node->Top());
		if (tree_node->father >= 0) {
			tree_nodes_[tree_node->father]->AddChildren(sibling);
			tree_nodes_[tree_node->father]->Include(bounding_box2);
			sibling->father = tree_node->father;
			//tree_node->father->AddChildren(sibling);
			//tree_node->father->Include(bounding_box2);
		}
		else {
			TreeNode* new_root = CreateNode();
			new_root->is_leaf = false;
			new_root->AddChildren(tree_node);
			new_root->AddChildren(sibling);
			new_root->Set(bounding_box1);
			new_root->Include(bounding_box2);
			new_root->origin_center[0] = 0.5 * (new_root->Left() + new_root->Right());
			new_root->origin_center[1] = 0.5 * (new_root->Bottom() + new_root->Top());
			root_ = new_root->id_;
			tree_node->father = new_root->id_;
			sibling->father = new_root->id_;
			height_ += 1;
		}
		next_node = tree_nodes_[tree_node->father];
	}

	return next_node;
}


int GetQueryResult(RTree* rtree) {
	return rtree->result_count;
}

int RTree::KNNQuery(double x, double y, int k, vector<int>& query_results){
	priority_queue < pair<double, int>, vector<pair<double, int> >, std::greater<pair<double, int> > > pqueue;
	vector<pair<double, int> > results;
	int access_num = 0;
	pqueue.emplace(MinDistanceToNode(x, y, root_), root_);
	vector<pair<double, int> > tmp;
	while (!pqueue.empty()) {
		pair<double, int> top = pqueue.top();
		if (results.size() >= k) {
			if (top.first > results[k - 1].first) {
				break;
			}
		}
		access_num += 1;
		pqueue.pop();
		TreeNode* node = tree_nodes_[top.second];
		if (node->is_leaf) {
			tmp.resize(node->entry_num);
			for (int i = 0; i < node->entry_num; i++) {
				int child = node->children[i];
				double d = MinDistanceToRec(x, y, child);
				tmp[i].first = d;
				tmp[i].second = child;
			}
			sort(tmp.begin(), tmp.end());
			int idx_result = 0;
			int idx_tmp = 0;
			list<pair<double, int> > topk;
			while (topk.size() < k) {
				if (idx_result == results.size() && idx_tmp == tmp.size()) {
					break;
				}
				if (idx_result == results.size()) {
					topk.push_back(tmp[idx_tmp]);
					idx_tmp += 1;
					continue;
				}
				if (idx_tmp == tmp.size()) {
					topk.push_back(results[idx_result]);
					idx_result += 1;
					continue;
				}
				if (results[idx_result].first < tmp[idx_tmp].first) {
					topk.push_back(results[idx_result]);
					idx_result += 1;
				}
				else if (results[idx_result].first > tmp[idx_tmp].first) {
					topk.push_back(tmp[idx_tmp]);
					idx_tmp += 1;
				}
				else {
					topk.push_back(tmp[idx_tmp]);
					topk.push_back(results[idx_result]);
					idx_tmp += 1;
					idx_result += 1;
				}
				
			}
			results.assign(topk.begin(), topk.end());
		}
		else {
			for (int i = 0; i < node->entry_num; i++) {
				int child = node->children[i];
				double d = MinDistanceToNode(x, y, child);
				//cout << "distance between query point and node " << child << " is " << d << endl;
				if (results.size() >= k) {
					double ub = results[k - 1].first;
					if (d < ub) {
						pqueue.emplace(d, child);
					}
				}
				else {
					pqueue.emplace(d, child);
				}
			}
		}
	}
	query_results.resize(k);
	for (int i = 0; i < results.size(); i++) {
		query_results[i] = results[i].second;
	}
	return access_num;
}

void RTree::RetrieveForReinsert(TreeNode* tree_node, list<int>& candidates) {
	vector<pair<double, int> > entries(tree_node->entry_num);
	for (int i = 0; i < tree_node->entry_num; i++) {
		entries[i].second = tree_node->children[i];
		Rectangle* rec = objects_[entries[i].second];
		double x_diff = 0.5 * (rec->left_ + rec->right_) - 0.5 * (tree_node->left_ + tree_node->right_);
		double y_diff = 0.5 * (rec->top_ + rec->bottom_) - 0.5 * (tree_node->top_ + tree_node->bottom_);
		entries[i].first = sqrt(x_diff * x_diff + y_diff * y_diff);
	}
	sort(entries.begin(), entries.end());
	int retrieve_num = int(ceil(tree_node->entry_num * 0.3));	
	tree_node->Set(*objects_[entries[0].second]);
	tree_node->children.clear();
	tree_node->entry_num = 0;
	tree_node->AddChildren(entries[0].second);
	tree_node->is_overflow = false;
	for (int i = 1; i < entries.size(); i++) {
		if (i < entries.size() - retrieve_num) {
			tree_node->Include(*objects_[entries[i].second]);
			tree_node->AddChildren(entries[i].second);
		}
		else {
			candidates.push_back(entries[i].second);
		}
	}
}

void RTree::UpdateMBRForReinsert(TreeNode* tree_node) {
	TreeNode* iter = tree_node;
	while (iter->father >= 0) {	
		iter = tree_nodes_[iter->father];
		iter->Set(*tree_nodes_[iter->children[0]]);
		for (int i = 1; i < iter->entry_num; i++) {
			iter->Include(*tree_nodes_[iter->children[i]]);
		}
	}

}

int RTree::Query(Rectangle& rectangle) {
	result_count = 0;
	list<TreeNode*> queue;
	queue.push_back(tree_nodes_[root_]);
	stats_.Reset();
	TreeNode* iter = tree_nodes_[root_];
	if (!iter->IsOverlap(&rectangle)) {
		return 0;
	}
	while (!queue.empty()) {
		iter = queue.front();
		stats_.node_access += 1;
		queue.pop_front();
		if (iter->is_leaf) {
			for (int i = 0; i < iter->entry_num; i++) {
				Rectangle* rec_iter = objects_[iter->children[i]];
				if (rec_iter->IsOverlap(&rectangle)) {
					result_count += 1;
				}
			}
		}
		else {
			for (int i = 0; i < iter->entry_num; i++) {
				TreeNode* node = tree_nodes_[iter->children[i]];
				if (node->IsOverlap(&rectangle)) {
					queue.push_back(node);
				}
			}
		}
	}
	return 1;
}

TreeNode* RTree::CreateNode() {
	TreeNode* node = new TreeNode();
	node->id_ = tree_nodes_.size();
	tree_nodes_.push_back(node);
	assert(tree_nodes_[node->id_]->id_ == node->id_);
	return node;
}


TreeNode* RTree::TryInsertStepByStep(const Rectangle* rectangle, TreeNode* tree_node) {
	TreeNode* next_node = nullptr;
	if (!tree_node->is_leaf) {
		switch (insert_strategy_) {
		case INS_AREA: {
			double min_area_increase = DBL_MAX;
			for (int idx = 0; idx < tree_node->entry_num; idx++) {
				TreeNode* it = tree_nodes_[tree_node->children[idx]];
				Rectangle new_rectangle = it->Merge(*rectangle);
				double area_increase = new_rectangle.Area() - it->Area();
				if (area_increase < min_area_increase) {
					//choose the subtree with the smaller area increase
					min_area_increase = area_increase;
					next_node = it;
				}
				else if (area_increase == min_area_increase) {
					//break the tie by favoring the smaller MBR
					if (next_node->Area() > it->Area()) {
						next_node = it;
					}
				}
			}
			break;
		}
		case INS_MARGIN: {
			double min_margin_increase = DBL_MAX;
			for (int idx = 0; idx < tree_node->entry_num; idx++) {
				TreeNode* it = tree_nodes_[tree_node->children[idx]];
				Rectangle new_rectangle = it->Merge(*rectangle);
				double margin_increase = new_rectangle.Perimeter() - it->Perimeter();
				if (margin_increase < min_margin_increase) {
					//choose the subtree with smaller perimeter increase
					next_node = it;
					min_margin_increase = margin_increase;
				}
				else if (margin_increase == min_margin_increase) {
					//break the tie by favoring the smaller MBR
					if (next_node->Area() > it->Area()) {
						next_node = it;
					}
				}
			}
			break;
		}
		case INS_OVERLAP: {
			double min_overlap_increase = DBL_MAX;
			for (int idx = 0; idx < tree_node->entry_num; idx++) {
				TreeNode* it = tree_nodes_[tree_node->children[idx]];
				Rectangle new_rectangle = it->Merge(*rectangle);
				double overlap_increase = 0;
				for (int idx2 = 0; idx2 < tree_node->entry_num; idx2++) {
					if (idx == idx2)continue;
					TreeNode* it2 = tree_nodes_[tree_node->children[idx2]];
					//overlap_increase += Overlap(new_rectangle, it2->bounding_box) - Overlap(it->bounding_box, it2->bounding_box);
					overlap_increase += SplitOverlap(new_rectangle, *it2) - SplitOverlap(*it, *it2);
				}
				if (overlap_increase < min_overlap_increase) {
					//choose the subtree with smaller overlap increase
					min_overlap_increase = overlap_increase;
					next_node = it;
				}
				else if (overlap_increase == min_overlap_increase) {
					//break the tie by favoring the one with smaller area increase
					//double area_increase = Area(Merge(rectangle, it->bounding_box)) - Area(it->bounding_box);
					double area_increase = it->Merge(*rectangle).Area() - it->Area();
					if (area_increase < next_node->Merge(*rectangle).Area() - next_node->Area()) {
						next_node = it;
					}

				}
			}
			break;
		}
		case INS_RANDOM: {
			int chosen_child = rand() % tree_node->entry_num;
			next_node = tree_nodes_[tree_node->children[chosen_child]];
			break;
		}
		}
	}
	return next_node;
}

TreeNode* RTree::RRInsert(Rectangle* rectangle, TreeNode* tree_node){
	TreeNode* next_node = nullptr;
	if(tree_node->is_leaf){
		if(objects_.size() == 0){
			tree_node->origin_center[0] = 0.5 * (rectangle->Left() + rectangle->Right());
			tree_node->origin_center[1] = 0.5 * (rectangle->Bottom() + rectangle->Top());
		}
		if(tree_node->entry_num == 0){
			tree_node->Set(*rectangle);
			tree_node->origin_center[0] = 0.5 * (tree_node->Right() + tree_node->Left());
			tree_node->origin_center[1] = 0.5 * (tree_node->Bottom() + tree_node->Top());
		}
		else{
			tree_node->Include(*rectangle);
		}
		tree_node->AddChildren(rectangle->id_);
	}
	else{
		tree_node->Include(*rectangle);
		list<int> COV;
		for(int i=0; i < tree_node->entry_num; i++){
			int node_id = tree_node->children[i];
			if(tree_nodes_[node_id]->Contains(rectangle)){
				COV.push_back(node_id);
			}
		}
		if(COV.empty()){
			vector<pair<double, int> > sequence(tree_node->entry_num);
			Rectangle r;
			for(int i=0; i<tree_node->entry_num; i++){
				int node_id = tree_node->children[i];
				r.Set(*tree_nodes_[node_id]);
				r.Include(*rectangle);
				sequence[i].first = r.Perimeter() - tree_nodes_[node_id]->Perimeter();
				sequence[i].second = node_id;
			}
			sort(sequence.begin(), sequence.end());
			vector<double> margin_ovlp_perim(tree_node->entry_num, 0);
			double total_ovlp_perim = 0;
			for(int i=0; i<tree_node->entry_num; i++){
				margin_ovlp_perim[i] = MarginOvlpPerim(tree_nodes_[sequence[0].second], rectangle, tree_nodes_[sequence[i].second]);
				total_ovlp_perim += margin_ovlp_perim[i];
			}
			if(total_ovlp_perim == 0){
				next_node = tree_nodes_[sequence[0].second];
				return next_node;
			}
			int p = 1;
			for(int i = 1; i<tree_node->entry_num; i++){
				if(margin_ovlp_perim[i] > margin_ovlp_perim[p]){
					p = i;
				}
			}
			vector<double> margin_ovlp_area(p+1, 0);
			for(int t=0; t<= p; t++){
				margin_ovlp_area[t] = 0;
				for(int j=0; j<=p; j++){
					if(t == j)continue;
					double ovlp_tj = MarginOvlpArea(tree_nodes_[sequence[t].second], rectangle, tree_nodes_[sequence[j].second]);
					margin_ovlp_area[t] += ovlp_tj;
				}
				if(margin_ovlp_area[t] == 0){
					next_node = tree_nodes_[sequence[t].second];
					return next_node;
				}
			}
			double min_ovlp = DBL_MAX;
			for(int t=0; t<=p; t++){
				if(margin_ovlp_area[t] < min_ovlp){
					min_ovlp = margin_ovlp_area[t];
					next_node = tree_nodes_[sequence[t].second];
				}
			}
			return next_node;

		}
		else{
			double min_volume = DBL_MAX;
			for(auto it = COV.begin(); it != COV.end(); ++it){
				int node_id = *it;
				if(tree_nodes_[node_id]->Area() < min_volume){
					min_volume = tree_nodes_[node_id]->Area();
					next_node = tree_nodes_[node_id];
				}
			}
			return next_node;
		}
	}
	return next_node;
}

TreeNode* RTree::InsertStepByStep(const Rectangle *rectangle, TreeNode *tree_node, INSERT_STRATEGY strategy) {
	TreeNode* next_node = nullptr;
	if (tree_node->is_leaf) {
		//this tree node is a leaf node
		if (tree_node->entry_num == 0) {
			tree_node->Set(*rectangle);
			tree_node->origin_center[0] = 0.5 * (tree_node->Left() + tree_node->Right());
			tree_node->origin_center[1] = 0.5 * (tree_node->Bottom() + tree_node->Top());
		}
		else {
			tree_node->Include(*rectangle);
		}
		tree_node->AddChildren(rectangle->id_);
	}
	else {
		//this tree node is an internal node
		switch (strategy) {
		case INS_AREA: {
			double min_area_increase = DBL_MAX;
			for (int idx = 0; idx < tree_node->entry_num; idx++) {
				TreeNode* it = tree_nodes_[tree_node->children[idx]];
				Rectangle new_rectangle = it->Merge(*rectangle);
				double area_increase = new_rectangle.Area() - it->Area();
				if (area_increase < min_area_increase) {
					//choose the subtree with the smaller area increase
					min_area_increase = area_increase;
					next_node = it;
				}
				else if (area_increase == min_area_increase) {
					//break the tie by favoring the smaller MBR
					if (next_node->Area() > it->Area()) {
						next_node = it;
					}
				}
			}
			break;
		}
		case INS_MARGIN: {
			double min_margin_increase = DBL_MAX;
			for (int idx = 0; idx < tree_node->entry_num; idx++) {
				TreeNode* it = tree_nodes_[tree_node->children[idx]];
				Rectangle new_rectangle = it->Merge(*rectangle);
				double margin_increase = new_rectangle.Perimeter() - it->Perimeter();
				if (margin_increase < min_margin_increase) {
					//choose the subtree with smaller perimeter increase
					next_node = it;
					min_margin_increase = margin_increase;
				}
				else if (margin_increase == min_margin_increase) {
					//break the tie by favoring the smaller MBR
					if (next_node->Area() > it->Area()) {
						next_node = it;
					}
				}
			}
			break;
		}
		case INS_OVERLAP: {
			double min_overlap_increase = DBL_MAX;
			for (int idx = 0; idx < tree_node->entry_num; idx++) {
				TreeNode* it = tree_nodes_[tree_node->children[idx]];
				Rectangle new_rectangle = it->Merge(*rectangle);
				double overlap_increase = 0;
				for (int idx2 = 0; idx2 < tree_node->entry_num; idx2++) {
					if (idx == idx2)continue;
					TreeNode* it2 = tree_nodes_[tree_node->children[idx2]];
					//overlap_increase += Overlap(new_rectangle, it2->bounding_box) - Overlap(it->bounding_box, it2->bounding_box);
					overlap_increase += SplitOverlap(new_rectangle, *it2) - SplitOverlap(*it, *it2);
				}
				if (overlap_increase < min_overlap_increase) {
					//choose the subtree with smaller overlap increase
					min_overlap_increase = overlap_increase;
					next_node = it;
				}
				else if (overlap_increase == min_overlap_increase) {
					//break the tie by favoring the one with smaller area increase
					//double area_increase = Area(Merge(rectangle, it->bounding_box)) - Area(it->bounding_box);
					double area_increase = it->Merge(*rectangle).Area() - it->Area();
					if (area_increase < next_node->Merge(*rectangle).Area() - next_node->Area()) {
						next_node = it;
					}

				}
			}
			break;
		}
		case INS_RANDOM: {
			int chosen_child = rand() % tree_node->entry_num;
			next_node = tree_nodes_[tree_node->children[chosen_child]];
			break;
		}
		}
		tree_node->Set(tree_node->Merge(*rectangle));
	}

	return next_node;
}

void RTree::GetInsertStates7Fill0(TreeNode *tree_node, Rectangle *rec, double *states){
	int size = 7 * TreeNode::maximum_entry;
	for(int i=0; i<size; i++){
		states[i] = 0;
	}
	Rectangle new_rectangle;
	double max_delta_area = 0;
	double max_delta_perimeter = 0;
	double max_delta_overlap = 0;
	double max_area = 0;
	double max_perimeter = 0;
	double max_overlap = 0;
	for(int i = 0; i < tree_node->entry_num; i++){
		int pos = i * 7;
		int child_id = tree_node->children[i];
		TreeNode* child = tree_nodes_[child_id];
		new_rectangle.Set(*child);
		new_rectangle.Include(*rec);
		double old_area = child->Area();
		double new_area = new_rectangle.Area();

		double old_perimeter = child->Perimeter();
		double new_perimeter = new_rectangle.Perimeter();

		double old_overlap = 0;
		double new_overlap = 0;
		for(int j=0; j<tree_node->entry_num; j++){
			if(i == j)continue;
			TreeNode* other_child = tree_nodes_[tree_node->children[j]];
			old_overlap += SplitOverlap(*child, *other_child);
			new_overlap += SplitOverlap(new_rectangle, *other_child);
		}
		if(new_area - old_area > max_delta_area){
			max_delta_area = new_area - old_area;
		}
		if(new_perimeter - old_perimeter > max_delta_perimeter){
			max_delta_perimeter = new_perimeter - old_perimeter;
		}
		if(new_overlap - old_overlap > max_delta_overlap){
			max_delta_overlap = new_overlap - old_overlap;
		}
		if(old_area > max_area){
			max_area = old_area;
		}
		if(old_perimeter > max_perimeter){
			max_perimeter = old_perimeter;
		}
		if(old_overlap > max_overlap){
			max_overlap = old_overlap;
		}
		states[pos] = child->Area();
		states[pos+1] = child->Perimeter();
		states[pos+2] = old_overlap;
		states[pos+3] = new_area - old_area;
		states[pos+4] = new_perimeter - old_perimeter;
		states[pos+5] = new_overlap - old_overlap;
		states[pos+6] = 1.0 * child->entry_num / TreeNode::maximum_entry;
	}
	for(int i=0; i<size; i++){
		switch(i%7){
			case 0:{
				states[i] = states[i] / (max_area + 0.001);
				break;
			}
			case 1:{
				states[i] = states[i] / (max_perimeter + 0.001);
				break;
			}
			case 2:{
				states[i] = states[i] / (max_overlap + 0.001);
				break;
			}
			case 3:{
				states[i] = states[i] / (max_delta_area + 0.001);
				break;
			}
			case 4:{
				states[i] = states[i] / (max_delta_perimeter + 0.001);
				break;
			}
			case 5:{
				states[i] = states[i] / (max_delta_overlap + 0.001);
				break;
			}
		}
	}
}

void RTree::GetInsertStates7(TreeNode *tree_node, Rectangle *rec, double *states){
	int size = 7 * TreeNode::maximum_entry;
	Rectangle new_rectangle;
	double max_delta_area = 0;
	double max_delta_perimeter = 0;
	double max_delta_overlap = 0;
	double max_area = 0;
	double max_perimeter = 0;
	double max_overlap = 0;
	for(int i = 0; i < tree_node->entry_num; i++){
		int pos = i * 7;
		int child_id = tree_node->children[i];
		TreeNode* child = tree_nodes_[child_id];
		new_rectangle.Set(*child);
		new_rectangle.Include(*rec);
		double old_area = child->Area();
		double new_area = new_rectangle.Area();

		double old_perimeter = child->Perimeter();
		double new_perimeter = new_rectangle.Perimeter();

		double old_overlap = 0;
		double new_overlap = 0;
		for(int j=0; j<tree_node->entry_num; j++){
			if(i == j)continue;
			TreeNode* other_child = tree_nodes_[tree_node->children[j]];
			old_overlap += SplitOverlap(*child, *other_child);
			new_overlap += SplitOverlap(new_rectangle, *other_child);
		}
		if(new_area - old_area > max_delta_area){
			max_delta_area = new_area - old_area;
		}
		if(new_perimeter - old_perimeter > max_delta_perimeter){
			max_delta_perimeter = new_perimeter - old_perimeter;
		}
		if(new_overlap - old_overlap > max_delta_overlap){
			max_delta_overlap = new_overlap - old_overlap;
		}
		if(old_area > max_area){
			max_area = old_area;
		}
		if(old_perimeter > max_perimeter){
			max_perimeter = old_perimeter;
		}
		if(old_overlap > max_overlap){
			max_overlap = old_overlap;
		}
		states[pos] = child->Area();
		states[pos+1] = child->Perimeter();
		states[pos+2] = old_overlap;
		states[pos+3] = new_area - old_area;
		states[pos+4] = new_perimeter - old_perimeter;
		states[pos+5] = new_overlap - old_overlap;
		states[pos+6] = 1.0 * child->entry_num / TreeNode::maximum_entry;
	}
	for(int i=tree_node->entry_num; i < TreeNode::maximum_entry; i++){
		int loc = (i - tree_node->entry_num) % tree_node->entry_num;
		for(int j=0; j<7; j++){
			states[i * 7 + j] = states[loc*7 +j];
		}
	}
	for(int i=0; i<size; i++){
		switch(i%7){
			case 0:{
				states[i] = states[i] / (max_area + 0.001);
				break;
			}
			case 1:{
				states[i] = states[i] / (max_perimeter + 0.001);
				break;
			}
			case 2:{
				states[i] = states[i] / (max_overlap + 0.001);
				break;
			}
			case 3:{
				states[i] = states[i] / (max_delta_area + 0.001);
				break;
			}
			case 4:{
				states[i] = states[i] / (max_delta_perimeter + 0.001);
				break;
			}
			case 5:{
				states[i] = states[i] / (max_delta_overlap + 0.001);
				break;
			}
		}
	}
}

void RTree::GetInsertStates6(TreeNode *tree_node, Rectangle *rec, double *states){
	int size = 6 * TreeNode::maximum_entry;
	Rectangle new_rectangle;
	double max_delta_area = 0;
	double max_delta_perimeter = 0;
	double max_delta_overlap = 0;
	double max_area = 0;
	double max_perimeter = 0;
	double max_overlap = 0;
	for(int i = 0; i < tree_node->entry_num; i++){
		int pos = i * 6;
		int child_id = tree_node->children[i];
		TreeNode* child = tree_nodes_[child_id];
		new_rectangle.Set(*child);
		new_rectangle.Include(*rec);
		double old_area = child->Area();
		double new_area = new_rectangle.Area();

		double old_perimeter = child->Perimeter();
		double new_perimeter = new_rectangle.Perimeter();

		double old_overlap = 0;
		double new_overlap = 0;
		for(int j=0; j<tree_node->entry_num; j++){
			if(i == j)continue;
			TreeNode* other_child = tree_nodes_[tree_node->children[j]];
			old_overlap += SplitOverlap(*child, *other_child);
			new_overlap += SplitOverlap(new_rectangle, *other_child);
		}
		if(new_area - old_area > max_delta_area){
			max_delta_area = new_area - old_area;
		}
		if(new_perimeter - old_perimeter > max_delta_perimeter){
			max_delta_perimeter = new_perimeter - old_perimeter;
		}
		if(new_overlap - old_overlap > max_delta_overlap){
			max_delta_overlap = new_overlap - old_overlap;
		}
		if(old_area > max_area){
			max_area = old_area;
		}
		if(old_perimeter > max_perimeter){
			max_perimeter = old_perimeter;
		}
		if(old_overlap > max_overlap){
			max_overlap = old_overlap;
		}
		states[pos] = child->Area();
		states[pos+1] = child->Perimeter();
		states[pos+2] = old_overlap;
		states[pos+3] = new_area - old_area;
		states[pos+4] = new_perimeter - old_perimeter;
		states[pos+5] = new_overlap - old_overlap;
	}
	for(int i=tree_node->entry_num; i < TreeNode::maximum_entry; i++){
		int loc = (i - tree_node->entry_num) % tree_node->entry_num;
		for(int j=0; j<6; j++){
			states[i * 6 + j] = states[loc*6 +j];
		}
	}
	for(int i=0; i<size; i++){
		switch(i%6){
			case 0:{
				states[i] = states[i] / (max_area + 0.001);
				break;
			}
			case 1:{
				states[i] = states[i] / (max_perimeter + 0.001);
				break;
			}
			case 2:{
				states[i] = states[i] / (max_overlap + 0.001);
				break;
			}
			case 3:{
				states[i] = states[i] / (max_delta_area + 0.001);
				break;
			}
			case 4:{
				states[i] = states[i] / (max_delta_perimeter + 0.001);
				break;
			}
			case 5:{
				states[i] = states[i] / (max_delta_overlap + 0.001);
				break;
			}
		}
	}
}

void RTree::GetInsertStates4(TreeNode *tree_node, Rectangle *rec, double *states){
	int size = 4 * TreeNode::maximum_entry;
	Rectangle new_rectangle;
	double max_delta_area = 0;
	double max_delta_perimeter = 0;
	double max_delta_overlap = 0;
	for(int i = 0; i < tree_node->entry_num; i++){
		int pos = i * 4;
		int child_id = tree_node->children[i];
		TreeNode* child = tree_nodes_[child_id];
		new_rectangle.Set(*child);
		new_rectangle.Include(*rec);
		double old_area = child->Area();
		double new_area = new_rectangle.Area();

		double old_perimeter = child->Perimeter();
		double new_perimeter = new_rectangle.Perimeter();

		double old_overlap = 0;
		double new_overlap = 0;
		for(int j=0; j<tree_node->entry_num; j++){
			if(i == j)continue;
			TreeNode* other_child = tree_nodes_[tree_node->children[j]];
			old_overlap += SplitOverlap(*child, *other_child);
			new_overlap += SplitOverlap(new_rectangle, *other_child);
		}
		states[pos] = new_area - old_area;
		states[pos+1] = new_perimeter - old_perimeter;
		states[pos+2] = new_overlap - old_overlap;
		states[pos+3] = 1.0 * child->entry_num / TreeNode::maximum_entry;

		//cout<<"state4: "<<states[pos]<<" "<<states[pos+1]<<" "<<states[pos+2]<<" "<<states[pos+3]<<endl;
		if(new_area - old_area > max_delta_area){
			max_delta_area = new_area - old_area;
		}
		if(new_perimeter - old_perimeter > max_delta_perimeter){
			max_delta_perimeter = new_perimeter - old_perimeter;
		}
		if(new_overlap - old_overlap > max_delta_overlap){
			max_delta_overlap = new_overlap - old_overlap;
		}
	}
	for(int i = tree_node->entry_num; i < TreeNode::maximum_entry; i++){
		int loc = (i - tree_node->entry_num) % tree_node->entry_num;
		for(int j=0; j<4; j++){
			states[i*4 + j] = states[loc*4 + j];
		}
	}
	for(int i=0; i<size; i++){
		switch(i%4){
			case 0:{
				states[i] = states[i] / (max_delta_area + 0.001);
				break;
			}
			case 1:{
				states[i] = states[i] / (max_delta_perimeter + 0.001);
				break;
			}
			case 2:{
				states[i] = states[i] / (max_delta_overlap + 0.001);
				break;
			}
		}
	}
}

void RTree::GetInsertStates3(TreeNode *tree_node, Rectangle *rec, double *states){
	int size = 3 * TreeNode::maximum_entry;
	Rectangle new_rectangle;
	double max_delta_area = 0;
	double max_delta_perimeter = 0;
	double max_delta_overlap = 0;
	for(int i = 0; i < tree_node->entry_num; i++){
		int pos = i * 3;
		int child_id = tree_node->children[i];
		TreeNode* child = tree_nodes_[child_id];
		new_rectangle.Set(*child);
		new_rectangle.Include(*rec);
		double old_area = child->Area();
		double new_area = new_rectangle.Area();

		double old_perimeter = child->Perimeter();
		double new_perimeter = new_rectangle.Perimeter();

		double old_overlap = 0;
		double new_overlap = 0;
		for(int j=0; j<tree_node->entry_num; j++){
			if(i == j)continue;
			TreeNode* other_child = tree_nodes_[tree_node->children[j]];
			old_overlap += SplitOverlap(*child, *other_child);
			new_overlap += SplitOverlap(new_rectangle, *other_child);
		}
		states[pos] = new_area - old_area;
		states[pos+1] = new_perimeter - old_perimeter;
		states[pos+2] = new_overlap - old_overlap;
		if(new_area - old_area > max_delta_area){
			max_delta_area = new_area - old_area;
		}
		if(new_perimeter - old_perimeter > max_delta_perimeter){
			max_delta_perimeter = new_perimeter - old_perimeter;
		}
		if(new_overlap - old_overlap > max_delta_overlap){
			max_delta_overlap = new_overlap - old_overlap;
		}
	}
	for(int i = tree_node->entry_num; i < TreeNode::maximum_entry; i++){
		int loc = (i - tree_node->entry_num) % tree_node->entry_num;
		for(int j=0; j<3; j++){
			states[i*3 + j] = states[loc*3 + j];
		}
	}
	for(int i=0; i<size; i++){
		switch(i%3){
			case 0:{
				states[i] = states[i] / (max_delta_area + 0.001);
				break;
			}
			case 1:{
				states[i] = states[i] / (max_delta_perimeter + 0.001);
				break;
			}
			case 2:{
				states[i] = states[i] / (max_delta_overlap + 0.001);
				break;
			}
		}
	}
}
void RTree::GetInsertStates(TreeNode *tree_node, Rectangle* rec, double *states){
	int size = 6 + 9 * TreeNode::maximum_entry;
	for(int i=0; i < size; i++){
		states[i] = 0;
	}
	states[0] = rec->Left();
	states[1] = rec->Bottom();
	states[2] = rec->Right() - rec->Left();
	states[3] = rec->Top() - rec->Bottom();
	states[4] = states[3] / states[2];
	states[5] = states[3] * states[2];
	Rectangle new_rectangle;
	for(int i=0; i < tree_node->entry_num; i++){
		int pos = 6 + i * 9;
		int child_id = tree_node->children[i];
		TreeNode* child = tree_nodes_[child_id];
		new_rectangle.Set(*child);
		new_rectangle.Include(*rec);
		states[pos] = child->Left();
		states[pos+1] = child->Bottom();
		states[pos+2] = child->Right() - child->Left();
		states[pos+3] = child->Top() - child->Bottom();
		states[pos+4] = states[pos+3] / states[pos+2];
		states[pos+5] = child->Area();
		states[pos+6] = new_rectangle.Area() - child->Area();
		states[pos+7] = new_rectangle.Perimeter() - child->Perimeter();
		double old_ovlp = 0;
		double new_ovlp = 0;
		for(int j=0; j<tree_node->entry_num; j++){
			if(i == j)continue;
			TreeNode* other_child = tree_nodes_[tree_node->children[j]];
			old_ovlp += SplitOverlap(*child, *other_child);
			new_ovlp += SplitOverlap(new_rectangle, *other_child);
		} 
		states[pos+8] = new_ovlp - old_ovlp;
	}
}

void RTree::GetShortSplitStates(TreeNode* tree_node, double* states) {
	int size = TreeNode::maximum_entry - 2 * TreeNode::minimum_entry + 2;
	double max_area = -DBL_MAX;
	double max_perimeter = -DBL_MAX;
	double max_overlap = -DBL_MAX;
	double min_area = DBL_MAX;
	double min_perimeter = DBL_MAX;
	double min_overlap = DBL_MAX;
	if (tree_node->is_leaf) {
		vector<Rectangle*> recs(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			recs[i] = objects_[tree_node->children[i]];
		}
		sort(recs.begin(), recs.end(), SortedByLeft);
		Rectangle prefix = MergeRange<Rectangle>(recs, 0, TreeNode::minimum_entry - 1);
		Rectangle suffix = MergeRange<Rectangle>(recs, TreeNode::maximum_entry - TreeNode::minimum_entry + 1, recs.size());
		//cout << "minimum_entry-1: " << TreeNode::minimum_entry - 1 << " maximum_entry - minimum_entry+1: " << TreeNode::maximum_entry - TreeNode::minimum_entry + 1 << endl;
		int loc = 0;
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; i++) {
			prefix.Include(*recs[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; j++) {
				remaining.Include(*recs[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
			min_area = min(min_area, min(prefix.Area(), remaining.Area()));
			min_perimeter = min(min_perimeter, min(prefix.Perimeter(), remaining.Perimeter()));
			min_overlap = min(min_overlap, SplitOverlap(prefix, remaining));
		}
	}
	else {
		vector<TreeNode*> nodes(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			nodes[i] = tree_nodes_[tree_node->children[i]];
		}
		int loc = 0;
		sort(nodes.begin(), nodes.end(), SortedByLeft);
		Rectangle prefix = MergeRange<TreeNode>(nodes, 0, TreeNode::minimum_entry - 1);
		Rectangle suffix = MergeRange<TreeNode>(nodes, TreeNode::maximum_entry - TreeNode::minimum_entry + 1, nodes.size());
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; i++) {
			prefix.Include(*nodes[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; j++) {
				remaining.Include(*nodes[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
			min_area = min(min_area, min(prefix.Area(), remaining.Area()));
			min_perimeter = min(min_perimeter, min(prefix.Perimeter(), remaining.Perimeter()));
			min_overlap = min(min_overlap, SplitOverlap(prefix, remaining));
		}
	}
	for (int i = 0; i < 5 * size; i++) {
		switch (i % 5) {
		case 0: {
			states[i] = (states[i] - min_area) / (max_area - min_area + 0.01);
			break;
		}
		case 1: {
			states[i] = (states[i] - min_area) / (max_area - min_area + 0.01);
			break;
		}
		case 2: {
			states[i] = (states[i] - min_perimeter) / (max_perimeter - min_perimeter + 0.01);
			break;
		}
		case 3: {
			states[i] = (states[i] - min_perimeter) / (max_perimeter - min_perimeter + 0.01);
			break;
		}
		case 4: {
			states[i] = (states[i] - min_overlap) / (max_overlap - min_overlap + 0.01);
			break;
		}
		}

	}
}

void RTree::GetSplitStates(TreeNode* tree_node, double* states) {
	int size_per_dim = TreeNode::maximum_entry - 2 * TreeNode::minimum_entry + 2;
	//cout << "size_per_dim " << size_per_dim << endl;
	double max_area = -DBL_MAX;
	double max_perimeter = -DBL_MAX;
	double max_overlap = -DBL_MAX;
	double min_area = DBL_MAX;
	double min_perimeter = DBL_MAX;
	double min_overlap = DBL_MAX;
	if (tree_node->is_leaf) {
		vector<Rectangle*> recs(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			recs[i] = objects_[tree_node->children[i]];
		}
		sort(recs.begin(), recs.end(), SortedByLeft);
		Rectangle prefix = MergeRange<Rectangle>(recs, 0, TreeNode::minimum_entry - 1);
		Rectangle suffix = MergeRange<Rectangle>(recs, TreeNode::maximum_entry - TreeNode::minimum_entry+1, recs.size());
		//cout << "minimum_entry-1: " << TreeNode::minimum_entry - 1 << " maximum_entry - minimum_entry+1: " << TreeNode::maximum_entry - TreeNode::minimum_entry + 1 << endl;
		int loc = 0;
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++) {
			prefix.Include(*recs[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry+1; j++) {
				remaining.Include(*recs[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
			min_area = min(min_area, min(prefix.Area(), remaining.Area()));
			min_perimeter = min(min_perimeter, min(prefix.Perimeter(), remaining.Perimeter()));
			min_overlap = min(min_overlap, SplitOverlap(prefix, remaining));
		
		}
		
		sort(recs.begin(), recs.end(), SortedByRight);
		prefix = MergeRange<Rectangle>(recs, 0, TreeNode::minimum_entry - 1);
		suffix = MergeRange<Rectangle>(recs, TreeNode::maximum_entry - TreeNode::minimum_entry+1, recs.size());
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++) {
			prefix.Include(*recs[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry+1; j++) {
				remaining.Include(*recs[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
			min_area = min(min_area, min(prefix.Area(), remaining.Area()));
			min_perimeter = min(min_perimeter, min(prefix.Perimeter(), remaining.Perimeter()));
			min_overlap = min(min_overlap, SplitOverlap(prefix, remaining));
		}
		sort(recs.begin(), recs.end(), SortedByBottom);
		prefix = MergeRange<Rectangle>(recs, 0, TreeNode::minimum_entry - 1);
		suffix = MergeRange<Rectangle>(recs, TreeNode::maximum_entry - TreeNode::minimum_entry+1, recs.size());
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++) {
			prefix.Include(*recs[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry+1; j++) {
				remaining.Include(*recs[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
			min_area = min(min_area, min(prefix.Area(), remaining.Area()));
			min_perimeter = min(min_perimeter, min(prefix.Perimeter(), remaining.Perimeter()));
			min_overlap = min(min_overlap, SplitOverlap(prefix, remaining));
		}
		sort(recs.begin(), recs.end(), SortedByTop);
		prefix = MergeRange<Rectangle>(recs, 0, TreeNode::minimum_entry - 1);
		suffix = MergeRange<Rectangle>(recs, TreeNode::maximum_entry - TreeNode::minimum_entry+1, recs.size());
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++) {
			prefix.Include(*recs[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry+1; j++) {
				remaining.Include(*recs[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
			min_area = min(min_area, min(prefix.Area(), remaining.Area()));
			min_perimeter = min(min_perimeter, min(prefix.Perimeter(), remaining.Perimeter()));
			min_overlap = min(min_overlap, SplitOverlap(prefix, remaining));
		}
		
	}
	else {
		vector<TreeNode*> nodes(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			nodes[i] = tree_nodes_[tree_node->children[i]];
		}
		int loc = 0;
		sort(nodes.begin(), nodes.end(), SortedByLeft);
		Rectangle prefix = MergeRange<TreeNode>(nodes, 0, TreeNode::minimum_entry - 1);
		Rectangle suffix = MergeRange<TreeNode>(nodes, TreeNode::maximum_entry - TreeNode::minimum_entry+1, nodes.size());
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++) {
			prefix.Include(*nodes[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry+1; j++) {
				remaining.Include(*nodes[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
			min_area = min(min_area, min(prefix.Area(), remaining.Area()));
			min_perimeter = min(min_perimeter, min(prefix.Perimeter(), remaining.Perimeter()));
			min_overlap = min(min_overlap, SplitOverlap(prefix, remaining));
		}
		sort(nodes.begin(), nodes.end(), SortedByRight);
		prefix = MergeRange<TreeNode>(nodes, 0, TreeNode::minimum_entry - 1);
		suffix = MergeRange<TreeNode>(nodes, TreeNode::maximum_entry - TreeNode::minimum_entry+1, nodes.size());
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++) {
			prefix.Include(*nodes[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry+1; j++) {
				remaining.Include(*nodes[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
			min_area = min(min_area, min(prefix.Area(), remaining.Area()));
			min_perimeter = min(min_perimeter, min(prefix.Perimeter(), remaining.Perimeter()));
			min_overlap = min(min_overlap, SplitOverlap(prefix, remaining));
		}
		sort(nodes.begin(), nodes.end(), SortedByBottom);
		prefix = MergeRange<TreeNode>(nodes, 0, TreeNode::minimum_entry - 1);
		suffix = MergeRange<TreeNode>(nodes, TreeNode::maximum_entry - TreeNode::minimum_entry+1, nodes.size());
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++) {
			prefix.Include(*nodes[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry+1; j++) {
				remaining.Include(*nodes[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
			min_area = min(min_area, min(prefix.Area(), remaining.Area()));
			min_perimeter = min(min_perimeter, min(prefix.Perimeter(), remaining.Perimeter()));
			min_overlap = min(min_overlap, SplitOverlap(prefix, remaining));
		}
		sort(nodes.begin(), nodes.end(), SortedByTop);
		prefix = MergeRange<TreeNode>(nodes, 0, TreeNode::minimum_entry - 1);
		suffix = MergeRange<TreeNode>(nodes, TreeNode::maximum_entry - TreeNode::minimum_entry+1, nodes.size());
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++) {
			prefix.Include(*nodes[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry+1; j++) {
				remaining.Include(*nodes[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
			min_area = min(min_area, min(prefix.Area(), remaining.Area()));
			min_perimeter = min(min_perimeter, min(prefix.Perimeter(), remaining.Perimeter()));
			min_overlap = min(min_overlap, SplitOverlap(prefix, remaining));
		}
	}
	for (int i = 0; i < 240; i++) {
		switch (i % 5) {
		case 0: {
			states[i] = (states[i] - min_area) / (max_area - min_area + 0.01);
			break;
		}
		case 1: {
			states[i] = (states[i] - min_area) / (max_area - min_area + 0.01);
			break;
		}
		case 2: {
			states[i] = (states[i] - min_perimeter)/ (max_perimeter - min_perimeter + 0.01);
			break;
		}
		case 3: {
			states[i] = (states[i] - min_perimeter) / (max_perimeter - min_perimeter + 0.01);
			break;
		}
		case 4: {
			states[i] = (states[i] - min_overlap)/ (max_overlap - min_overlap + 0.01);
			break;
		}

		}

	}
}

void RTree::SplitAREACost(TreeNode* tree_node, vector<double>& values, Rectangle& bounding_box1, Rectangle& bounding_box2) {
	double minimum_area = DBL_MAX;
	double minimum_overlap = DBL_MAX;
	//choose the split with the minimum total area, break the tie by preferring the split with smaller overlap
	if (tree_node->is_leaf) {
		vector<Rectangle*> recs(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			recs[i] = objects_[tree_node->children[i]];
		}
		//sort by left
		sort(recs.begin(), recs.end(), SortedByLeft);
		Rectangle rec1;
		Rectangle rec2;
		int split = FindMinimumSplit<Rectangle>(recs, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}

		//sort by right
		sort(recs.begin(), recs.end(), SortedByRight);
		split = FindMinimumSplit<Rectangle>(recs, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}

		//sort by bottom
		sort(recs.begin(), recs.end(), SortedByBottom);
		split = FindMinimumSplit<Rectangle>(recs, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}

		//sort by top
		sort(recs.begin(), recs.end(), SortedByTop);
		split = FindMinimumSplit<Rectangle>(recs, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
	}
	else {
		vector<TreeNode*> child_nodes(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			child_nodes[i] = tree_nodes_[tree_node->children[i]];
		}
		//sort by left
		sort(child_nodes.begin(), child_nodes.begin() + tree_node->entry_num, SortedByLeft);
		Rectangle rec1;
		Rectangle rec2;
		int split = FindMinimumSplit<TreeNode>(child_nodes, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by right
		sort(child_nodes.begin(), child_nodes.end(), SortedByRight);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by bottom
		sort(child_nodes.begin(), child_nodes.end(), SortedByBottom);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by top
		sort(child_nodes.begin(), child_nodes.end(), SortedByTop);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}

	}
	if (values.size() != 5) {
		values.resize(5);
	}
	values[0] = bounding_box1.Perimeter() / tree_node->Perimeter();
	values[1] = bounding_box2.Perimeter() / tree_node->Perimeter();
	values[2] = bounding_box1.Area() / tree_node->Area();
	values[3] = bounding_box2.Area() / tree_node->Area();
	values[4] = SplitOverlap(bounding_box1, bounding_box2) / tree_node->Area();
}


int RTree::GetMinAreaContainingChild(TreeNode *tree_node, Rectangle *rec){
	double min_area = DBL_MAX;
	int child_id = -1;
	for(int i=0; i<tree_node->entry_num; i++){
		TreeNode* child = tree_nodes_[tree_node->children[i]];
		if(child->Contains(rec)){
			if(child->Area() < min_area){
				min_area = child->Area();
				child_id = i;
			}
		}
	}
	return child_id;
}

int RTree::GetMinAreaEnlargementChild(TreeNode *tree_node, Rectangle *rec){
	double min_area_enlargement = DBL_MAX;
	int child_id = 0;
	Rectangle new_rectangle;
	for(int i=0; i<tree_node->entry_num; i++){
		int id = tree_node->children[i];
		TreeNode* child = tree_nodes_[id];
		new_rectangle.Set(*child);
		new_rectangle.Include(*rec);
		double area_enlargement = new_rectangle.Area() - child->Area();
		if(area_enlargement < min_area_enlargement){
			min_area_enlargement = area_enlargement;
			child_id = i;
		}
	}
	return child_id;
}

int RTree::GetMinMarginIncrementChild(TreeNode *tree_node, Rectangle *rec){
	double min_margin_increase = DBL_MAX;
	int child_id = 0;
	Rectangle new_rectangle;
	for(int i=0; i<tree_node->entry_num; i++){
		int id = tree_node->children[i];
		TreeNode* child = tree_nodes_[id];
		new_rectangle.Set(*child);
		new_rectangle.Include(*rec);
		double margin_increase = new_rectangle.Perimeter() - child->Perimeter();
		if(margin_increase < min_margin_increase){
			min_margin_increase = margin_increase;
			child_id = i;
		}
	}
	return child_id;
}

int RTree::GetMinOverlapIncrementChild(TreeNode *tree_node, Rectangle *rec){
	double min_overlap_increase = DBL_MAX;
	int child_id = 0;
	Rectangle new_rectangle;
	for(int i=0; i<tree_node->entry_num; i++){
		int id = tree_node->children[i];
		TreeNode* child = tree_nodes_[id];
		new_rectangle.Set(*child);
		new_rectangle.Include(*rec);
		double old_overlap = 0;
		double new_overlap = 0;
		for(int j=0; j<tree_node->entry_num; j++){
			if(i == j)continue;
			TreeNode* other_child = tree_nodes_[tree_node->children[j]];
			old_overlap += SplitOverlap(*child, *other_child);
			new_overlap += SplitOverlap(new_rectangle, *other_child);
		}
		if(new_overlap - old_overlap < min_overlap_increase){
			min_overlap_increase = new_overlap - old_overlap;
			child_id = i;
		}
	}
	return child_id;
}

void RTree::SplitMARGINCost(TreeNode* tree_node, vector<double>& values, Rectangle& bounding_box1, Rectangle& bounding_box2) {
	double minimum_perimeter = DBL_MAX;
	double minimum_overlap = DBL_MAX;
	//choose the split with the minimum total perimeter, break the tie by preferring the split with smaller overlap

	if (tree_node->is_leaf) {
		vector<Rectangle*> recs(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			recs[i] = objects_[tree_node->children[i]];
		}
		Rectangle rec1;
		Rectangle rec2;
		//sort by left
		sort(recs.begin(), recs.end(), SortedByLeft);
		int split = FindMinimumSplit<Rectangle>(recs, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by right
		sort(recs.begin(), recs.end(), SortedByRight);
		split = FindMinimumSplit<Rectangle>(recs, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by bottom 
		sort(recs.begin(), recs.end(), SortedByBottom);
		split = FindMinimumSplit<Rectangle>(recs, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by top
		sort(recs.begin(), recs.end(), SortedByTop);
		split = FindMinimumSplit<Rectangle>(recs, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
	}
	else {
		vector<TreeNode*> child_nodes(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			child_nodes[i] = tree_nodes_[tree_node->children[i]];
		}
		Rectangle rec1;
		Rectangle rec2;
		//sort by left
		sort(child_nodes.begin(), child_nodes.end(), SortedByLeft);
		int split = FindMinimumSplit<TreeNode>(child_nodes, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by right
		sort(child_nodes.begin(), child_nodes.end(), SortedByRight);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by bottom
		sort(child_nodes.begin(), child_nodes.end(), SortedByBottom);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by top
		sort(child_nodes.begin(), child_nodes.end(), SortedByTop);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
	}
	if (values.size() != 5) {
		values.resize(5);
	}
	values[0] = bounding_box1.Perimeter() / tree_node->Perimeter();
	values[1] = bounding_box2.Perimeter() / tree_node->Perimeter();
	values[2] = bounding_box1.Area() / tree_node->Area();
	values[3] = bounding_box2.Area() / tree_node->Area();
	values[4] = SplitOverlap(bounding_box1, bounding_box2) / tree_node->Area();
}

void RTree::PrepareSplitLocations(TreeNode *tree_node){
	if(split_locations.size() == 0){
		split_locations.resize((TreeNode::maximum_entry - 2 * TreeNode::minimum_entry + 2) * 2);
	}
	if(tree_node->is_leaf){
		vector<Rectangle*> recs(tree_node->entry_num);
		for(int i=0; i<tree_node->entry_num; i++){
			int obj_id = tree_node->children[i];
			recs[i] = objects_[obj_id];
		}
		sort(recs.begin(), recs.end(), SortedByLeft);
		Rectangle prefix = MergeRange<Rectangle>(recs, 0, TreeNode::minimum_entry - 1);
		Rectangle suffix = MergeRange<Rectangle>(recs, TreeNode::maximum_entry - TreeNode::minimum_entry + 1, recs.size());
		int loc = 0;
		for(int i=TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; i++){
			prefix.Include(*recs[i]);
			Rectangle remaining(suffix);
			for(int j=i+1; j<TreeNode::maximum_entry - TreeNode::minimum_entry + 1; j++){
				remaining.Include(*recs[j]);
			}
			split_locations[loc].perimeter1 = max(prefix.Perimeter(), remaining.Perimeter());
			split_locations[loc].perimeter2 = min(prefix.Perimeter(), remaining.Perimeter());
			split_locations[loc].area1 = max(prefix.Area(), remaining.Area());
			split_locations[loc].area2 = min(prefix.Area(), remaining.Area());
			split_locations[loc].overlap = SplitOverlap(prefix, remaining);
			split_locations[loc].location = i;
			split_locations[loc].dimension = 0;
			loc += 1;
		}
		sort(recs.begin(), recs.end(), SortedByBottom);
		prefix = MergeRange<Rectangle>(recs, 0, TreeNode::minimum_entry - 1);
		suffix = MergeRange<Rectangle>(recs, TreeNode::maximum_entry - TreeNode::minimum_entry + 1, recs.size());
		for(int i=TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; i++){
			prefix.Include(*recs[i]);
			Rectangle remaining(suffix);
			for(int j=i+1; j < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; j++){
				remaining.Include(*recs[j]);
			}
			split_locations[loc].perimeter1 = max(prefix.Perimeter(), remaining.Perimeter());
			split_locations[loc].perimeter2 = min(prefix.Perimeter(), remaining.Perimeter());
			split_locations[loc].area1 = max(prefix.Area(), remaining.Area());
			split_locations[loc].area2 = min(prefix.Area(), remaining.Area());
			split_locations[loc].overlap = SplitOverlap(prefix, remaining);
			split_locations[loc].location = i;
			split_locations[loc].dimension = 1;
			loc += 1;
		}
	}
	else{
		vector<TreeNode*> nodes(tree_node->entry_num);
		for(int i=0; i<tree_node->entry_num; i++){
			nodes[i] = tree_nodes_[tree_node->children[i]];
		}
		int loc = 0;
		sort(nodes.begin(), nodes.end(), SortedByLeft);
		Rectangle prefix = MergeRange<TreeNode>(nodes, 0, TreeNode::minimum_entry - 1);
		Rectangle suffix = MergeRange<TreeNode>(nodes, TreeNode::maximum_entry - TreeNode::minimum_entry+1, nodes.size());
		for(int i=TreeNode::minimum_entry - 1; i<TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++){
			prefix.Include(*nodes[i]);
			Rectangle remaining(suffix);
			for(int j=i+1; j<TreeNode::maximum_entry - TreeNode::minimum_entry + 1; j++){
				remaining.Include(*nodes[j]);
			}
			split_locations[loc].perimeter1 = max(prefix.Perimeter(), remaining.Perimeter());
			split_locations[loc].perimeter2 = min(prefix.Perimeter(), remaining.Perimeter());
			split_locations[loc].area1 = max(prefix.Area(), remaining.Area());
			split_locations[loc].area2 = min(prefix.Area(), remaining.Area());
			split_locations[loc].overlap = SplitOverlap(prefix, remaining);
			split_locations[loc].location = i;
			split_locations[loc].dimension = 0;
			loc += 1;
		}
		sort(nodes.begin(), nodes.end(), SortedByBottom);
		prefix = MergeRange<TreeNode>(nodes, 0, TreeNode::minimum_entry - 1);
		suffix = MergeRange<TreeNode>(nodes, TreeNode::maximum_entry - TreeNode::minimum_entry+1, nodes.size());
		for(int i=TreeNode::minimum_entry - 1; i<TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++){
			prefix.Include(*nodes[i]);
			Rectangle remaining(suffix);
			for(int j=i+1; j<TreeNode::maximum_entry - TreeNode::minimum_entry + 1; j++){
				remaining.Include(*nodes[j]);
			}
			split_locations[loc].perimeter1 = max(prefix.Perimeter(), remaining.Perimeter());
			split_locations[loc].perimeter2 = min(prefix.Perimeter(), remaining.Perimeter());
			split_locations[loc].area1 = max(prefix.Area(), remaining.Area());
			split_locations[loc].area2 = min(prefix.Area(), remaining.Area());
			split_locations[loc].overlap = SplitOverlap(prefix, remaining);
			split_locations[loc].location = i;
			split_locations[loc].dimension = 1;
			loc += 1;
		}
	}
}

void RTree::SortSplitLocByPerimeter(TreeNode* tree_node){
	//Compute the perimeter for every split location, and sort them.
	if(tree_node->is_leaf){
		vector<Rectangle*> recs(tree_node->entry_num);
		for(int i=0; i<tree_node->entry_num; i++){
			recs[i] = objects_[tree_node->children[i]];	
		}
		sort(recs.begin(), recs.end(), SortedByLeft);
		Rectangle prefix = MergeRange<Rectangle>(recs, 0, TreeNode::minimum_entry - 1);
		Rectangle suffix = MergeRange<Rectangle>(recs, TreeNode::maximum_entry-TreeNode::minimum_entry+1, recs.size());
		int loc = 0;
		for(int i=TreeNode::minimum_entry - 1; i<TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++){
			prefix.Include(*recs[i]);
			Rectangle remaining(suffix);
			for(int j=i+1; j<TreeNode::maximum_entry - TreeNode::minimum_entry + 1; j++){
				remaining.Include(*recs[j]);
			}
			sorted_split_loc[loc].first = prefix.Perimeter() + remaining.Perimeter();
			sorted_split_loc[loc].second = make_pair(true, i);
			loc += 1;
		}
		sort(recs.begin(), recs.end(), SortedByBottom);
		prefix = MergeRange<Rectangle>(recs, 0, TreeNode::minimum_entry - 1);
		suffix = MergeRange<Rectangle>(recs, TreeNode::maximum_entry - TreeNode::minimum_entry + 1, recs.size());
		for(int i=TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; i++){
			prefix.Include(*recs[i]);
			Rectangle remaining(suffix);
			for(int j=i+1; j < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; j++){
				remaining.Include(*recs[j]);
			}
			sorted_split_loc[loc].first = prefix.Perimeter() + remaining.Perimeter();
			sorted_split_loc[loc].second = make_pair(false, i);
			loc += 1;
		}
	}
	else{
		vector<TreeNode*> nodes(tree_node->entry_num);
		for(int i=0; i<tree_node->entry_num; i++){
			nodes[i] = tree_nodes_[tree_node->children[i]];
		}
		int loc = 0;
		sort(nodes.begin(), nodes.end(), SortedByLeft);
		Rectangle prefix = MergeRange<TreeNode>(nodes, 0, TreeNode::minimum_entry - 1);
		Rectangle suffix = MergeRange<TreeNode>(nodes, TreeNode::maximum_entry - TreeNode::minimum_entry+1, nodes.size());
		for(int i=TreeNode::minimum_entry - 1; i<TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++){
			prefix.Include(*nodes[i]);
			Rectangle remaining(suffix);
			for(int j=i+1; j<TreeNode::maximum_entry - TreeNode::minimum_entry + 1; j++){
				remaining.Include(*nodes[j]);
			}
			sorted_split_loc[loc].first = prefix.Perimeter() + remaining.Perimeter();
			sorted_split_loc[loc].second = make_pair(true, i);
			loc += 1;
		}
		sort(nodes.begin(), nodes.end(), SortedByBottom);
		prefix = MergeRange<TreeNode>(nodes, 0, TreeNode::minimum_entry - 1);
		suffix = MergeRange<TreeNode>(nodes, TreeNode::maximum_entry - TreeNode::minimum_entry + 1, nodes.size());
		for(int i=TreeNode::minimum_entry - 1; i<TreeNode::maximum_entry - TreeNode::minimum_entry +1; i++){
			prefix.Include(*nodes[i]);
			Rectangle remaining(suffix);
			for(int j=i+1; j<TreeNode::maximum_entry - TreeNode::minimum_entry + 1; j++){
				remaining.Include(*nodes[j]);
			}
			sorted_split_loc[loc].first = prefix.Perimeter() + remaining.Perimeter();
			sorted_split_loc[loc].second = make_pair(false, i);
			loc += 1;
		}
	}
	sort(sorted_split_loc.begin(), sorted_split_loc.end());
//	for(int i=0; i<sorted_split_loc.size(); i++){
//		cout<<"("<<sorted_split_loc[i].first<<", "<<sorted_split_loc[i].second<<") ";
//	}
//	cout<<endl;
//	getchar();
}

void RTree::SortChildrenByArea(TreeNode *tree_node){
	tmp_sorted_children.clear();
	if(!tree_node->is_leaf){
		tmp_sorted_children.resize(tree_node->entry_num);
		for(int i=0; i<tree_node->entry_num; i++){
			tmp_sorted_children[i] = tree_nodes_[tree_node->children[i]];
		}
		sort(tmp_sorted_children.begin(), tmp_sorted_children.end(), CompareByArea);
	}
}

void RTree::SortChildrenByMarginArea(TreeNode *tree_node, Rectangle* rec){
	vector<pair<double, int> > margin_area_children;
	Rectangle new_rectangle;
	tmp_sorted_children.clear();
	if(!tree_node->is_leaf){
		margin_area_children.resize(tree_node->entry_num);
		tmp_sorted_children.resize(tree_node->entry_num);
		for(int i=0; i<tree_node->entry_num; i++){
			int node_id = tree_node->children[i];
			TreeNode* node = tree_nodes_[node_id];
			new_rectangle.Set(*node);
			new_rectangle.Include(*rec);
			margin_area_children[i].first = new_rectangle.Area() - node->Area();
			margin_area_children[i].second = node_id; 
		}
		sort(margin_area_children.begin(), margin_area_children.end());
		for(int i=0; i<tree_node->entry_num; i++){
			tmp_sorted_children[i] = tree_nodes_[margin_area_children[i].second];
		}
	}
}

int RTree::GetNumberOfEnlargedChildren(TreeNode *tree_node, Rectangle *rec){
	int enlarged_children_num = 0;
	for(int i=0; i<tree_node->entry_num; i++){
		TreeNode* child = tree_nodes_[tree_node->children[i]];
		if(!child->Contains(rec)){
			enlarged_children_num += 1;
		}
	}
	return enlarged_children_num;
}

void RTree::GetSortedSplitStates(TreeNode* tree_node, double* states, int topk){
	double max_area = -DBL_MAX;
	double min_area = DBL_MAX;
	double max_perimeter = -DBL_MAX;
	double min_perimeter = DBL_MAX;
	double max_overlap = -DBL_MAX;
	double min_overlap = DBL_MAX;
	if(tree_node->is_leaf){
		vector<Rectangle*> recs_h(tree_node->entry_num);
		vector<Rectangle*> recs_v(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			recs_h[i] = objects_[tree_node->children[i]];
			recs_v[i] = objects_[tree_node->children[i]];
		}
		sort(recs_v.begin(), recs_v.end(), SortedByBottom);
		sort(recs_h.begin(), recs_h.end(), SortedByLeft);
		Rectangle prefix_h = MergeRange<Rectangle>(recs_h, 0, TreeNode::minimum_entry - 1);
		Rectangle suffix_h = MergeRange<Rectangle>(recs_h, TreeNode::maximum_entry - TreeNode::minimum_entry + 1, recs_h.size());
		Rectangle prefix_v = MergeRange<Rectangle>(recs_v, 0, TreeNode::minimum_entry - 1);
		Rectangle suffix_v = MergeRange<Rectangle>(recs_v, TreeNode::maximum_entry - TreeNode::minimum_entry + 1, recs_v.size());
		//Rectangle prefix = MergeRange<Rectangle>(recs, 0, TreeNode::minimum_entry - 1);
		//Rectangle suffix = MergeRange<Rectangle>(recs, TreeNode::maximum_entry - TreeNode::minimum_entry + 1, recs.size());
		Rectangle head_part;
		Rectangle end_part;
		for(int i=0; i<topk; i++){
			bool is_horizontal = sorted_split_loc[i].second.first;
			int split_loc = sorted_split_loc[i].second.second;
			if(is_horizontal){
				head_part.Set(prefix_h);
				end_part.Set(suffix_h);
				for(int j=TreeNode::minimum_entry - 1; j <= split_loc; j++){
					head_part.Include(*recs_h[j]);
				}
				for(int j=split_loc+1; j<TreeNode::maximum_entry - TreeNode::minimum_entry + 1; j++){
					end_part.Include(*recs_h[j]);
				}
			}
			else{
				head_part.Set(prefix_v);
				end_part.Set(suffix_v);
				for(int j=TreeNode::minimum_entry - 1; j<=split_loc; j++){
					head_part.Include(*recs_v[j]);
				}
				for(int j=split_loc+1; j<TreeNode::maximum_entry - TreeNode::minimum_entry + 1; j++){
					end_part.Include(*recs_v[j]);
				}
			}
			//head_part.Set(prefix);
			//end_part.Set(suffix);
		//	for(int j=TreeNode::minimum_entry-1; j<=sorted_split_loc[i].second; j++){
		//		head_part.Include(*recs[j]);
		//	}
		//	for(int j=sorted_split_loc[i].second+1; j < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; j++){
		//		end_part.Include(*recs[j]);
		//	}
			states[i * 5] = max(head_part.Area(), end_part.Area());
			states[i * 5 + 1] = min(head_part.Area(), end_part.Area());
			states[i * 5 + 2] = max(head_part.Perimeter(), end_part.Perimeter());
			states[i * 5 + 3] = min(head_part.Perimeter(), end_part.Perimeter());
			states[i * 5 + 4] = SplitOverlap(head_part, end_part);
			max_area = max(max_area, max(head_part.Area(), end_part.Area()));
			min_area = min(min_area, min(head_part.Area(), end_part.Area()));
			max_perimeter = max(max_perimeter, max(head_part.Perimeter(), end_part.Perimeter()));
			min_perimeter = min(min_perimeter, min(head_part.Perimeter(), end_part.Perimeter()));
			max_overlap = max(max_overlap, states[i*5 + 4]);
			min_overlap = min(min_overlap, states[i*5 + 4]);
		}
	}
	for(int i=0 ; i < topk * 5; i++){
		switch(i%5){
			case 0:{
				states[i] = (states[i] - min_area + 1) / (max_area - min_area+ 1);
				break;
			}
			case 1:{
				states[i] = (states[i] - min_area + 1) / (max_area - min_area+ 1);
				break;
			}
			case 2:{
				states[i] = (states[i] - min_perimeter + 1)/(max_perimeter - min_perimeter+ 1);
				break;
			}
			case 3:{
				states[i] = (states[i] - min_perimeter + 1) / (max_perimeter - min_perimeter+ 1);
				break;
			}
			case 4:{
				states[i] = (states[i] - min_overlap + 1)/(max_overlap - min_overlap + 1);
				break;
			}
		}
	}
//	cout<<endl;
//	cout<<"max area: "<<max_area<<" min area: "<<min_area<<endl;
//	cout<<"max perimeter: "<<max_perimeter<<" min perimeter: "<<min_perimeter<<endl;
//	cout<<"max_overlap: "<<max_overlap<<" min overlap: "<<min_overlap<<endl;

}

void RTree::GetSortedInsertStates(TreeNode *tree_node, Rectangle *rec, double *states, int topk, INSERT_STATE_TYPE state_type){
	int dimension = topk * 4;
	for(int i=0; i<dimension; i++){
		states[i] = 0;
	}
	Rectangle new_rectangle;
	double area_normalize = -DBL_MAX;
	double margin_normalize = -DBL_MAX;
	double overlap_normalize = - DBL_MAX;
	double occupancy_normalize = -DBL_MAX;
	for(int i=0; i<tmp_sorted_children.size(); i++){
		if(i == topk)break;
		TreeNode* node = tmp_sorted_children[i];
		int loc = i * 4;
		switch(state_type){
			case RL_FOR_CONTAINING_CHILDREN:{
				states[loc] = node->Area();
				states[loc+1] = node->Perimeter();
				states[loc+2] = 0;
				for(int j=0; j<tmp_sorted_children.size(); j++){
					if(i==j)continue;
					TreeNode* another = tmp_sorted_children[j];
					states[loc+2] += SplitOverlap(*node, *another);
				}
				states[loc+3] = 1.0 * node->entry_num / TreeNode::maximum_entry;
				break;
			}
			case RL_FOR_ENLARGED_CHILDREN:{
				new_rectangle.Set(*node);
				new_rectangle.Include(*rec);
				states[loc] = new_rectangle.Area() - node->Area();
				states[loc+1] = new_rectangle.Perimeter() - node->Perimeter();
				double old_overlap = 0, new_overlap = 0;
				for(int j=0; j<tmp_sorted_children.size(); j++){
					if(i == j)continue;
					TreeNode* another = tmp_sorted_children[j];
					old_overlap += SplitOverlap(*node, *another);
					new_overlap += SplitOverlap(new_rectangle, *another);
				}
				states[loc+2] = new_overlap - old_overlap;
				states[loc+3] = 1.0 * node->entry_num / TreeNode::maximum_entry;
				break;
			}
		}
		if(states[loc] > area_normalize){
			area_normalize = states[loc];
		}
		if(states[loc+1] > margin_normalize){
			margin_normalize = states[loc+1];
		}
		if(states[loc+2] > overlap_normalize){
			overlap_normalize = states[loc+2];
		}
		if(states[loc+3] > occupancy_normalize){
			occupancy_normalize = states[loc+3];
		}
	}
	if(tmp_sorted_children.size() < topk){
		for(int i = tree_node->entry_num; i < topk; i++){
			int loc = i * 4;
			int copy_loc = (i % tree_node->entry_num) * 4;
			for(int j=0; j<4; j++){
				states[loc+j] = states[copy_loc + j];
			}
		}
	}
	for(int i=0; i<dimension; i++){
		switch(i%4){
			case 0:{
				states[i] = states[i] / (area_normalize + 0.001);
				break;
			}
			case 1:{
				states[i] = states[i] / (margin_normalize + 0.001);
				break;
			}
			case 2:{
				states[i] = states[i] / (overlap_normalize + 0.001);
				break;
			}
			case 3:{
				states[i] = states[i] / (occupancy_normalize + 0.001);
				break;
			}
		}
	}
}

void RTree::SplitOVERLAPCost(TreeNode* tree_node, vector<double>& values, Rectangle& bounding_box1, Rectangle& bounding_box2) {
	double minimum_overlap = DBL_MAX;
	double minimum_area = DBL_MAX;
	//choose the split with the minimum overlap, break the tie by preferring the split with smaller total area
	if (tree_node->is_leaf) {
		vector<Rectangle*> recs(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			recs[i] = objects_[tree_node->children[i]];
		}
		Rectangle rec1;
		Rectangle rec2;
		//sort by left
		sort(recs.begin(), recs.end(), SortedByLeft);
		int split = FindMinimumSplit<Rectangle>(recs, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by right
		sort(recs.begin(), recs.end(), SortedByRight);
		split = FindMinimumSplit<Rectangle>(recs, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by bottom
		sort(recs.begin(), recs.end(), SortedByBottom);
		split = FindMinimumSplit<Rectangle>(recs, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by top
		sort(recs.begin(), recs.end(), SortedByTop);
		split = FindMinimumSplit<Rectangle>(recs, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
	}
	else {
		vector<TreeNode*> child_nodes(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			child_nodes[i] = tree_nodes_[tree_node->children[i]];
		}
		Rectangle rec1;
		Rectangle rec2;
		//sort by left
		sort(child_nodes.begin(), child_nodes.end(), SortedByLeft);
		int split = FindMinimumSplit<TreeNode>(child_nodes, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by right
		sort(child_nodes.begin(), child_nodes.end(), SortedByRight);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by bottom
		sort(child_nodes.begin(), child_nodes.end(), SortedByBottom);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by top
		sort(child_nodes.begin(), child_nodes.end(), SortedByTop);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
	}
	if (values.size() != 5) {
		values.resize(5);
	}
	values[0] = bounding_box1.Perimeter() / tree_node->Perimeter();
	values[1] = bounding_box2.Perimeter() / tree_node->Perimeter();
	values[2] = bounding_box1.Area() / tree_node->Area();
	values[3] = bounding_box2.Area() / tree_node->Area();
	values[4] = SplitOverlap(bounding_box1, bounding_box2) / tree_node->Area();
}

void RTree::SplitGREENECost(TreeNode* tree_node, vector<double>& values, Rectangle& bounding_box1, Rectangle& bounding_box2) {
	int seed1 = -1;
	int seed2 = -1;
	double max_waste = - DBL_MAX;
	for (int i = 0; i < tree_node->entry_num - 1; i++) {
		for (int j = i + 1; j < tree_node->entry_num; j++) {
			double waste = 0;
			int id1 = tree_node->children[i];
			int id2 = tree_node->children[j];
			if (tree_node->is_leaf) {
				/*if(debug){
					cout << "id1 " << id1 << " id2 " << id2 << endl;
					cout << objects_[id1]->Area() << " " << objects_[id2]->Area() << " " << objects_[id1]->Merge(*objects_[id2]).Area() << endl;
					cout << objects_[id1]->left_ << " " << objects_[id1]->right_ << " " << objects_[id1]->bottom_ << " " << objects_[id1]->top_ << endl;
					cout << objects_[id2]->left_ << " " << objects_[id2]->right_ << " " << objects_[id2]->bottom_ << " " << objects_[id2]->top_ << endl;
					Rectangle r = objects_[id1]->Merge(*objects_[id2]);
					cout << r.left_ << " " << r.right_ << " " << r.bottom_ << " " << r.top_ << endl;
				}*/
				waste = objects_[id1]->Merge(*objects_[id2]).Area() - objects_[id1]->Area() - objects_[id2]->Area();
				/*if (debug) {
					cout << "waste: " << waste << "max waste: "<<max_waste<<" "<<(waste > max_waste)<<" "<<(waste < max_waste)<<endl;
				}*/
				//waste = ((Rectangle*)tree_node->children[i])->Merge(*((Rectangle*)tree_node->children[j])).Area() - ((Rectangle*)tree_node->children[i])->Area() - ((Rectangle*)tree_node->children[j])->Area();
			}
			else {
				waste = tree_nodes_[id1]->Merge(*tree_nodes_[id2]).Area() - tree_nodes_[id1]->Area() - tree_nodes_[id2]->Area();
				//waste = tree_node->children[i]->Merge(*tree_node->children[j]).Area() - tree_node->children[i]->Area() - tree_node->children[j]->Area();
			}
			/*cout << "waste: " << waste << endl;
			getchar();*/
			if (waste > max_waste) {
				max_waste = waste;
				seed1 = id1;
				seed2 = id2;
			}
		}
	}
	/*if (debug) {
		cout << "seeds found" << " " << seed1 << " " << seed2 << endl;
	}*/
	//cout << "seeds found" <<" "<<seed1<<" "<<seed2<< endl;
	double max_seed_left, min_seed_right, max_seed_bottom, min_seed_top;
	if (tree_node->is_leaf) {
		max_seed_left = max(objects_[seed1]->Left(), objects_[seed2]->Left());
		min_seed_right = min(objects_[seed1]->Right(), objects_[seed2]->Right());
		max_seed_bottom = max(objects_[seed1]->Bottom(), objects_[seed2]->Bottom());
		min_seed_top = min(objects_[seed1]->Top(), objects_[seed2]->Top());
	}
	else {
		max_seed_left = max(tree_nodes_[seed1]->Left(), tree_nodes_[seed2]->Left());
		min_seed_right = min(tree_nodes_[seed1]->Right(), tree_nodes_[seed2]->Right());
		max_seed_bottom = max(tree_nodes_[seed1]->Bottom(), tree_nodes_[seed2]->Bottom());
		min_seed_top = min(tree_nodes_[seed1]->Top(), tree_nodes_[seed2]->Top());
	}
	//cout << max_seed_left << " " << min_seed_right << " " << max_seed_bottom << " " << min_seed_top << endl;
	double x_seperation = min_seed_right > max_seed_left ? (min_seed_right - max_seed_left) : (max_seed_left - min_seed_right);
	double y_seperation = min_seed_top > max_seed_bottom ? (min_seed_top - max_seed_bottom) : (max_seed_bottom - min_seed_top);

	x_seperation = x_seperation / (tree_node->Right() - tree_node->Left());
	y_seperation = y_seperation / (tree_node->Top() - tree_node->Bottom());
	//cout << "seperation computed" << endl;
	vector<Rectangle*> recs;
	vector<TreeNode*> child_nodes;
	if (tree_node->is_leaf) {
		recs.resize(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			recs[i] = objects_[tree_node->children[i]]; 
		}
	}
	else {
		child_nodes.resize(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			child_nodes[i] = tree_nodes_[tree_node->children[i]];
		}
	}
	if (x_seperation < y_seperation) {
		if (tree_node->is_leaf) {
			sort(recs.begin(), recs.end(), SortedByBottom);
		}
		else {
			sort(child_nodes.begin(), child_nodes.end(), SortedByBottom);
		}
	}
	else {
		if (tree_node->is_leaf) {
			sort(recs.begin(), recs.end(), SortedByLeft);
		}
		else {
			sort(child_nodes.begin(), child_nodes.end(), SortedByLeft);
		}
	}
	if (tree_node->is_leaf) {
		bounding_box1 = MergeRange<Rectangle>(recs, 0, tree_node->entry_num / 2);
		bounding_box2 = MergeRange<Rectangle>(recs, tree_node->entry_num - tree_node->entry_num / 2, tree_node->entry_num);
		if (tree_node->entry_num % 2 == 1) {
			Rectangle rec1 = bounding_box1.Merge(*recs[tree_node->entry_num / 2]);
			Rectangle rec2 = bounding_box2.Merge(*recs[tree_node->entry_num / 2]);
			double area_increase1 = rec1.Area() - bounding_box1.Area();
			double area_increase2 = rec2.Area() - bounding_box2.Area();
			if (area_increase1 < area_increase2) {
				bounding_box1 = rec1;
			}
			else {
				bounding_box2 = rec2;
			}
		}
	}
	else {
		bounding_box1 = MergeRange<TreeNode>(child_nodes, 0, tree_node->entry_num / 2);
		bounding_box2 = MergeRange<TreeNode>(child_nodes, tree_node->entry_num - tree_node->entry_num / 2, tree_node->entry_num);
		if (tree_node->entry_num % 2 == 1) {
			Rectangle rec1 = bounding_box1.Merge(*child_nodes[tree_node->entry_num / 2]);
			Rectangle rec2 = bounding_box2.Merge(*child_nodes[tree_node->entry_num / 2]);
			double area_increase1 = rec1.Area() - bounding_box1.Area();
			double area_increase2 = rec2.Area() - bounding_box2.Area();
			if (area_increase1 < area_increase2) {
				bounding_box1 = rec1;
			}
			else {
				bounding_box2 = rec2;
			}
		}
	}
	if (values.size() != 5) {
		values.resize(5);
	}
	values[0] = bounding_box1.Perimeter() / tree_node->Perimeter();
	values[1] = bounding_box2.Perimeter() / tree_node->Perimeter();
	values[2] = bounding_box1.Area() / tree_node->Area();
	values[3] = bounding_box2.Area() / tree_node->Area();
	values[4] = SplitOverlap(bounding_box1, bounding_box2) / tree_node->Area();
}

void RTree::SplitQUADRATICCost(TreeNode* tree_node, vector<double>& values, Rectangle& bounding_box1, Rectangle& bounding_box2) {
	int seed1 = -1;
	int seed2 = -1;
	int seed_idx1 = -1, seed_idx2 = -1;
	double max_waste = - DBL_MAX;
	//find the pair of children that waste the most area were they to be inserted in the same node
	for (int i = 0; i < tree_node->entry_num - 1; i++) {
		for (int j = i + 1; j < tree_node->entry_num; j++) {
			double waste = 0;
			int id1 = tree_node->children[i];
			int id2 = tree_node->children[j];
			if (tree_node->is_leaf) {
				waste = objects_[id1]->Merge(*objects_[id2]).Area() - objects_[id1]->Area() - objects_[id2]->Area();
				//waste = ((Rectangle*)tree_node->children[i])->Merge(*((Rectangle*)tree_node->children[j])).Area() - ((Rectangle*)tree_node->children[i])->Area() - ((Rectangle*)tree_node->children[j])->Area();
			}
			else {
				waste = tree_nodes_[id1]->Merge(*tree_nodes_[id2]).Area() - tree_nodes_[id1]->Area() - tree_nodes_[id2]->Area();
				//waste = tree_node->children[i]->Merge(*tree_node->children[j]).Area() - tree_node->children[i]->Area() - tree_node->children[j]->Area();
			}
			if (waste > max_waste) {
				max_waste = waste;
				seed1 = id1;
				seed2 = id2;
				seed_idx1 = i;
				seed_idx2 = j;
			}
		}
	}
	int entry_count1 = 1;
	int entry_count2 = 1;
	if (tree_node->is_leaf) {
		bounding_box1.Set(*objects_[seed1]);
		bounding_box2.Set(*objects_[seed2]);
	}
	else {
		bounding_box1.Set(*tree_nodes_[seed1]);
		bounding_box2.Set(*tree_nodes_[seed2]);
	}
	//list<TreeNode*> unassigned_entry;
	list<int> unassigned_entry;
	for (int i = 0; i < tree_node->entry_num; i++) {
		if (i == seed_idx1 || i == seed_idx2)continue;
		unassigned_entry.push_back(tree_node->children[i]);
	}
	while (!unassigned_entry.empty()) {
		//make sure the two child nodes are balanced.
		if (unassigned_entry.size() + entry_count1 == TreeNode::minimum_entry) {
			if (tree_node->is_leaf) {
				for (auto it = unassigned_entry.begin(); it != unassigned_entry.end(); ++it) {
					Rectangle* rec_ptr = objects_[*it];
					bounding_box1.Include(*rec_ptr);
				}
			}
			else {
				for (auto it = unassigned_entry.begin(); it != unassigned_entry.end(); ++it) {
					TreeNode* node_ptr = tree_nodes_[*it];
					bounding_box1.Include(*node_ptr);
				}
			}
			break;
		}
		if (unassigned_entry.size() + entry_count2 == TreeNode::minimum_entry) {
			if (tree_node->is_leaf) {
				for (auto it = unassigned_entry.begin(); it != unassigned_entry.end(); ++it) {
					Rectangle* rec_ptr = objects_[*it];
					bounding_box2.Include(*rec_ptr);
				}
			}
			else {
				for (auto it = unassigned_entry.begin(); it != unassigned_entry.end(); ++it) {
					TreeNode* node_ptr = tree_nodes_[*it];
					bounding_box2.Include(*node_ptr);
				}
			}
			break;
		}
		//pick next: pick an unassigned entry that maximizes the difference between adding into different groups
		double max_difference = - DBL_MAX;
		double new_area1 = 0, new_area2 = 0;
		list<int>::iterator iter;
		int next_entry = -1;
		for (auto it = unassigned_entry.begin(); it != unassigned_entry.end(); ++it) {
			double d1 = 0, d2 = 0;
			if (tree_node->is_leaf) {
				d1 = bounding_box1.Merge(*objects_[*it]).Area();
				d2 = bounding_box2.Merge(*objects_[*it]).Area();
			}
			else {
				d1 = bounding_box1.Merge(*tree_nodes_[*it]).Area();
				d2 = bounding_box2.Merge(*tree_nodes_[*it]).Area();
			}
			double difference = d1 > d2 ? d1 - d2 : d2 - d1;
			if (difference > max_difference) {
				max_difference = difference;
				iter = it;
				next_entry = *it;
				new_area1 = d1;
				new_area2 = d2;
			}
		}
		unassigned_entry.erase(iter);
		//add the entry to the group with smaller area
		Rectangle *chosen_bounding_box = nullptr;
		if (new_area1 < new_area2) {
			chosen_bounding_box = &bounding_box1;
			entry_count1 += 1;
		}
		else if (new_area1 > new_area2) {
			chosen_bounding_box = &bounding_box2;
			entry_count2 += 1;
		}
		else {
			if (entry_count1 < entry_count2) {
				entry_count1 += 1;
				chosen_bounding_box = &bounding_box1;
			}
			else {
				entry_count2 += 1;
				chosen_bounding_box = &bounding_box2;
			}
		}
		if (tree_node->is_leaf) {
			chosen_bounding_box->Include(*objects_[next_entry]);
		}
		else {
			chosen_bounding_box->Include(*tree_nodes_[next_entry]);
		}
	}
	if (values.size() != 5) {
		values.resize(5);
	}
	values[0] = bounding_box1.Perimeter() / tree_node->Perimeter();
	values[1] = bounding_box2.Perimeter() / tree_node->Perimeter();
	values[2] = bounding_box1.Area() / tree_node->Area();
	values[3] = bounding_box2.Area() / tree_node->Area();
	values[4] = SplitOverlap(bounding_box1, bounding_box2) / tree_node->Area();

}


RTree* ConstructTree(int max_entry, int min_entry) {
	TreeNode::maximum_entry = max_entry;
	TreeNode::minimum_entry = min_entry;
	RTree* rtree = new RTree();
	return rtree;
}

void SetDefaultInsertStrategy(RTree* rtree, int strategy) {
	switch (strategy) {
	case 0: {
		rtree->insert_strategy_ = INS_AREA;
		break;
	}
	case 1: {
		rtree->insert_strategy_ = INS_MARGIN;
		break;
	}
	case 2: {
		rtree->insert_strategy_ = INS_OVERLAP;
		break;
	}
	case 3: {
		rtree->insert_strategy_ = INS_RANDOM;
		break;
	}
	}	
}
void SetDefaultSplitStrategy(RTree* rtree, int strategy) {
	switch (strategy) {
	case 0: {
		rtree->split_strategy_ = SPL_MIN_AREA;
		break;
	}
	case 1: {
		rtree->split_strategy_ = SPL_MIN_MARGIN;
		break;
	}
	case 2: {
		rtree->split_strategy_ = SPL_MIN_OVERLAP;
		break;
	}
	case 3: {
		rtree->split_strategy_ = SPL_QUADRATIC;
		break;
	}
	case 4: {
		rtree->split_strategy_ = SPL_GREENE;
	}
	}
}

int KNNQuery(RTree* rtree, double x, double y, int k) {
	vector<int> results;
	int node_access = rtree->KNNQuery(x, y, k, results);
	// cout << "node_access " << node_access << endl;
	// cout << "KNN query: ";
	// for (int i = 0; i < results.size(); i++) {
	// 	cout << results[i] << ", ";
	// }
	// cout << endl;

	vector<pair<double, int> > brute(rtree->objects_.size());
	for (int i = 0; i < rtree->objects_.size(); i++) {
		brute[i].first = rtree->MinDistanceToRec(x, y, i);
		brute[i].second = i;
	}
	sort(brute.begin(), brute.end());
	// cout << "Brute force: ";
	// for (int i = 0; i < k; i++) {
	// 	cout << brute[i].second << ", ";
	// }
	// cout << endl;
	return node_access;
}

int QueryRectangle(RTree* rtree, double left, double right, double bottom, double top) {
	Rectangle rec(left, right, bottom, top);
	rtree->Query(rec);
	int node_access = rtree->stats_.node_access;
	return node_access;
}

TreeNode* GetRoot(RTree* rtree) {
	return rtree->tree_nodes_[rtree->root_];
}

void SetDebug(RTree* rtree, int value) {
	rtree->debug = value;
}

int IsLeaf(TreeNode* node) {
	if (node->is_leaf) {
		return 1;
	}
	else {
		return 0;
	}
}

Rectangle* InsertRec(RTree* rtree, double left, double right, double bottom, double top) {
	Rectangle* rec = rtree->InsertRectangle(left, right, bottom, top);
	return rec;
}

TreeNode* InsertOneStep(RTree* rtree, Rectangle* rec, TreeNode* node, int strategy) {
	INSERT_STRATEGY ins_strat;
	switch (strategy) {
	case 0: {
		ins_strat = INS_AREA;
		break;
	}
	case 1: {
		ins_strat = INS_MARGIN;
		break;
	}
	case 2: {
		ins_strat = INS_OVERLAP;
		break;
	}
	case 3: {
		ins_strat = INS_RANDOM;
		break;
	}
	}
	TreeNode* next_iter = rtree->InsertStepByStep(rec, node, ins_strat);
	return next_iter;
}



TreeNode* SplitOneStep(RTree* rtree, TreeNode* node, int strategy) {
	SPLIT_STRATEGY spl_strat;
	switch (strategy) {
	case 0: {
		spl_strat = SPL_MIN_AREA;
		break;
	}
	case 1: {
		spl_strat = SPL_MIN_MARGIN;
		break;
	}
	case 2: {
		spl_strat = SPL_MIN_OVERLAP;
		break;
	}
	case 3: {
		spl_strat = SPL_QUADRATIC;
		break;
	}
	case 4: {
		spl_strat = SPL_GREENE;
	}
	}
	TreeNode* next_node = rtree->SplitStepByStep(node, spl_strat);
	return next_node;
}

int IsOverflow(TreeNode* node) {
	if (node->is_overflow) {
		return 1;
	}
	else {
		return 0;
	}
}

void Swap(Rectangle& rec1, Rectangle& rec2){
    Rectangle tmp(rec1);
    rec1.Set(rec2);
    rec2.Set(tmp);
}

void CheckOrder(Rectangle& rec1, Rectangle& rec2){
    //check the order of rec1 and rec2, sorted by left, right, bottom, top
    if(rec1.left_ < rec2.left_){
        return;
    }
    if(rec1.left_ > rec2.left_){
        Swap(rec1, rec2);
        return;
    }
    if(rec1.right_ < rec2.right_){
        return;
    }
    if(rec1.right_ > rec2.right_){
        Swap(rec1, rec2);
        return;
    }
    if(rec1.bottom_ < rec2.bottom_){
        return;
    }
    if(rec1.bottom_ > rec2.bottom_){
        Swap(rec1, rec2);
        return;
    }
    if(rec1.top_ < rec2.top_){
        return;
    }
    if(rec1.top_ > rec2.top_){
        Swap(rec1, rec2);
        return;
    }
    return;
}

bool IsSameSplit(Rectangle& rec00, Rectangle& rec01, Rectangle& rec10, Rectangle& rec11){
    bool is_same1 = false, is_same2 = false;
    if(rec00.left_ == rec10.left_ && rec00.right_ == rec10.right_ && rec00.bottom_ == rec10.bottom_ && rec00.top_ == rec10.top_){
        is_same1 = true;
    }
    if(rec01.left_ == rec11.left_ && rec01.right_ == rec11.right_ && rec01.bottom_ == rec11.bottom_ && rec01.top_ == rec11.top_){
        is_same2 = true;
    }
    return (is_same1 && is_same2);
}

int GetMinAreaContainingChild(RTree* rtree, TreeNode* tree_node, Rectangle* rec){
	return rtree->GetMinAreaContainingChild(tree_node, rec);
}


void RetrieveSortedInsertStates(RTree* tree, TreeNode* tree_node, Rectangle* rec, int topk, int state_type, double* states){
	switch(state_type){
		case 0:{
			//RL_FOR_ENLARGED_CHILDREN, deterministic for containing children
			tree->SortChildrenByMarginArea(tree_node, rec);
			tree->GetSortedInsertStates(tree_node, rec, states, topk, RL_FOR_ENLARGED_CHILDREN);
			break;
		}
		case 1:{
			//RL_FOR_CONTAINING_CHILDREN, deterministic for enlarged children
			tree->SortChildrenByArea(tree_node);
			tree->GetSortedInsertStates(tree_node, rec, states, topk, RL_FOR_CONTAINING_CHILDREN);
			break;
		}
	}
}

void RetrieveZeroOVLPSplitSortedByWeightedPerimeterState(RTree* tree, TreeNode* tree_node, double* states){
	if(tree->candidate_split_action.size() == 0){
		tree->candidate_split_action.resize(2);
	}
	vector<pair<double, int> > zero_ovlp_splits;
	double length[2] = {tree_node->Right() - tree_node->Left(), tree_node->Top() - tree_node->Bottom()};
	double center[2] = {0.5 * (tree_node->Left() + tree_node->Right()), 0.5 * (tree_node->Bottom() + tree_node->Top())};
	//cout<<"original center: "<<tree_node->origin_center[0]<<" "<<tree_node->origin_center[1]<<endl;
	//cout<<"new center: "<<center[0]<<" "<<center[1]<<endl;
	double perim_max = 2 * ( length[0] + length[1]) - min(length[0], length[1]);
	double asym[2] = {0, 0};
	double miu[2] = {0, 0};
	double delta[2] = {0, 0};
	double s = 0.5;
	double y1 = exp(-1 / s / s);
	double ys = 1 / (1 - y1);
	for(int i=0; i<2; i++){
		asym[i] = 2 * (center[i] - tree_node->origin_center[i])/length[i];
		miu[i] = (1 - 2 * TreeNode::minimum_entry/(TreeNode::maximum_entry + 1)) * asym[i];
		delta[i] = s * (1 + abs(miu[i]));
	}
	for(int i=0; i<tree->split_locations.size(); i++){
		int idx = i % (TreeNode::maximum_entry - 2 * TreeNode::minimum_entry + 2);
		int dim = i / (TreeNode::maximum_entry - 2 * TreeNode::minimum_entry + 2);
		double xi = 2.0 * (idx + TreeNode::minimum_entry) / (TreeNode::maximum_entry + 1)  - 1;
		double wf = ys * (exp(0 - (xi - miu[dim]) * (xi - miu[dim]) / delta[dim] / delta[dim]) - y1);
		double wg = tree->split_locations[i].perimeter1 + tree->split_locations[i].perimeter2 - perim_max;
		zero_ovlp_splits.emplace_back(wg * wf, i);
		//cout<<"loc: "<<idx<<" wg: "<<wg<<" wf "<<wf<<" xi: "<<xi<< " miu: "<<miu[dim]<<endl;
	}
	sort(zero_ovlp_splits.begin(), zero_ovlp_splits.end());
	//for(int i=0; i<zero_ovlp_splits.size(); i++){
	//	cout<<"("<<zero_ovlp_splits[i].first<<", "<<zero_ovlp_splits[i].second<<") ";
	//}
	//cout<<endl;
	double max_area = -DBL_MAX;
	double min_area = DBL_MAX;
	double max_perimeter = -DBL_MAX;
	double min_perimeter = DBL_MAX;

	for(int i=0; i<2; i++){
		int idx = zero_ovlp_splits[i].second;
		tree->candidate_split_action[i] = idx;
		states[i * 4] = tree->split_locations[idx].area1;
		states[i * 4 + 1] = tree->split_locations[idx].area2;
		states[i * 4 + 2] = tree->split_locations[idx].perimeter1;
		states[i * 4 + 3] = tree->split_locations[idx].perimeter2;
		max_area = max(max_area, states[i*4]);
		min_area = min(min_area, states[i*4 + 1]);
		max_perimeter = max(max_perimeter, states[i*4 + 2]);
		min_perimeter = min(min_perimeter, states[i*4 + 3]);
	}
	for(int i = 0; i<2; i++){
		states[i * 4] = (states[i * 4] - min_area) / (max_area - min_area + 0.1);
		states[i * 4 + 1] = (states[i * 4 + 1] - min_area) / (max_area - min_area + 0.1);
		states[i * 4 + 2] = (states[i * 4 + 2] - min_perimeter) / (max_perimeter - min_perimeter + 0.1);
		states[i * 4 + 3] = (states[i * 4 + 3] - min_perimeter) / (max_perimeter - min_perimeter + 0.1);
	}
}

void RetrieveZeroOVLPSplitSortedByPerimeterState(RTree* tree, TreeNode* tree_noe, double* states){
	if(tree->candidate_split_action.size() == 0){
		tree->candidate_split_action.resize(2);
	}
	vector<pair<double, int> > zero_ovlp_splits;
	for(int i=0; i<tree->split_locations.size(); i++){
		if(tree->split_locations[i].overlap == 0){
			double perimeter = max(tree->split_locations[i].perimeter1, tree->split_locations[i].perimeter2);
			zero_ovlp_splits.emplace_back(perimeter, i);
		}
	}
	sort(zero_ovlp_splits.begin(), zero_ovlp_splits.end());
	double max_area = -DBL_MAX;
	double min_area = DBL_MAX;
	double max_perimeter = -DBL_MAX;
	double min_perimeter = DBL_MAX;
	
	for(int i=0; i<2; i++){
		int idx = zero_ovlp_splits[i].second;
		tree->candidate_split_action[i] = idx;
		states[i * 4] = tree->split_locations[idx].area1;
		states[i * 4 + 1] = tree->split_locations[idx].area2;
		states[i * 4 + 2] = tree->split_locations[idx].perimeter1;
		states[i * 4 + 3] = tree->split_locations[idx].perimeter2;
		max_area = max(max_area, states[i*4]);
		min_area = min(min_area, states[i*4 + 1]);
		max_perimeter = max(max_perimeter, states[i*4 + 2]);
		min_perimeter = min(min_perimeter, states[i*4 + 3]);
	}
	for(int i = 0; i<2; i++){
		states[i * 4] = (states[i * 4] - min_area) / (max_area - min_area + 0.1);
		states[i * 4 + 1] = (states[i * 4 + 1] - min_area) / (max_area - min_area + 0.1);
		states[i * 4 + 2] = (states[i * 4 + 2] - min_perimeter) / (max_perimeter - min_perimeter + 0.1);
		states[i * 4 + 3] = (states[i * 4 + 3] - min_perimeter) / (max_perimeter - min_perimeter + 0.1);
	}
}

void RetrieveSortedSplitStates(RTree* tree, TreeNode* tree_node, int topk, double* states){
	if(tree->sorted_split_loc.size() == 0){
		tree->sorted_split_loc.resize(2 * (TreeNode::maximum_entry - TreeNode::minimum_entry * 2 + 2));
	}
	tree->SortSplitLocByPerimeter(tree_node);
	tree->GetSortedSplitStates(tree_node, states, topk);
}

void RetrieveShortSplitStates(RTree* tree, TreeNode* tree_node, double* states) {
	tree->GetShortSplitStates(tree_node, states);
}

void RetrieveSpecialStates(RTree* tree, TreeNode* tree_node, double* states) {
	tree->GetSplitStates(tree_node, states);
}

void RetrieveSpecialInsertStates(RTree* tree, TreeNode* tree_node, Rectangle* rec, double* states){
	tree->GetInsertStates(tree_node, rec, states);
}

void RetrieveSpecialInsertStates3(RTree* tree, TreeNode* tree_node, Rectangle* rec, double* states){
	tree->GetInsertStates3(tree_node, rec, states);
}

void RetrieveSpecialInsertStates4(RTree* tree, TreeNode* tree_node, Rectangle* rec, double* states){
	tree->GetInsertStates4(tree_node, rec, states);
}

void RetrieveSpecialInsertStates6(RTree* tree, TreeNode* tree_node, Rectangle* rec, double* states){
	tree->GetInsertStates6(tree_node, rec, states);
}


void RetrieveSpecialInsertStates7Fill0(RTree* tree, TreeNode* tree_node, Rectangle* rec, double* states){
	tree->GetInsertStates7Fill0(tree_node, rec, states);
}

void RetrieveSpecialInsertStates7(RTree* tree, TreeNode* tree_node, Rectangle* rec, double* states){
	tree->GetInsertStates7(tree_node, rec, states);
}

int RetrieveStates(RTree* tree, TreeNode* tree_node, double* states) {
	vector<double> values(5, 0);
	Rectangle rec[5][2];
	/*if (tree->debug) {
		cout << "tree_node->id=" << tree_node->id_ << " is_leaf: " << tree_node->is_leaf << " is_overflow: " << tree_node->is_overflow <<" entry_num: "<<tree_node->entry_num<< endl;
		cout << "retrieving states" << endl;
	}*/
	//cout << "tree_node->id=" << tree_node->id_ << " is_leaf: " << tree_node->is_leaf << " is_overflow: " << tree_node->is_overflow <<" entry_num: "<<tree_node->entry_num<< endl;
	//cout << "retrieving states" << endl;
	tree->SplitAREACost(tree_node, values, rec[0][0], rec[0][1]);
	for (int i = 0; i < 5; i++) {
		states[i] = values[i];
	}
	/*if (tree->debug) {
		cout << 1 << endl;
	}*/
	//cout << "1" << endl;
	tree->SplitMARGINCost(tree_node, values, rec[1][0], rec[1][1]);
	for (int i = 0; i < 5; i++) {
		states[5 + i] = values[i];
	}
	/*if (tree->debug) {
		cout << 2 << endl;
	}*/
	//cout << "2" << endl;
	tree->SplitOVERLAPCost(tree_node, values, rec[2][0], rec[2][1]);
	for (int i = 0; i < 5; i++) {
		states[10 + i] = values[i];
	}
	/*if (tree->debug) {
		cout << 3 << endl;
	}*/
	//cout << "3" << endl;
	tree->SplitGREENECost(tree_node, values, rec[3][0], rec[3][1]);
	for (int i = 0; i < 5; i++) {
		states[15 + i] = values[i];
	}
	/*if (tree->debug) {
		cout << 4 << endl;
	}*/
	//cout << "4" << endl;
	tree->SplitQUADRATICCost(tree_node, values, rec[4][0], rec[4][1]);
	for (int i = 0; i < 5; i++) {
		states[20 + i] = values[i];
	}
	/*if (tree->debug) {
		cout << 5 << endl;
	}*/
	//cout << "5" << endl;
	int is_valid = 0;
	Rectangle tmp;
    CheckOrder(rec[0][0], rec[0][1]);
	for(int i=1; i<5; i++){
        CheckOrder(rec[1][0], rec[1][1]);
        if(!IsSameSplit(rec[0][0], rec[0][1], rec[i][0], rec[i][1])){
            is_valid = 1;
            break;
        }
	}
	return is_valid;
}

int GetChildNum(TreeNode* node){
	return node->entry_num;
}

void GetMBR(RTree* rtree, double* boundary){
	TreeNode* root = rtree->tree_nodes_[rtree->root_];
    boundary[0] = root->left_;
    boundary[1] = root->right_;
    boundary[2] = root->bottom_;
    boundary[3] = root->top_;
}

void PrintTree(RTree* rtree) {
	rtree->Print();
}

void PrintEntryNum(RTree* rtree) {
	rtree->PrintEntryNum();
}

int GetActualSplitLocFromSortedPos(RTree* rtree, TreeNode* node, int sorted_loc){
	return rtree->sorted_split_loc[sorted_loc].second.second;
}
int GetActualSplitDimFromSortedPos(RTree* rtree, TreeNode* node, int sorted_loc){
	return 1 - (int)rtree->sorted_split_loc[sorted_loc].second.first;
}

void PrintSortedSplitLocs(RTree* rtree){
	for(int i=0; i<rtree->sorted_split_loc.size(); i++){
		cout<<"("<<rtree->sorted_split_loc[i].first<<" ["<<rtree->sorted_split_loc[i].second.first<<", "<<rtree->sorted_split_loc[i].second.second<<"]) ";
	}
	cout<<endl;

}

TreeNode* DirectInsert(RTree* rtree, Rectangle* rec) {
	TreeNode* iter = rtree->Root();
	while (true) {
		TreeNode* next_iter = rtree->InsertStepByStep(rec, iter);
		if (next_iter != nullptr) {
			iter = next_iter;
		}
		else {
			break;
		}
	}
	return iter;
}

void PrintTreeEntry(RTree* rtree){
	for(int i=0; i<rtree->tree_nodes_.size(); i++){
		cout<<rtree->tree_nodes_[i]->entry_num<<" ";
	}
	cout<<endl;
}


TreeNode* RRInsert(RTree* rtree, Rectangle* rec){
	TreeNode* iter = rtree->Root();
	while(true){
		TreeNode* next_iter = rtree->RRInsert(rec, iter);
		if(next_iter != nullptr){
			iter = next_iter;
		}
		else{
			break;
		}
	}
	return iter;
}

void RRSplit(RTree* rtree, TreeNode* node){
	TreeNode* iter = node;
	while(iter->is_overflow){
		iter = rtree->RRSplit(iter);
	}	
}

void DirectSplit(RTree* rtree, TreeNode* node) {
	TreeNode* iter = node;
	while (iter->is_overflow) {
		iter = rtree->SplitStepByStep(iter);
	}
}

void DirectSplitWithReinsert(RTree* rtree, TreeNode* node) {
	if (node->is_overflow) {
		list<int> candidates;
		rtree->RetrieveForReinsert(node, candidates);
		rtree->UpdateMBRForReinsert(node);
		for (auto it = candidates.begin(); it != candidates.end(); ++it) {
			Rectangle* rec = rtree->objects_[*it];
			DefaultInsert(rtree, rec);
		}
	}
	
}

int TryInsert(RTree* rtree, Rectangle* rec) {
	TreeNode* iter = rtree->Root();
	while (true) {
		TreeNode* next_iter = rtree->TryInsertStepByStep(rec, iter);
		if (next_iter != nullptr) {
			iter = next_iter;
		}
		else {
			if (iter->entry_num < TreeNode::maximum_entry) {
				if (iter->entry_num == 0) {
					iter->Set(*rec);
					iter->origin_center[0] = 0.5 * (iter->Right() + iter->Left());
					iter->origin_center[1] = 0.5 * (iter->Bottom() + iter->Top());
				}
				else {
					iter->Include(*rec);
				}
				iter->AddChildren(rec->id_);
				while (iter->father >= 0) {
					next_iter = rtree->tree_nodes_[iter->father];
					next_iter->Include(*rec);
					iter = next_iter;
				}
				return 1;
			}
			else {
				return 0;
			}
		}
	}
}

void DefaultSplit(RTree* rtree, TreeNode* tree_node){
	TreeNode* iter = tree_node;
	while(iter->is_overflow){
		iter = rtree->SplitStepByStep(iter);

	}
}

void DefaultInsert(RTree* rtree, Rectangle* rec) {
	TreeNode* iter = rtree->Root();
	while (true) {
		TreeNode* next_iter = rtree->InsertStepByStep(rec, iter);
		if (next_iter != nullptr) {
			iter = next_iter;
		}
		else {
			break;
		}
	}
	while (iter->is_overflow) {
		iter = rtree->SplitStepByStep(iter);
	}
}

int TotalTreeNode(RTree* rtree) {
	return (int)rtree->tree_nodes_.size();
}

double AverageNodeArea(RTree* rtree){
	double area = 0;
	int total_num = (int)rtree->tree_nodes_.size();
	for(int i=0; i<total_num; i++){
		area += rtree->tree_nodes_[i]->Area() / total_num;
	}
	return area;
}

double AverageNodeChildren(RTree* rtree){
	double average_children = 0;
	int total_num = (int)rtree->tree_nodes_.size();
	for(int i=0; i<total_num; i++){
		average_children += 1.0 * rtree->tree_nodes_[i]->entry_num / total_num;
	}
	return average_children;
}

int TreeHeight(RTree* rtree){
	return rtree->height_;
}

void Clear(RTree* rtree) {
	for(int i=0; i<rtree->objects_.size(); i++){
		delete rtree->objects_[i];
	}
	for(int i=0; i<rtree->tree_nodes_.size(); i++){
		delete rtree->tree_nodes_[i];
	}
	rtree->height_ = 1;
	rtree->tree_nodes_.clear();
	rtree->objects_.clear();
	rtree->root_ = rtree->CreateNode()->id_;
	rtree->tree_nodes_[rtree->root_]->is_leaf = true;
}

void RTree::Copy(RTree* tree) {
	for (int i = 0; i < objects_.size(); i++) {
		delete objects_[i];
	}
	for (int i = 0; i < tree_nodes_.size(); i++) {
		delete tree_nodes_[i];
	}
	objects_.resize(tree->objects_.size());
	tree_nodes_.resize(tree->tree_nodes_.size());
	for (int i = 0; i < objects_.size(); i++) {
		objects_[i] = new Rectangle(*tree->objects_[i]);
		objects_[i]->id_ = tree->objects_[i]->id_;
		assert(objects_[i]->id_ == i);
	}
	height_ = tree->height_;
	for (int i = 0; i < tree_nodes_.size(); i++) {
		tree_nodes_[i] = new TreeNode(tree->tree_nodes_[i]);
		assert(tree_nodes_[i]->id_ == i);
	}
	root_ = tree->root_;
}

void CopyTree(RTree* tree, RTree* from_tree) {
	Clear(tree);
	tree->Copy(from_tree);
}

void GetNodeBoundary(TreeNode* node, double* boundary){
	boundary[0] = node->Left();
	boundary[1] = node->Right();
	boundary[2] = node->Bottom();
	boundary[3] = node->Top();
	return;
}

int GetMinAreaEnlargementChild(RTree* rtree, TreeNode* tree_node, Rectangle* rec){
	return rtree->GetMinAreaEnlargementChild(tree_node, rec);
}
int GetMinMarginIncrementChild(RTree* rtree, TreeNode* tree_node, Rectangle* rec){
	return rtree->GetMinMarginIncrementChild(tree_node, rec);
}
int GetMinOverlapIncrementChild(RTree* rtree, TreeNode* tree_node, Rectangle* rec){
	return rtree->GetMinOverlapIncrementChild(tree_node, rec);
}

int GetNumberOfEnlargedChildren(RTree* rtree, TreeNode* tree_node, Rectangle* rec){
	return rtree->GetNumberOfEnlargedChildren(tree_node, rec);
}

int GetNumberOfNonOverlapSplitLocs(RTree* rtree, TreeNode* tree_node){
	rtree->PrepareSplitLocations(tree_node);
	int non_overlap_split_num = 0;
	for(int i = 0; i<rtree->split_locations.size(); i++){
		if(rtree->split_locations[i].overlap == 0){
			non_overlap_split_num += 1;
		}
	}
	return non_overlap_split_num;
}


TreeNode* SplitInMinOverlap(RTree* rtree, TreeNode* tree_node){
	double min_overlap = DBL_MAX;
	int split_loc = 0;
	int split_dim = 0;
	for(int i=0; i<rtree->split_locations.size(); i++){
		if(rtree->split_locations[i].overlap < min_overlap){
			min_overlap = rtree->split_locations[i].overlap;
			split_loc = rtree->split_locations[i].location;
			split_dim = rtree->split_locations[i].dimension;
		}
	}
	TreeNode* next_node = rtree->SplitInLoc(tree_node, split_loc, split_dim);
	return next_node;
}

void SetRR_s(double s_value) {
	TreeNode::RR_s = s_value;
}

void SetStartTimestamp(RTree* rtree){
	rtree->start_point = high_resolution_clock::now();
}
	
void SetEndTimestamp(RTree* rtree){
	rtree->end_point = high_resolution_clock::now();
}

double GetDurationInSeconds(RTree* rtree){
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(rtree->end_point - rtree->start_point);
	return time_span.count();
}

double GetIndexSizeInMB(RTree* rtree){
	double total_size = 0.0;
	//space cost of objects
	total_size += rtree->objects_.size() * (sizeof(double) * 4 + sizeof(int));
	//space cost of tree nodes
	for(int i=0; i<rtree->tree_nodes_.size(); i++){
		TreeNode* node = rtree->tree_nodes_[i];
		total_size += sizeof(int) * 2  + sizeof(int) * node->entry_num + sizeof(double) * 2 + sizeof(bool) * 2;
	}
	total_size = total_size / 1024 / 1024;
	return total_size;
}