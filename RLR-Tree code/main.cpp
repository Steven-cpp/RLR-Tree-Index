#include <iostream>
#include<fstream>
#include"RTree.h"
#include<time.h>
#include<cstdlib>
#include<sstream>
using std::ifstream;
using std::ofstream;
using std::rand;
using std::stringstream;
using std::cout;

void DataLoader(vector<Rectangle>& rectangles) {
	rectangles.resize(10000);
	ifstream ifs("d:\\projects\\RLRTree\\dataset\\uniform10k.txt", std::ifstream::in);
	for (int i = 0; i < 10000; i++) {
		double l, r, b, t;
		ifs >> rectangles[i].left_ >> rectangles[i].right_ >> rectangles[i].bottom_ >> rectangles[i].top_;
	}
	ifs.close();
}
int NaiveVerifier(vector<Rectangle>& rectangles, Rectangle& query) {
	int result = 0;
	for (int i = 0; i < rectangles.size(); i++) {
		if (rectangles[i].IsOverlap(&query)) {
			result += 1;
		}
	}
	return result;
}

void TestRRStar() {
	RTree* tree = ConstructTree(50, 20);
	int total_access = 0;
	ifstream ifs("./dataset/skew100k.txt", std::ifstream::in);
	SetDefaultInsertStrategy(tree, 1);
	SetDefaultSplitStrategy(tree, 1);
	for (int i = 0; i < 100000; i++) {
		double l, r, b, t;
		ifs >> l >> r >> b >> t;
		Rectangle* rectangle = InsertRec(tree, l, r, b, t);
		TreeNode* tree_node = DirectInsert(tree, rectangle);
		//TreeNode* tree_node = RRInsert(tree, rectangle);
		RRSplit(tree, tree_node);
	}
	ifs.close();
	ifs.open("./dataset/query1k.txt", std::ifstream::in);
	ofstream ofs("./rrstar.log", std::ofstream::out);
	for (int i = 0; i < 1000; i++) {
		//cout<<"query "<<i<<endl;
		double l, r, b, t;
		ifs >> l >> r >> b >> t;
		Rectangle query(l, r, b, t);
		int access = QueryRectangle(tree, l, r, b, t);
		ofs << tree->result_count << endl;
		total_access += access;
	}
	ofs.close();
	ifs.close();
	Clear(tree);
	cout << "average node access is " << 1.0 * total_access / 1000 << endl;
}

void TestBaseline(int insert_strategy, int split_strategy) {
	//stringstream ss;
	//ss << "d:\\projects\\RLRTree\\dataset\\uniform10k.txt" << insert_strategy << "_" << split_strategy << ".txt";
	RTree* tree = ConstructTree(50, 20);
	SetDefaultInsertStrategy(tree, insert_strategy);
	SetDefaultSplitStrategy(tree, split_strategy);
	int total_access = 0;
	ifstream ifs("./dataset/skew100k.txt", std::ifstream::in);
	for (int i = 0; i < 100000; i++) {
		double l, r, b, t;
		ifs >> l >> r >> b >> t;
		Rectangle* rectangle = InsertRec(tree, l, r, b, t);
		//srand(time(NULL));
		//int split_strategy = rand() % 5;
		//SetDefaultSplitStrategy(tree, split_strategy);
		DefaultInsert(tree, rectangle);
	}
	ifs.close();
	ifs.open("./dataset/query1k.txt", std::ifstream::in);
	ofstream ofs("./reference.log", std::ofstream::out);
	for (int i = 0; i < 1000; i++) {
		//cout<<"query "<<i<<endl;
		double l, r, b, t;
		ifs >> l >> r >> b >> t;
		//Rectangle query(l, r, b, t);
		//int access = QueryRectangle(tree, l, r, b, t);
		//ofs << tree->result_count << endl;
		//total_access += access;
		KNNQuery(tree, l, b, 10);
		getchar();
	}
	ofs.close();
	ifs.close();
	Clear(tree);
	cout << "insert strategy " << tree->insert_strategy_ << " split strategy " << tree->split_strategy_ << endl;
	cout << "average node access is " << 1.0 * total_access / 1000 << endl;
}

int main() {
	
	TestBaseline(1, 1);
	return 0;	
}
