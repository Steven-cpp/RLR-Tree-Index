g++ -std=c++11 -shared -O2 -fPIC RTree.cpp -o tree.so
g++ -std=c++11 -O2 RTree.cpp main.cpp -o RTree
