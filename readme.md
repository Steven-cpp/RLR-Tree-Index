# Deep Learning for Big Data Management

这是本项目的记录文档，包括相关知识的学习笔记、对于该项目各周期的进度记录，以及讨论的汇总。

## I. Paper Reading

在与Prof Gao讨论之后，我的 Master Project 就是要将 RLR Tree 实现到 PostgresSQL 中。首先需要深入理解 RTree，以及 RLR Tree 的实现思路。

### 1. R-Tree

R-Tree 是 RLR Tree 的基础，它是于 1984 年由 Guttman 提出来的，主要用于**空间搜索**。以下的笔记参考的材料有：

- [Introduction to R-Tree](https://www.geeksforgeeks.org/introduction-to-r-tree/)
- [The R-Tree: A dynamic index structure for spatial searching](https://hpi.de/rabl/teaching/winter-term-2019-20/foundations-of-database-systems/the-r-tree-a-dynamic-index-structure-for-spatial-searching.html)
- [R-Tree: algorithm for efficient indexing of spatial data](https://bartoszsypytkowski.com/r-tree/)

R-Tree 是用于对高维数据和地理数据 (例如坐标和矩形) 进行有效地存取，它的特点是只有一个根结点，而且子节点指向的内容完全包含在父节点的范围中。而只有叶子结点才真正包含指向的对象的内容，这里的数据对象指的是一个闭区间的 $n$ 维矩形。一个典型的 R-Tree 示意图如下：

![image-20220923193722920](https://cdn.jsdelivr.net/gh/Steven-cpp/myPhotoSet@master/img/image-20220923193722920.png)

<div style='font-size: 14px; 
            color: rgba(117, 117, 117, 1); 
            line-height: 20px;     
    				max-width: 80%;
    				min-height: 43px;
    				display: inline-block;
   	 				padding: 10px;
    				margin: 0 4em;
    				border-bottom: 1px solid #eee;' > 
图1: R-Tree的示意图. 图a显示了一个三层的R-Tree, 它每个结点的最大指针数为3, 从而每个结点的可用指针数都不能小于3/2(即2). 而且, 只有叶子结点指向的才是实际的数据对象, 而且子结点完全包含在父结点中, 这一点从图b中可以见得.</div>

**搜索目标对象**

这里的目标对象指的就是图1中的实线矩形，搜索算法会自顶向下地遍历每个结点，检查它是否完全包含目标矩形。如果是，就选中它的子节点继续遍历。该算法的问题是一个结点下需要搜索多个子树，如果树的高度特别高，时间就会很长，难以度量最差的表现。

**更新 R-Tree**

CondenseTree: 在删除结点时触发。当数据对象被删掉后，该算法检查对应的叶子结点是否仍有 $m/2$ 个可用指针，其中 $m$ 为每层的最大结点数。如果小于该阈值，则会删除该叶子结点，以及父结点中的指针，并将叶子结点中的所有指针保存至临时的数组 $Q$ 中。同时，再对父结点进行类似的检查，最后将 $Q$ 中的元素插入到 R-Tree 中。

AdjustTree: 在插入结点时触发。如果插入后，当前结点的指针数 > $m$，那么就需要对该结点进行分割。在分割的时候需要确保分割后的区域应当是最小化的，正如下图所示：

![image-20220924152147338](https://cdn.jsdelivr.net/gh/Steven-cpp/myPhotoSet@master/img/image-20220924152147338.png)

<div style='font-size: 14px; 
            color: rgba(117, 117, 117, 1); 
            text-align: center; 
            line-height: 20px;     
    				min-height: 43px;
   	 				padding: 10px;
    				margin: 0 1em;
    				border-bottom: 1px solid #eee;' > 
图2: 对结点进行split操作</div>

**结点切分**

切分结点的方法有两种：

1. **线性复杂度切分**

   从 $Q$ 中选取距离最远的两个点分别作为新分组的头元素，然后将剩余的点随机分配至新分组中

2. **平方复杂度切分**

   从 $Q$ 中选取所能张成的最大区域的两个区域作为新分组的头元素

### 2. Reinforcement Learning Based R-Tree

将该篇论文的要点整理如下：

| Title             | A Reinforcement Learning Based R-Tree for Spatial Data Indexing in Dynamic Environments |
| ----------------- | ------------------------------------------------------------ |
| Author            | TuGu, GaoCong @ NTU                                          |
| Year              | 2021                                                         |
| Prerequisite      | [R-Tree, 1984](https://www.google.com/url?sa=t&source=web&rct=j&url=http://www-db.deis.unibo.it/courses/SI-LS/papers/Gut84.pdf&ved=2ahUKEwjIo4Tigpz6AhU0TmwGHetMAnYQFnoECBYQAQ&usg=AOvVaw39B_K-orDTFqVkCujGjYVz), [Recursive Model Index, 2018@MIT](file:///Users/shiqi/Downloads/DBM02_RMI%20Learned%20Index.pdf), |
| Motivation        | 1. 使用 learned indices 替换传统的索引结构 (e.g B-Tree) 往往能够取得不错的性能表现；<br />2. 但是这需要完全替换原有的结构和查询算法，遇到了很多实现上的困难；<br />3. 本文想在<u>不改变索引结构</u>的情况下，采用基于 RL 的方法，提高空间查找的效率。 |
| Current Challenge | 1. 现有 R-tree 的各种 insert 和 split 操作得到的索引树在查询的速度上，都没有显著的优势；<br />2. 将 ChooseSubTree 和 Split 操作形式化为两个连续的 MDP 是相当困难的，如何定义每个过程的状态、动作和奖励信号呢？<br />3. 难以使用 RL 来找到最优的过程，因为当前的 good action 可能会由于之前的 bad action 而得到惩罚值。 |
| Related Work      | 1. Learned Index<br />- data and query limited;<br />- not accurate;<br />- <u>cannot handle updates, or need to periodic rebuild</u>.<br />- replace index structure and query algorithm<br />2. Heuristic Strategies used in R-Tree<br />- no single index outperforms the others |
| Method            | 通过基于 RL 的模型，确定如何建立 R-Tree<br />具体地，这是通过将 insert 和 split 操作形式化为两个连续的 MDP，再使用 RL 来最优化。这就需要定义 MDP 的 state, action, reward signal, transition.<br />**1. State**<br />对每个结点的子节点进行遍历，选取前 $k$ 个插入后面积增加量最少的子节点。并计算$\Delta Area$, $\Delta Peri$, $\Delta Ovlp$, $OR(R)$ 并以相应的最大值正则化，连接后作为该结点的状态向量；<br />**2. Action**<br />类似的，选取当前结点的 $k$ 个子节点构成其动作空间<br />**3. Reward Signal**<br />设计 1 个 reference tree (RT)，将所要插入的对象同时插入到 RT 和 RLR-Tree 中，以两者的*结点访问率 (node access rate)* 的差作为激励信号。 |
| Baseline          |                                                              |
| Highlight         |                                                              |
| Future Challenge  |                                                              |
| Relevant Work     | 1. [The "AI+R"-tree: An Instance-optimized R-tree](https://arxiv.org/pdf/2207.00550v1.pdf): 将原有的数据库查找操作变为多标签分类任务；<br />2. |



## II. Psql Learning



### 1. Index Archetecture

在 PostgreSQL 8.4.1 中支持的索引有：B-Tree 索引、Hash 索引、GiST 索引和 GIN 索引。

[PostgreSQL: BTree-implementation](https://www.postgresql.org/docs/current/btree-implementation.html)

> **如何实现一个索引？**
>
> 1. 把树的结构写出来，确定它所有接口的 API；
> 2. 链接到数据库的操作中。
>    - 索引如何存储？

#### 1）B-Tree

[Postgres Indexes Under the Hood](https://rcoh.me/posts/postgres-indexes-under-the-hood/#:~:text=Indexes%20in%20Postgres&text=These%20indexes%20are%20implemented%20internally,implementer%20of%20the%20data%20structure.)

**Branching Factor 的选取**

就是一个结点最多能容纳的数据元素的个数

B-Trees are extremely shallow data structures. Because the branching factor is typically in the thousands, they can store millions of elements in only 2-3 layers. When used in a database, this means only 2-3 disk seeks are required to find any given item, greatly improving performance over the dozens of seeks required for a comparable on-disk binary search tree or similar data structure.

Typical branching factors will be between a few hundred to a few thousand items per page.

**Specification**

1. Postgres nodes have a fixed amount of bytes

   If you have variable-size data, each node in your index will actually have a different number of children

2. Highr key allows concurrency

   The “high-key” pointer allows readers to detect that this split has occurred: If you’re looking for a value greater than the high key, you must follow the right-link! The right link allows the reader to traverse directly to the newly split node where the key now resides.

#### 2）GiST Index

[Implementation of GiST indexing for Postgres](https://github.com/postgres/postgres/tree/master/src/backend/access/gist)

[【参考材料1】The GiST Indexing Project](http://gist.cs.berkeley.edu/)

GiST (Generalized Search Tree) 称为通用搜索树，它为各种类型的索引树 (R-trees, B+-trees, hB-trees, TV-trees, Ch-Trees 等) 都提供了一个统一的接口，允许用户在任意数据类型上进行索引。除此之外，GiST 还具有数据和 *查询的可拓展性*。

> 🔍 **查询的可拓展性**
>
> 这里指用于可以在 GiST 中自定义查询谓词。以前的搜索树在其处理的数据方面是可扩展的。例如，POSTGRES支持可扩展的B+树和R树。这意味着你可以使用POSTGRES在任何你想要的数据类型上建立一个B+树或R树。但是 B+ 树只支持范围谓词（<, = >），而 R 树只支持 $[n, d]$ 范围查询（包含、包含、相等）。因此，如果你用 POSTGRES B+ 树来索引，比如说，一堆电影，你只能提出类似 "查找所有 < T2 的电影 "的查询。虽然这个查询可能有意义（例如，小于可能意味着价格不那么贵、评分不那么高），但这样的写法并不显然。相反，你想问的是关于电影的特定查询，比如 "找到所有有爆炸场面的电影"，"找到所有有吴京的电影"，或者 "找到所有有摩托车追逐的电影"。这样的查询在 B+ 树、R 树或者除了 GiST 之外的任何其他已知结构中都无法直接支持。
>
> 相比之下，你可以通过编程让 GiST 支持任何查询谓词，包括上面提到的 `爆炸场面` 和其他谓词。要让 GiST 启动和运行，只需要实现 4 个用户定义的方法，这些方法定义了树中键的行为。当然，这些方法会是非常复杂的，来支持复杂的查询。但对于所有的标准查询（如 B- 树、R- 树等），就不需要这些了。简而言之，GiST 结合了新的可扩展性、通用性、代码重用和一个漂亮的简洁界面。

由于 B-Tree 处理的是数值型、R-Tree 是 Bounding Box，这种统一性就意味着 GiST 的 key 是独特的。它的 Key 是由用户自定义的类的成员，并且可以通过判断它的某些属性来使得键的指针能够指向所有的 item，即支持类似于小于操作的属性。

**Key 的 Class 的实现**

以下给出了用于键的用户自定义的 class 需要实现的 4 个接口：

1. **Consistent:** This method lets the tree search correctly. Given a key **p** on a tree page, and user query **q**, the Consistent method should return **NO** if it is certain that both **p** and **q** cannot be true for a given data item. Otherwise it should return **MAYBE**.

   > ? **p** 为 true 的含义是什么

2. **Union:** This method consolidates information in the tree. Given a set **S** of entries, this method returns a new key **p** which is true for all the data items below **S**. A simple way to implement **Union** is to return a predicate equivalent to the disjunction of the keys in **S**, i.e. "**p1** or **p2** or **p3** or...".

3. **Penalty:** Given a choice of inserting a new data item in a subtree rooted by entry **<p, ptr>**, return a number representing how bad it would be to do that. Items will get inserted down the path of least **Penalty** in the tree.

4. **PickSplit:** As in a B-tree, pages in a GiST occasionally need to be split upon insertion of a new data item. This routine is responsible for deciding which items go to the new page, and which ones stay on the old page.

There are some optional additional methods that can enhance performance. These are described in [the original paper](http://s2k-ftp.cs.berkeley.edu/gist/gist.ps) on the data structure.

而对于索引项的增删改查，GiST 已经内置实现了，但这恰恰是本项目需要修改的地方。本项目应当是通过使用与索引项管理相关的 7 种方法，实现：

1. 索引的创建 `gistbuild`；
2. 索引项的插入 `gistdoinsert`;
3. 索引的查询 `gistnext`.



## III. Implementation

首先，我要了解 R-Tree 是如何进行增删的，我找到了[Delete a Node from BST](https://practice.geeksforgeeks.org/problems/delete-a-node-from-bst/1?utm_source=gfg&utm_medium=article&utm_campaign=bottom_sticky_on_article)， 可以在有空的时候练一练。不过我的重点还是应该在看论文，了解这个模型的架构。因为对于这些增删改查的操作，这篇论文是使用了基于 RL 的方法，不要求先学懂传统的增删的方法。

### 1. Data Structure



### 2. Operators

the Split operation may be propagated upwards













