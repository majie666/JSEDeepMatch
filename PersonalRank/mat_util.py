# -*-coding:utf8-*-
from __future__ import division
from scipy.sparse import coo_matrix
# >> >  # Constructing a matrix using ijv format
# >> > row = np.array([0, 3, 1, 0])
# >> > col = np.array([0, 3, 1, 2])
# >> > data = np.array([4, 5, 7, 9])
# >> > coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
# array([[4, 0, 9, 0],
#        [0, 7, 0, 0],
#        [0, 0, 0, 0],
#        [0, 0, 0, 5]])
import numpy as np
from util import read

# 1.由之前的二分图得到矩阵公式中的M矩阵，所有(item+user)顶点，所有(item+user)顶点位置（为了求r）
def graph_to_m(graph):
    '''
    :param graph: user and item graph
    :return:matrix M, a list 所有(item+user)顶点, a dict 所有(item+user)顶点位置
    '''
    vertex = list(graph.keys())  # 所有(item+user)顶点
    address_dict = {}  # 所有(item+user)顶点位置
    total_len = len(vertex)
    for index in range(len(vertex)):
        address_dict[vertex[index]] = index  # 每一行对应一个顶点
    row = []
    col = []
    data = []
    for key in graph:  # element所有item+user的顶点
        weight = round(1 / len(graph[key]), 3)  # graph[element]二分图中所有和element相连接的顶点
        row_index = address_dict[key]
        for element in graph[key]:
            col_index = address_dict[element]
            row.append(row_index)
            col.append(col_index)
            data.append(weight)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    m = coo_matrix((data, (row, col)), shape=(total_len, total_len))
    return m, vertex, address_dict

# 2.矩阵算法personal_rank的公式，得到（1-alpha*M^T）
def mat_all_point(m_matrix, vertex, alpha):
    '''
    矩阵算法personal_rank的公式
    :param m_matrix:
    :param vertex: 所有(item+user)顶点
    :param alpha: 随机游走的概率
    :return:  矩阵
    '''
    # 初始化单位矩阵（如果使用numpy创建，容易超内存）
    total_len = len(vertex)
    row = []
    col = []
    data = []
    for index in range(total_len):
        row.append(index)
        col.append(index)
        data.append(1)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    eye_t = coo_matrix((data, (row, col)), shape=(total_len, total_len))
    # 稀疏矩阵运算使用csr格式，加快计算
    return eye_t.tocsr() - alpha * m_matrix.tocsr().transpose()

if __name__=="__main__":
    graph = read.get_graph_from_data("../data/log.txt")
    m,vertex,address_dict = graph_to_m(graph)
    print(mat_all_point(m,vertex,0.8))