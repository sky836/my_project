import csv

import numpy as np


num_nodes = 1024
adj_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)
len_row = 32
len_column = 32
edges = []
dirs = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
for i in range(len_row):
    for j in range(len_column):
        index = i * len_column + j  # grid_id
        for d in dirs:
            nei_i = i + d[0]
            nei_j = j + d[1]
            if nei_i >= 0 and nei_i < len_row and nei_j >= 0 and nei_j < len_column:
                nei_index = nei_i * len_column + nei_j  # neighbor_grid_id
                adj_mx[index][nei_index] = 1
                adj_mx[nei_index][index] = 1
                edges.append((index, nei_index, 1))
                edges.append((nei_index, index, 1))

# 写入新的文件
with open('T-Drive/TDriveadj.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')  # 使用空格作为分隔符
    # 写入每一行数据
    for row in edges:
        writer.writerow(row)




