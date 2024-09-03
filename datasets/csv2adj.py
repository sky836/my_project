import csv
import pickle

import numpy as np

distance_df_filename = r'PEMS03/PEMS03.csv'
id_filename = r'PEMS03/PEMS03.txt'
num_of_vertices = 358
with open(id_filename, 'r') as f:
    id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
             dtype=np.float32)

distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                    dtype=np.float32)

# distance file中的id并不是从0开始的 所以要进行重新的映射；id_filename是节点的顺序
with open(id_filename, 'r') as f:
    id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引


# 用于存储转换后的数据
converted_data = []
with open(distance_df_filename, 'r') as f:
    f.readline()  # 略过表头那一行
    reader = csv.reader(f)
    for row in reader:
        if len(row) != 3:
            continue
        i, j, distance = int(row[0]), int(row[1]), float(row[2])
        # A[id_dict[i], id_dict[j]] = 1
        distaneA[id_dict[i], id_dict[j]] = distance
        converted_data.append([id_dict[i], id_dict[j],distance])

print('distaneA:', distaneA)
# Save distanceA as a pickle file
with open('PEMS03/pems03adj.pkl', 'wb') as f:
    pickle.dump(distaneA, f)
# # 写入新的文件
# with open('pems03adj.txt', 'w', newline='') as f:
#     writer = csv.writer(f, delimiter=' ')  # 使用空格作为分隔符
#
#     # 写入每一行数据
#     for row in converted_data:
#         writer.writerow(row)
