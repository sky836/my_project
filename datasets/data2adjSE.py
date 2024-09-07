import csv

distance_df_filename = r'PEMS04/PEMS04.csv'
converted_data = []
with open(distance_df_filename, 'r') as f:
    f.readline()  # 略过表头那一行
    reader = csv.reader(f)
    for row in reader:
        if len(row) != 3:
            continue
        i, j, distance = int(row[0]), int(row[1]), float(row[2])
        converted_data.append([i, j, distance])

# 写入新的文件
with open('PEMS04/pems04adj.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')  # 使用空格作为分隔符
    # 写入每一行数据
    for row in converted_data:
        writer.writerow(row)
