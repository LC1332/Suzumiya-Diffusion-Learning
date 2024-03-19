import pandas as pd

# 读取CSV文件
df = pd.read_csv('FaceData/IMDb-Face_meta-information.csv', sep='\t')  # 假设字段之间是用制表符（\t）分隔的


from tqdm import tqdm


# 获取DataFrame的行数
num_rows = len(df)
print("行数:", num_rows)

# 初始化两个映射字典
index_to_name = {}
index_to_row_number = {}

from tqdm import tqdm

# 遍历DataFrame来填充字典
for i, row in tqdm(df.iterrows(), total = num_rows):
    # 创建索引键
    key = (row['IMDbIndex'], row['ImageIndex'])
    
    # 填充(IMDbIndex, ImageIndex)到Name的映射
    index_to_name[key] = row['Name']
    
    # 填充(IMDbIndex, ImageIndex)到行号的映射
    index_to_row_number[key] = i


import pickle
import os

# 假设df, index_to_name, 和 index_to_row_number 已经被定义并填充了数据
# df = ...
# index_to_name = ...
# index_to_row_number = ...

# 确保output目录存在
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 指定保存文件的路径
file_path = os.path.join(output_dir, 'index.pkl')

# 将df, index_to_name 和 index_to_row_number 保存到一个pkl文件
with open(file_path, 'wb') as f:
    pickle.dump((df, index_to_name, index_to_row_number), f)

# 获取文件大小
file_size = os.path.getsize(file_path)

# 对应的读取

# 打印文件大小，转换为KB
print(f"文件大小: {file_size / 1024:.2f} KB")

# 指定.pkl文件的路径
file_path = 'output/index.pkl'

# 读取.pkl文件
with open(file_path, 'rb') as f:
    df, index_to_name, index_to_row_number = pickle.load(f)

# 打印DataFrame的前几行来确认
print("DataFrame的前几行:")
print(df.head())

# 打印index_to_name和index_to_row_number的一个小样本来确认
print("\nindex_to_name的一部分:")
print(list(index_to_name.items())[:5])

print("\nindex_to_row_number的一部分:")
print(list(index_to_row_number.items())[:5])
