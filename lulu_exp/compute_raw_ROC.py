import pandas as pd

# 指定Parquet文件的路径
output_parquet_path = 'output/celeb_features_FaRL_50k.parquet'

# 读取Parquet文件
df = pd.read_parquet(output_parquet_path)

# 这个df有三列分别是 'celeb_name', 'image_name', 'FaRL'

# 计算 'celeb_name' 列中不同名字的数量
unique_celeb_names_count = df['celeb_name'].nunique()

print(f"不同的名人名字数量为: {unique_celeb_names_count}")

# 我希望进一步建立celeb_name到features的映射，每个celeb对应一个list of feature
celeb_to_features = {}

# 遍历df的每一行
for index, row in df.iterrows():
    # 获取celeb_name
    celeb_name = row['celeb_name']
    # 获取feature
    feature = row['FaRL']
    # 如果celeb_name不在celeb
    if celeb_name not in celeb_to_features:
        celeb_to_features[celeb_name] = []

    # 将feature添加到celeb_name对应的列表中
    celeb_to_features[celeb_name].append(feature)

positive_pairs = []
negative_pairs = []

import random

celeb_names = list(celeb_to_features.keys())

negative_index = [x for x in range(len(celeb_names))]

random.shuffle(negative_index)

for x in range(len(celeb_names)):
    if negative_index[x] == x:
        while negative_index[x] == x:
            negative_index[x] = random.randint(0, len(celeb_names) - 1)
        

for index, celeb_name in enumerate(celeb_names):
    # 获取这个celeb对应的所有feature
    features = celeb_to_features[celeb_name]

    if len(features) == 0:
        continue

    A = None
    B = None
    C = None

    # 如果len(features) > 2 ，随机抽取两个feature A, B 将 (A,B) append到positive_pairs
    if len(features) > 2:
        A, B, C = random.sample(features, 3)
    else:
        A, B = random.sample(features, 2)
    if A is not None and B is not None:
        positive_pairs.append((A, B))

    if C is None:
        C = random.sample(features, 1)
        C = C[0]

    neg_index = negative_index[index]
    features_neg = celeb_to_features[celeb_names[neg_index]]
    neg_feat_sel = random.sample(features_neg, 1)
    negative_pairs.append((C, neg_feat_sel[0]))
    # break

# 在这之后我想像lfw一样，计算negative和positive的consine相似度，然后画ROC曲线
# 将曲线保存到output/FaRL_ROC.png

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# 计算正面对和负面对的余弦相似度
positive_similarities = [cosine_similarity([pair[0]], [pair[1]])[0][0] for pair in positive_pairs]
negative_similarities = [cosine_similarity([pair[0]], [pair[1]])[0][0] for pair in negative_pairs]

# 合并正面对和负面对的相似度，并创建标签（正面对为1，负面对为0）
similarities = np.array(positive_similarities + negative_similarities)
labels = np.array([1] * len(positive_similarities) + [0] * len(negative_similarities))

# 计算ROC曲线的值
fpr, tpr, thresholds = roc_curve(labels, similarities)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# 保存ROC曲线到指定路径
output_path = 'output/FaRL_ROC.png'
plt.savefig(output_path)
plt.close()
