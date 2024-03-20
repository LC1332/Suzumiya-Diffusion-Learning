import pandas as pd
import numpy as np

# 指定Parquet文件的路径
output_parquet_path = 'output/celeb_features-clip_openai.parquet'
# 保存ROC曲线到指定路径
feature_column_name = 'FaRL'

model_save_name = "output/lda_openai_clip_model_norm.pkl"

if_normalization = True # l2 normalization for each feature

# 读取Parquet文件
df = pd.read_parquet(output_parquet_path)

# 这个df有三列分别是 'celeb_name', 'image_name', 'FaRL'

# 计算 'celeb_name' 列中不同名字的数量
unique_celeb_names_count = df['celeb_name'].nunique()

print(f"不同的名人名字数量为: {unique_celeb_names_count}")

X = []
y = []



# 遍历df的每一行
for index, row in df.iterrows():
    # 获取celeb_name
    celeb_name = row['celeb_name']
    # 获取feature
    feature = row[feature_column_name]
    # 如果celeb_name不在celeb

    if if_normalization:
        feature = feature / (1e-10 + np.linalg.norm(feature))

    X.append( feature )
    y.append( celeb_name )

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


X = np.array(X)
y = np.array(y)

# 初始化LDA模型，n_components设置为目标维度512
# 注意: LDA最大能降到类别数-1的维度，因此，如果类别数少于513，则实际维度会低于512
lda = LinearDiscriminantAnalysis(n_components=512)

# 由于类别数可能影响能达到的最大维度，这里打印实际的维度
print(f"实际降维到: min(512, 类别数-1) = {min(512, len(np.unique(y))-1)}")

print("Starting to fit LDA model")

# 训练LDA模型
lda.fit(X, y)

print("LDA model fitted")

# 定义LDA_project函数，用于将新特征向量投影到LDA空间
def LDA_project(feature):
    return lda.transform([feature])

# 测试LDA_project函数
new_feature = LDA_project(X[0])
print(f"变换后的特征维度: {new_feature.shape}")


# 保存LDA模型
import pickle
with open(model_save_name, 'wb') as f:
    pickle.dump(lda, f)

# 已知这段程序可以正常运行，并且每个celeb对应的features的dimension都是一致的
# 我希望训练一个LDA矩阵，把整个features 线性变换到512维，并且人脸识别的准确率尽可能大
# 请用python为我实现，并最终给出new_feature = LDA_project( feature ) 这个函数

