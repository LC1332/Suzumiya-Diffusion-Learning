import pickle
import os

root_path = "FaceData/Data"

# imdbface_loose_crop_subset_0

# 指定.pkl文件的路径
file_path = 'output/index.pkl'

# 读取.pkl文件
with open(file_path, 'rb') as f:
    df, index_to_name, index_to_row_number = pickle.load(f)


sub_folders = []
for i in range(10):
    sub_str = "imdbface_loose_crop_subset_" + str(i)
    sub_folders.append( f"{root_path}/{sub_str}/{sub_str}" )

celeb2images = {}

from tqdm import tqdm

count = 0

for sub_folder in tqdm(sub_folders):
    # 使用os.walk遍历root_dir下的所有子目录
    for subdir, dirs, files in os.walk(sub_folder):
        for file in files:
            if not file.endswith('.jpg'):
                continue

            folder_name = os.path.basename(subdir)

            key = (folder_name, file)

            image_path = os.path.join(subdir, file)

            if key not in index_to_name:
                continue

            celeb_name = index_to_name[key]

            count += 1

            if celeb_name not in celeb2images:
                celeb2images[celeb_name] = []
                celeb2images[celeb_name].append(image_path)
                # print("create celeb with name ", celeb_name, " and image path ", image_path)
            else:
                celeb2images[celeb_name].append(image_path)
    # break

# 确保output目录存在
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 指定保存文件的路径
file_path = os.path.join(output_dir, 'celeb2images.pkl')
print(count)

# 将df, index_to_name 和 index_to_row_number 保存到一个pkl文件
with open(file_path, 'wb') as f:
    pickle.dump(celeb2images, f)
