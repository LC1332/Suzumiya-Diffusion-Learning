import os
from PIL import Image

def mirror_images(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否为图片文件
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            # 拼接文件路径
            input_path = os.path.join(input_folder, filename)
            
            # 获取原始文件名和扩展名
            name, extension = os.path.splitext(filename)
            
            # 生成镜像文件名
            mirrored_filename = f"{name}-mirror image{extension}"
            output_path = os.path.join(output_folder, mirrored_filename)

            # 打开图片文件
            try:
                with Image.open(input_path) as img:
                    # 镜像化图片
                    mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)

                    # 保存镜像化后的图片
                    mirrored_img.save(output_path)
                    print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # 输入文件夹路径
    input_folder = "D://python project//Suzumiya-Diffusion-Learning//train_data//data//image//CeleAHQ_subset_200"
    # 输出文件夹路径
    output_folder = "D://python project//Suzumiya-Diffusion-Learning//train_data//data//image//CeleAHQ_subset_200 mirror image"

    # 执行镜像化图片操作
    mirror_images(input_folder, output_folder)


