import requests
import os
import time
import logging

# 配置日志
logging.basicConfig(filename='download_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# 请替换以下变量的值
api_token = 'YOUR_API_TOKEN'
search_query = 'asian face'  # 根据你的需求可以更改搜索查询
per_page = 20  # 每页结果数，默认为20
page = 15  # 要检索的页码
sleep_time = 3  # 休眠时间（秒）,根据你的需求调整

# API的基础URL
base_url = "https://cgfaces.com/api"

# 创建API请求头
headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {api_token}"
}

# 构建搜索URL
search_url = f"{base_url}/search?q={search_query}&per_page={per_page}&page={page}"

# 发送搜索请求
response = requests.get(search_url, headers=headers)

# 记录请求
logging.info(f"发送搜索请求: {search_url}")

# 检查请求是否成功
if response.status_code == 200:
    # 解析响应数据
    data = response.json()
    
    # 创建一个目录来保存图片
    if not os.path.exists('downloaded_images'):
        os.makedirs('downloaded_images')
    
    # 遍历搜索结果并下载图片
    for image in data['results']:
        uuid = image['uuid']
        image_url = image['urls']['full']  # 获取大尺寸图片的URL
        
        # 构建图片的文件名
        image_filename = f"{uuid}.jpg"
        
        # 下载图片并保存到指定目录
        with open(f"downloaded_images/{image_filename}", 'wb') as file:
            file.write(requests.get(image_url).content)
            print(f"下载了图片: {image_filename}")
            # 记录下载信息
            logging.info(f"下载了图片: {image_filename}")
        
        # 休眠一段时间
        time.sleep(sleep_time)
else:
    print("请求失败，状态码:", response.status_code)
    # 记录请求失败信息
    logging.error(f"请求失败，状态码: {response.status_code}")
