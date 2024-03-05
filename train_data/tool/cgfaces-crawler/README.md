# cgfaces-crawler

cgfaces网站的爬虫，通过官方提供的API爬取cgfaces网站的人像图片

## 开始

如要使用该项目，请先克隆该储存库

```
git clone https://github.com/2404589803/cgfaces-crawler.git
```

## 在cgfaces官网创建账号，并获取api令牌

网址如下：

```
https://cgfaces.com/user/api-tokens
```

## 在cgfaces crawler.py文件中填写需要修改的参数

在cgfaces crawler.py文件中填写以下参数

- api_token='
- search_query='face'  # 根据你的需求可以更改搜索查询
- per_page=10  # 每页结果数，默认为20

- page=2  # 要检索的页码

- sleep_time=5  # 休眠时间（秒）,根据你的需求调整

## 运行爬虫

```
py cgfaces crawler.py
```
