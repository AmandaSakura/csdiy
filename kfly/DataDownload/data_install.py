import requests
import os
from tqdm import tqdm

def download_file_with_progress(url, folder_path, file_name):
    """
    从给定的URL下载文件，并保存到指定文件夹，同时显示下载进度条。

    Args:
        url (str): 文件的下载URL。
        folder_path (str): 文件保存的目录路径。
        file_name (str): 文件保存时的本地文件名。
    """
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"创建了文件夹: {folder_path}")

    local_filepath = os.path.join(folder_path, file_name)

    print(f"开始下载: {file_name} 到 {local_filepath}")
    try:
        # 使用 stream=True 允许我们逐步下载大文件
        response = requests.get(url, stream=True)
        response.raise_for_status() # 检查请求是否成功

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 KB

        # 使用 tqdm 显示进度条
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=file_name) as pbar:
            with open(local_filepath, 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))
        print(f"文件 '{file_name}' 下载成功！\n")

    except requests.exceptions.RequestException as e:
        print(f"下载文件 '{file_name}' 失败: {e}\n")
    except Exception as e:
        print(f"发生未知错误: {e}\n")

# 定义下载链接
train_url = "https://bjcdn.openstorage.cn/aicontest/2025%E7%AE%97%E6%B3%95%E8%B5%9B/%E7%94%A8%E6%88%B7%E6%96%B0%E5%A2%9E%E9%A2%84%E6%B5%8B%E6%8C%91%E6%88%98%E8%B5%9B%20%E8%B5%9B%E5%AD%A33/train.csv"
test_url = "https://bjcdn.openstorage.cn/aicontest/2025%E7%AE%97%E6%B3%95%E8%B5%9B/%E7%94%A8%E6%88%B7%E6%96%B0%E5%A2%9E%E9%A2%84%E6%B5%8B%E6%8C%91%E6%88%98%E8%B5%9B%20%E8%B5%9B%E5%AD%A33/testA_data.csv"

# 定义保存的文件夹和文件名
train_folder = "train_data"
train_file_name = "train.csv"

test_folder = "test_data"
test_file_name = "testA_data.csv"

# 执行下载任务
download_file_with_progress(train_url, train_folder, train_file_name)
download_file_with_progress(test_url, test_folder, test_file_name)