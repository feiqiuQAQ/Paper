import os
from datetime import datetime


def mkdir(save_path):
    # 获取今天的日期
    today = datetime.now().strftime('%Y-%m-%d')

    # 创建文件夹路径
    folder_path = f'{save_path}/{today}'

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 如果文件夹不存在，则创建它
        os.makedirs(folder_path)
        print(f'文件夹 "{folder_path}" 已创建。')
    else:
        print(f'文件夹 "{folder_path}" 已存在。')

    return folder_path
