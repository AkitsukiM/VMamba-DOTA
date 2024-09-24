# code from https://www.cnblogs.com/xiaodekaixin/p/11203857.html

import hashlib
import argparse

def get_file_md5(path):
    m = hashlib.md5()       # 创建md5对象
    with open(path,'rb') as fobj:
        while True:
            data = fobj.read(4096)
            if not data:
                break
            m.update(data)  # 更新md5对象

    print(m.hexdigest())    # 打印md5对象

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type = str, default = None, help = "")
    opt = parser.parse_args()

    get_file_md5(opt.path)

