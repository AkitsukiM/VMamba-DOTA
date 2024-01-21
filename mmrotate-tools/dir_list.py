import os
import argparse


def dir_list(path, output, ext):
    dirs = os.listdir(path)
    count = 0
    with open(output, 'w') as w:
        for file in dirs:
            filename_ext = os.path.splitext(file)
            if filename_ext[1] == ext:
                w.write(filename_ext[0] + '\n')
                count += 1
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type = str, default = "/home/ubuntu/Workspace/YuHongtian/Dataset/DOTA/val/images/", help = "path")
    parser.add_argument("--output", type = str, default = "/home/ubuntu/Workspace/YuHongtian/Dataset/DOTA/val/valset.txt", help = "output file")
    parser.add_argument("--ext", type = str, default = ".png", help = "extension name")
    opt = parser.parse_args()

    count = dir_list(path = opt.path, output = opt.output, ext = opt.ext)
    print(count)

