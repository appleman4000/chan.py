#!/usr/bin/env python
# encoding:utf-8
import os
import shutil

main_files = ["App.py", "Monitor.py", "Simulator.py", "GenerateDataset.py", "TrainingKerasModel.py", "Simulator.py",
              "AutoML.py"]
# 项目根目录下不用（能）转译的py文件（夹）名，用于启动的入口脚本文件一定要加进来
ignore_files = ['build', 'package', 'venv', '__pycache__', '.git', 'setup.py', 'setup_main.py',
                '__init__.py'] + main_files
# 项目子目录下不用（能）转译的'py文件（夹）名
ignore_names = ['__init__.py']
# 不需要原样复制到编译文件夹的文件或者文件夹
ignore_move = ['venv', '__pycache__', 'server.log', 'setup.py', 'setup_main.py']
# 需要编译的文件夹绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 将以上不需要转译的文件(夹)加上绝对路径
ignore_files = [os.path.join(BASE_DIR, x) for x in ignore_files]
# 是否将编译打包到指定文件夹内 (True)，还是和源文件在同一目录下(False)，默认True
# 若没有打包文件夹，则生成一个
translate_pys = []


# 编译需要的py文件
def translate_dir(path):
    pathes = os.listdir(path)
    for p in pathes:
        if p in ignore_names:
            continue
        if p.startswith('__') or p.startswith('.') or p.startswith('build'):
            continue
        f_path = os.path.join(path, p)
        if f_path in ignore_files:
            continue
        if os.path.isdir(f_path):
            translate_dir(f_path)
        else:
            if not f_path.endswith('.py') and not f_path.endswith('.pyx'):
                continue
            if f_path.endswith('__init__.py') or f_path.endswith('__init__.pyx'):
                continue
            with open(f_path, 'r', encoding='utf8') as f:
                content = f.read()
                if not content.startswith('# cython: language_level=3'):
                    content = '# cython: language_level=3\n' + content
                    with open(f_path, 'w', encoding='utf8') as f1:
                        f1.write(content)
            print(f_path)
            os.system('python setup.py ' + f_path + ' build_ext')
            translate_pys.append(f_path)
            f_name = '.'.join(f_path.split('.')[:-1])
            py_file = '.'.join([f_name, 'py'])
            c_file = '.'.join([f_name, 'c'])
            print(f"f_path: {f_path}, c_file: {c_file}, py_file: {py_file}")
            if os.path.exists(c_file):
                os.remove(c_file)


# 移除编译临时文件
def remove_dir(path, rm_path=True):
    if not os.path.exists(path):
        return
    pathes = os.listdir(path)
    for p in pathes:
        f_path = os.path.join(path, p)
        if os.path.isdir(f_path):
            remove_dir(f_path, False)
            os.rmdir(f_path)
        else:
            os.remove(f_path)
    if rm_path:
        os.rmdir(path)


def batch_rename(src_path):
    filenames = os.listdir(src_path)
    same_name = []
    count = 0
    for filename in filenames:
        old_name = os.path.join(src_path, filename)
        if os.path.isdir(old_name):
            batch_rename(old_name)
        file_name, file_extension = os.path.splitext(filename)
        if file_extension == ".pyd" or file_extension == ".so":
            old_pyd = filename.split(".")
            new_pyd = str(old_pyd[0]) + file_extension
        else:
            continue
        change_name = new_pyd
        count += 1
        new_name = os.path.join(src_path, change_name)
        if change_name in filenames:
            same_name.append(change_name)
            continue
        os.rename(old_name, new_name)


def find_directories(root_dir, target_name):
    matching_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == target_name:
                matching_dirs.append(os.path.join(dirpath, dirname))
    return matching_dirs


def run():
    remove_dir(os.path.join(BASE_DIR, 'build'))
    translate_dir(BASE_DIR)
    batch_rename(os.path.join(BASE_DIR, 'build'))
    target_project = find_directories("build", "py")[0]
    for file in main_files:
        dst = target_project + "/" + file
        if os.path.isdir(file):
            shutil.copytree(file, dst, dirs_exist_ok=True)
        else:
            shutil.copy(file, dst)


if __name__ == '__main__':
    run()