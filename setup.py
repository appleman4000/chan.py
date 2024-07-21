#!/usr/bin/env python
#  -*- coding: utf-8 -*-
import sys
from distutils.core import setup

from setuptools import Extension

try:
    from Cython.Build import cythonize
except:
    print("你没有安装Cython，请安装 pip install Cython")
    print("本项目需要 Visual Studio 2022 的C++开发支持，请确认安装了相应组件")

arg_list = sys.argv
f_name = arg_list[1]
sys.argv.pop(1)

# setup(ext_modules=cythonize(f_name), compiler_directives={'boundscheck': False,
#                                                           'wraparound': False,
#                                                           'cdivision': True,
#                                                           'profile': False,
#                                                           'linetrace': False,
#                                                           'language_level': 3,
#                                                           }, extra_compile_args=["-O3"])
# 定义扩展模块
extensions = [
    Extension(
        "*",  # 模块名称
        [f_name],  # Cython 源文件
        extra_compile_args=["/O2"],  # 编译器选项
        include_dirs=[]  # 包含目录，例如 NumPy 头文件
    )
]

# 配置和编译扩展模块
setup(
    name="MyModule",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'boundscheck': False,
            'wraparound': True,
            'cdivision': True,
            'profile': False,
            'linetrace': False,
            'language_level': 3,
        }
    ),
)
