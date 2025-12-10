#!/usr/bin/env python
"""
临时补丁：修复 av2 包中的 np.bool 问题
在导入 av2 之前运行此脚本
"""
import numpy as np
import sys

# 为 NumPy 1.20+ 添加 np.bool 的后向兼容性
if not hasattr(np, 'bool'):
    np.bool = np.bool_
    print("✅ 已添加 np.bool 兼容性别名")

if not hasattr(np, 'int'):
    np.int = np.int_
    print("✅ 已添加 np.int 兼容性别名")

if not hasattr(np, 'float'):
    np.float = np.float_
    print("✅ 已添加 np.float 兼容性别名")

if not hasattr(np, 'complex'):
    np.complex = np.complex_
    print("✅ 已添加 np.complex 兼容性别名")

if not hasattr(np, 'object'):
    np.object = np.object_
    print("✅ 已添加 np.object 兼容性别名")

if not hasattr(np, 'str'):
    np.str = np.str_
    print("✅ 已添加 np.str 兼容性别名")

print(f"NumPy version: {np.__version__}")
print("兼容性补丁已应用！")
