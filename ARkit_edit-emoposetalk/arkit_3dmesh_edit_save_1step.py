import tkinter as tk
from PIL import Image, ImageTk
import os, shutil
import cv2
import tempfile
import numpy as np
from subprocess import call
import argparse
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa' #egl、osmesa
import pyrender
import trimesh
from psbody.mesh import Mesh


'''
1 Step：
Extract obj 文件的所有信息，存储为字典形式
'''

def load_and_process_blendshapes(basis_path, blendshapes_dir, template_path):
    # 加载基础mesh
    basis_mesh = trimesh.load_mesh(basis_path)
    basis_vertices = basis_mesh.vertices

    # 初始化存储
    blendshape_names = []
    vertices_data = {}
    deltas = {}

    # 遍历blendshapes目录
    for file in sorted(os.listdir(blendshapes_dir)):
        if not file.endswith('.obj'):
            continue

        # 获取blendshape名称(去掉.obj后缀)
        name = file[:-4]
        blendshape_names.append(name)

        # 加载mesh并存储顶点数据
        mesh_path = os.path.join(blendshapes_dir, file)
        mesh = trimesh.load_mesh(mesh_path)
        vertices_data[name] = mesh.vertices.copy()

        # 计算与基础mesh的差值
        deltas[name] = mesh.vertices - basis_vertices

    # 加载模板mesh
    template_mesh = Mesh(filename=template_path)

    return {
        'basis_vertices': basis_vertices,
        'blendshape_vertices': vertices_data,
        'blendshape_deltas': deltas,
        'blendshape_names': blendshape_names,
        'template': template_mesh
    }


# 路径配置
basis_path = '/home/china/zxwork/FLAME_PyTorch-master/ARkit_edit-emoposetalk/bs/Basis.obj'
blendshapes_dir = '/home/china/zxwork/FLAME_PyTorch-master/ARkit_edit-emoposetalk/bs/exp'
template_path = '/home/china/zxwork/FLAME_PyTorch-master/model/FLAME_sample.ply'

# 处理数据
blendshape_data = load_and_process_blendshapes(basis_path, blendshapes_dir, template_path)

# 保存处理后的数据以便后续使用
output_path = '/home/china/zxwork/ARkit_edit-emoposetalk/Arkit_3Dmesh/arkit_blendshapes.npz'
np.savez(
    output_path,
    basis_vertices=blendshape_data['basis_vertices'],
    blendshape_vertices=blendshape_data['blendshape_vertices'],
    blendshape_deltas=blendshape_data['blendshape_deltas'],
    blendshape_names=blendshape_data['blendshape_names']
)

print(f"处理完成，数据已保存到 {output_path}")
print(f"包含 {len(blendshape_data['blendshape_names'])} 个blendshapes:")
print(blendshape_data['blendshape_names'])