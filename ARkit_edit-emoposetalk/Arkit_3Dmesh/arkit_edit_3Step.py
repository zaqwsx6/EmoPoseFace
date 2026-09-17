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

import numpy as np
'''
03.Step
Function:
use base-arkit facial edit: global-emotion edit (base emotion template: arkit_3dmesh_edit_2Step.py)
'''
# 加载数据
# face_mesh = np.load('/media/china/DATA1/Result-work2/exp_result/ARkitedit/edit/emotion/M037-angry-level_3-001.npy')
# face_mesh = np.load('/media/china/DATA1/Result-work2/exp_result/ARkitedit/edit/emotion/M037-fear-level_3-001.npy')
face_mesh = np.load('/media/china/DATA1/Result-work2/exp_result/ARkitedit/edit/3DMesh_flame_emotion/M037-fear-level_3-001.npy')

# face_bl = np.load('/home/china/zxwork/ARkit_edit-emoposetalk/Arkit_3Dmesh/AU/emo/disgusted03.npy')

face_bl = np.load('/home/china/zxwork/ARkit_edit-emoposetalk/Arkit_3Dmesh/AU/emoUS/fear-US.npy') # 情感编辑模板
# f = np.load('mesh_faces.npy')  # 假设面相同
face_bl = face_bl.reshape(1, -1)
# 检查形状
# assert v1.shape == v2.shape, "顶点形状不匹配"
weight=0.6
# 加权相加 (例如70% mesh1 + 30% mesh2)
combined = face_mesh[:,:15069] + weight * face_bl

# output_path = '/media/china/DATA1/Result-work2/exp_result/ARkitedit/edit/3DMesh_flame_emotion_add/M037-disgusted-level_3-001-80editdisgusted03.npy' # def 1
# output_path = '/media/china/DATA1/Result-work2/exp_result/ARkitedit/edit/3DMesh_flame_emotion/M037-surprised-level_3-001-100edit.npy' # def 1

output_path = '/media/china/DATA1/Result-work2/exp_result/ARkitedit/edit/3DMesh_flame_emotion_US/M037-fear-level_3-001-60editfear-US.npy' # def 1
np.save(output_path,combined)
# 保存
# np.save('combined_vertices_weighted.npy', combined)
# np.save('combined_faces.npy', f)