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
04 Step = 02+03
Function: local fine-grained edit 
use:
***facial control panel*** : arkit_control_1.py
Visualise local actions using the control panel, and edit them using the 52 BlendShape-3D mesh parameters in the control panel
'''

def add_text_to_image(img, text):
    # setup text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = text

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]

    # get coords based on boundary
    textX = (img.shape[1] - textsize[0]) // 2
    textY = img.shape[0] - (img.shape[0] + textsize[1]) // 16

    # add text centered on image
    cv2.putText(img, text, (textX, textY), font, 1, (255, 255, 255), 2)
    return img


# The implementation of rendering is borrowed from VOCA: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0):
    camera_params = {'c': np.array([400, 400]),
                     'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                     'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v - t_center).T).T + t_center

    intensity = 2.0

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        metallicFactor=0.8,
        roughnessFactor=0.8
    )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f)

    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material, smooth=True)

    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])

    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                       fy=camera_params['f'][1],
                                       cx=camera_params['c'][0],
                                       cy=camera_params['c'][1],
                                       znear=frustum['near'],
                                       zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 1.0 - z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3, 3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3, 3] = pos
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')

        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]


def render_sequence_meshes(sequence_vertices, template, ):
    num_frames = sequence_vertices.shape[0]
    center = np.mean(sequence_vertices[0], axis=0)
    for i_frame in range(num_frames):
        render_mesh = Mesh(sequence_vertices[i_frame], template.f)
        pred_img = render_mesh_helper(render_mesh, center, tex_img=None)
        pred_img = pred_img.astype(np.uint8)
    return pred_img



#linner—edit
def add_blend_shape(face_mesh, face_bl, I, window_size,tau):
    """
    将face_bl混合到face_mesh的第I帧及其周围帧

    参数:
        face_mesh: (N, 15069) 的面部运动序列
        face_bl: (1, 15069) 的基准面部
        I: 要插入的中心帧索引
        window_size: 混合窗口的半径（左右各扩展多少帧）

    返回:
        混合后的面部运动序列
    """
    N = face_mesh.shape[0]
    blended = face_mesh.copy()

    # 确保I在有效范围内
    if I < 0 or I >= N:
        raise ValueError("帧索引I超出范围")

    # 计算混合窗口的起始和结束
    start = max(0, I - window_size)
    end = min(N, I + window_size + tau + 1)

    # 为窗口内的帧计算权重
    for i in range(start, end):
        # 计算当前帧相对于中心帧的距离
        # distance = abs(i - I)

        # if distance == 0:
        #     # 中心帧完全使用face_bl
        #     weight = 1.0
        # elif distance <= window_size:
        #     # 线性衰减权重
        #     weight = 1.0 - (distance / (window_size + 1))
        # else:
        #     weight = 0.0

        # 计算权重
        if I - window_size <= i < I:
            weight = (i - (I - window_size)) / (window_size + 1)  # 上升段
        elif I <= i < I + tau:
            weight = 1.0  # 平台段
        elif I + tau <= i < I + tau + window_size:
            weight = 1.0 - (i - (I + tau - 1)) / (window_size + 1)  # 下降段
        else:
            weight = 0.0



        # 混合当前帧
        blended[i][:15069] = face_mesh[i][:15069] + weight * face_bl

    return blended

# 示例用法
# face_mesh = np.random.rand(100, 15069)  # 示例数据 (N=100帧)
# face_bl = np.random.rand(1, 15069)      # 示例基准面部
# I = 50                                 # 中心帧索引
# window_size = 10                       # 混合窗口半径

# blended_mesh = add_blend_shape(face_mesh, face_bl, I, window_size)


# 后续调控操作过程
# 加载之前保存的npz文件
# data = np.load('/home/china/zxwork/FLAME_PyTorch-master/ARkit_edit-emoposetalk/bs/processed_blendshapes.npz',
#                allow_pickle=True)
data = np.load('/home/china/zxwork/ARkit_edit-emoposetalk/Arkit_3Dmesh/arkit_blendshapes.npz',
               allow_pickle=True)

# 获取数据
basis_vertices = data['basis_vertices']
blendshape_vertices = data['blendshape_vertices'].item()  # 转换为字典
blendshape_deltas = data['blendshape_deltas'].item()  # 转换为字典
blendshape_names = data['blendshape_names']

# 检查所有可用的blendshape名称
print("所有可用的blendshapes:")
print(blendshape_names)

# 示例：计算这三个表情的混合效果
# 52 BlendShape-3D mesh parameters
weights={'browDownLeft':0, 'browDownRight':0, 'browInnerUp':0,
    'browOuterUpLeft':0, 'browOuterUpRight':0, 'cheekPuff':0,
    'cheekSquintLeft':0, 'cheekSquintRight':0, 'eyeBlinkLeft':0,
    'eyeBlinkRight':0, 'eyeLookDownLeft':0.8, 'eyeLookDownRight':0.8,
    'eyeLookInLeft':0, 'eyeLookInRight':0, 'eyeLookOutLeft':0,
    'eyeLookOutRight':0, 'eyeLookUpLeft':0, 'eyeLookUpRight':0,
    'eyeSquintLeft':0.8, 'eyeSquintRight':0.8, 'eyeWideLeft':0,
    'eyeWideRight':0, 'jawForward':0, 'jawLeft':0, 'jawOpen':0,
    'jawRight':0, 'mouthClose':0, 'mouthDimpleLeft':0,
    'mouthDimpleRight':0, 'mouthFrownLeft':0, 'mouthFrownRight':0,
    'mouthFunnel':0, 'mouthLeft':0, 'mouthLowerDownLeft':0,
    'mouthLowerDownRight':0, 'mouthPressLeft':0, 'mouthPressRight':0,
    'mouthPucker':0, 'mouthRight':0, 'mouthRollLower':0,
    'mouthRollUpper':0, 'mouthShrugLower':0, 'mouthShrugUpper':0,
    'mouthSmileLeft':0, 'mouthSmileRight':0, 'mouthStretchLeft':0,
    'mouthStretchRight':0, 'mouthUpperUpLeft':0, 'mouthUpperUpRight':0,
    'noseSneerLeft':0, 'noseSneerRight':0}
# weights = {'browDownLeft': 0.5, 'browDownRight': 0.3, 'browInnerUp': 0.2}

combined_delta = np.zeros_like(basis_vertices)  # 初始化组合偏移

for shape_name, weight in weights.items():
    if shape_name in blendshape_deltas:
        combined_delta += blendshape_deltas[shape_name] * weight
    else:
        print(f"警告: {shape_name} 不存在，跳过")



print("\n组合后的顶点数据:")
print(f"形状: {combined_delta.shape}")
print(f"前5个顶点:\n{combined_delta[:5]}")

# 应用组合偏移到基础mesh
# final_vertices = basis_vertices + combined_delta

# print("\n组合后的顶点数据:")
# print(f"形状: {final_vertices.shape}")
# print(f"前5个顶点:\n{final_vertices[:5]}")
# output_path = '/home/china/zxwork/ARkit_edit-emoposetalk/Arkit_3Dmesh/AU/sad.npy' # def 1
# output_path = '/home/china/zxwork/ARkit_edit-emoposetalk/Arkit_3Dmesh/AU/emo/happy.npy' # def 1
# output_path = '/home/china/zxwork/ARkit_edit-emoposetalk/Arkit_3Dmesh/AU/emo/surprised.npy' # def 1
# output_path = '/home/china/zxwork/ARkit_edit-emoposetalk/Arkit_3Dmesh/AU/emo/fear.npy' # def 1
# output_path = '/home/china/zxwork/ARkit_edit-emoposetalk/Arkit_3Dmesh/AU/emo/angry.npy' # def 1
# output_path = '/home/china/zxwork/ARkit_edit-emoposetalk/Arkit_3Dmesh/AU/emo/disgusted.npy' # def 1
# todo
# output_path = '/home/china/zxwork/ARkit_edit-emoposetalk/Arkit_3Dmesh/AU/emo/disgusted03.npy' # def 1
# np.save(output_path,combined_delta)

# 渲染
temp = np.load('/home/china/zxwork/ARkit_edit-emoposetalk/AU/FLAME_template.npy')
final_vertices = temp + combined_delta
final_vertices = final_vertices.reshape(1, 5023, 3)
image_path = '/home/china/zxwork/ARkit_edit-emoposetalk/Arkit_3Dmesh/image/Arkit_bl/editadd.png'  # todo 1 def 2
template_file = '/home/china/zxwork/FLAME_PyTorch-master/model/FLAME_sample.ply'
template = Mesh(filename=template_file)
img=render_sequence_meshes(final_vertices, template)
# 选择对应的图片路径
cv2.imwrite(image_path,img)


#***arkit edit***
# face_mesh---3d mesh path
face_mesh = np.load('/media/china/DATA1/Result-work2/exp_result/ARkitedit/edit/3DMesh_flame_emotion/M037-happy-level_3-001.npy')

# face_bl = np.load('/home/china/zxwork/ARkit_edit-emoposetalk/Arkit_3Dmesh/AU/emo/disgusted03.npy')
face_bl = combined_delta
# f = np.load('mesh_faces.npy')  # 假设面相同
face_bl = face_bl.reshape(1, -1)

I = 30                                 # Edit Frame I
window_size = 15                       # window w #kuai 3/5/7 #low 10/15/20
tau= 60                                # Duration

blended_mesh = add_blend_shape(face_mesh, face_bl, I, window_size,tau)
# weight = 0.8
# # 加权相加 (例如70% mesh1 + 30% mesh2)
# combined = face_mesh[:,:15069] + weight * face_bl

# output_path--final path
# todo 2

output_path = '/media/china/DATA1/Result-work2/exp_result/ARkitedit/edit/3DMesh_flame_ArkitLinner_US/M037-happy-level_3-001-edit05win15_30f_t60.npy' # def 1
# output_path = '/media/china/DATA1/Result-work2/exp_result/ARkitedit/edit/3DMesh_flame_emotion/M037-surprised-level_3-001-100edit.npy' # def 1
np.save(output_path,blended_mesh)


