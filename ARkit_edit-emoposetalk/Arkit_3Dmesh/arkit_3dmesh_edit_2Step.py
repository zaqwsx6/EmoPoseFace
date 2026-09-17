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
02 Step (base Flame 3D-Mesh 52 blendshape)
Emotion-based facial template generation
use:
Emotion facial template for global emotion edit
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
# 假设权重分别为0.5, 0.3, 0.2
# weights={'browDownLeft':0, 'browDownRight':0, 'browInnerUp':0,
#     'browOuterUpLeft':0, 'browOuterUpRight':0, 'cheekPuff':0,
#     'cheekSquintLeft':0, 'cheekSquintRight':0, 'eyeBlinkLeft':0,
#     'eyeBlinkRight':0, 'eyeLookDownLeft':0, 'eyeLookDownRight':0,
#     'eyeLookInLeft':0, 'eyeLookInRight':0, 'eyeLookOutLeft':0,
#     'eyeLookOutRight':0, 'eyeLookUpLeft':0, 'eyeLookUpRight':0,
#     'eyeSquintLeft':0, 'eyeSquintRight':0, 'eyeWideLeft':0,
#     'eyeWideRight':0, 'jawForward':0, 'jawLeft':0, 'jawOpen':0,
#     'jawRight':0, 'mouthClose':0, 'mouthDimpleLeft':0,
#     'mouthDimpleRight':0, 'mouthFrownLeft':0, 'mouthFrownRight':0,
#     'mouthFunnel':0, 'mouthLeft':0, 'mouthLowerDownLeft':0,
#     'mouthLowerDownRight':0, 'mouthPressLeft':0, 'mouthPressRight':0,
#     'mouthPucker':0, 'mouthRight':0, 'mouthRollLower':0,
#     'mouthRollUpper':0, 'mouthShrugLower':0, 'mouthShrugUpper':0,
#     'mouthSmileLeft':0, 'mouthSmileRight':0, 'mouthStretchLeft':0,
#     'mouthStretchRight':0, 'mouthUpperUpLeft':0, 'mouthUpperUpRight':0,
#     'noseSneerLeft':0, 'noseSneerRight':0}
weights={'browDownLeft':1, 'browDownRight':1, 'browInnerUp':0,
    'browOuterUpLeft':0, 'browOuterUpRight':0, 'cheekPuff':0,
    'cheekSquintLeft':0, 'cheekSquintRight':0, 'eyeBlinkLeft':0,
    'eyeBlinkRight':0, 'eyeLookDownLeft':0, 'eyeLookDownRight':0,
    'eyeLookInLeft':0, 'eyeLookInRight':0, 'eyeLookOutLeft':0,
    'eyeLookOutRight':0, 'eyeLookUpLeft':0, 'eyeLookUpRight':0,
    'eyeSquintLeft':1, 'eyeSquintRight':1, 'eyeWideLeft':1,
    'eyeWideRight':1, 'jawForward':0, 'jawLeft':0, 'jawOpen':0,
    'jawRight':0, 'mouthClose':0.15, 'mouthDimpleLeft':0,
    'mouthDimpleRight':0, 'mouthFrownLeft':0, 'mouthFrownRight':0,
    'mouthFunnel':0, 'mouthLeft':0, 'mouthLowerDownLeft':0,
    'mouthLowerDownRight':0, 'mouthPressLeft':0, 'mouthPressRight':0,
    'mouthPucker':0.5, 'mouthRight':0, 'mouthRollLower':0,
    'mouthRollUpper':0, 'mouthShrugLower':0, 'mouthShrugUpper':0,
    'mouthSmileLeft':0, 'mouthSmileRight':0, 'mouthStretchLeft':0,
    'mouthStretchRight':0, 'mouthUpperUpLeft':0, 'mouthUpperUpRight':0,
    'noseSneerLeft':0.6, 'noseSneerRight':0.6}
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
# output_path = '/home/china/zxwork/ARkit_edit-emoposetalk/Arkit_3Dmesh/AU/emo/disgusted03.npy' # def 1

# userstudy
output_path = '/home/china/zxwork/ARkit_edit-emoposetalk/Arkit_3Dmesh/AU/emoUS/angry-US.npy' # def 1


np.save(output_path,combined_delta)

# 渲染
temp = np.load('/home/china/zxwork/ARkit_edit-emoposetalk/AU/FLAME_template.npy')
final_vertices = temp + combined_delta
final_vertices = final_vertices.reshape(1, 5023, 3)
image_path = '/home/china/zxwork/ARkit_edit-emoposetalk/Arkit_3Dmesh/image/emoUS/angry-US.png'  # def 2
template_file = '/home/china/zxwork/FLAME_PyTorch-master/model/FLAME_sample.ply'
template = Mesh(filename=template_file)
img=render_sequence_meshes(final_vertices, template)
# 选择对应的图片路径
cv2.imwrite(image_path,img)