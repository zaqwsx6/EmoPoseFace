# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 11:32:10 2024

@author: zx

conda env : FLAMEPY trimesh==3.6 run err
conda env : FLAMEedit trimesh==4.6.4 run success
"""

import tkinter as tk
from PIL import Image, ImageTk
import os, shutil
import cv2
import tempfile
import numpy as np
from subprocess import call
import argparse
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa' #egl、osmesa
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa' #egl、osmesa
import pyrender
import trimesh
from psbody.mesh import Mesh


'''arkit 52 blendshape Visual control panel'''

def add_text_to_image(img, text):
    # setup text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = text

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]

    # get coords based on boundary
    textX = (img.shape[1] - textsize[0]) // 2
    textY = img.shape[0]- (img.shape[0] + textsize[1]) // 16

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
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
    
    intensity = 2.0

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 
            )


    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f)

    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)


    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')

        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def render_sequence_meshes( sequence_vertices, template,):
    num_frames = sequence_vertices.shape[0]
    center = np.mean(sequence_vertices[0], axis=0)
    for i_frame in range(num_frames):
        render_mesh = Mesh(sequence_vertices[i_frame], template.f)
        pred_img = render_mesh_helper(render_mesh, center, tex_img=None)
        pred_img = pred_img.astype(np.uint8)
    return pred_img
    
def update_image():
    role_index = role.get()
    if role_index == 0 :
        
        # temp = np.load('/home/lh/Data/lihao/AU32/0/AU1.npy')[0]
        temp = np.load('/home/china/zxwork/ARkit_edit-emoposetalk/AU/FLAME_template.npy')
        # temp = np.load('/home/china/zxwork/ARkit_edit-emoposetalk/AU/AU1.npy')
        # temp = temp[0]

    if role_index ==1 :
        
        # temp = np.load('/home/lh/Data/lihao/AU32/2/AU1.npy')[0]
        temp = np.load('/home/china/zxwork/ARkit_edit-emoposetalk/AU/AU1.npy')[0]

    if role_index ==2 :
       
        # temp = np.load('/home/lh/Data/lihao/AU32/7/AU1.npy')[0]
        temp = np.load('/home/china/zxwork/ARkit_edit-emoposetalk/AU/AU1.npy')[0]

    """根据滑动条选择加载不同的图片"""
    # image_path = '/home/lh/lihao/AU/LDMau-main/img1.png'
    image_path = '/home/china/zxwork/FLAME_PyTorch-master/ARkit_edit-emoposetalk/image/img.png'
    vertices_out = temp
    index_dif = 0
    # for i in range(5):
    # todo 01
    for i in range(6):
        # 前5列每列10个滑块，第6列1个滑块
        num_sliders = 10 if i < 5 else 1
        for j in range(num_sliders):
            cont = sliders[i][j].get()
            vertices_out = vertices_out + cont*diff[index_dif:index_dif+1]/100
            index_dif = index_dif+1
    img=render_sequence_meshes(vertices_out, template)
    # 选择对应的图片路径
    cv2.imwrite(image_path,img)
    
        # 使用 Pillow 加载图片
    image = Image.open(image_path)
    image = image.resize((300, 300))  # 调整图片大小
    photo = ImageTk.PhotoImage(image)

        # 保持对图片对象的引用，防止被垃圾回收
       
    img_lable.config(image=photo)
    img_lable.image=photo
        # 更新 Canvas 中的图片
        #canvas.create_image(200, 200, image=root.photo)


def auto_update_image():
    update_image() # 调用更新图片的函数
    root.after(33, auto_update_image) # 每33毫秒再次调用，相当于每秒30次

# def reset_sliders():
#     for frame_sliders in sliders:
#          for slider in frame_sliders:
#             slider.set(0) # 将滑动条的值设置为0 update_image() # 重置后更新图片
#     update_image()
def reset_sliders():
    for i in range(6):  # 将5改为6
        num_sliders = 10 if i < 5 else 1
        for j in range(num_sliders):
            sliders[i][j].set(0)
    update_image()

#获取不同arkit au的mesh点偏移
# 读取OBJ文件
# temp_mesh = trimesh.load_mesh('/home/lh/Data/lihao/bs/Basis.obj')
temp_mesh = trimesh.load_mesh('/home/china/zxwork/FLAME_PyTorch-master/ARkit_edit-emoposetalk/bs/Basis.obj')
# 获取顶点数据
temp = temp_mesh.vertices
lable = []
# data_path = '/home/lh/Data/lihao/bs/exp'
data_path = '/home/china/zxwork/FLAME_PyTorch-master/ARkit_edit-emoposetalk/bs/exp'
index = 0 
vertices = np.zeros((51,5023,3))
for file in sorted(os.listdir(data_path)):
    data_file_path = os.path.join(data_path,file)
    # mesh = trimesh.load_mesh(data_file_path)
    mesh = trimesh.load_mesh(data_file_path)
    vertices[index] = mesh.vertices
    index = index + 1
    lable.append(file[:-4])

# template_file = '/home/lh/lihao/AU/LDMau-main/dataset/FLAME_sample.ply'
template_file = '/home/china/zxwork/FLAME_PyTorch-master/model/FLAME_sample.ply'
template = Mesh(filename=template_file)
#predicted_vertices = np.load('/home/lh/Data/lihao/AU32/54/AU_vertice32.npy')
#predicted_vertices = np.reshape(predicted_vertices,(-1,5023,3))

diff = vertices - temp

# 创建主窗口
root = tk.Tk()
root.title("控制面板 - 显示不同图片")

# 创建Canvas用于显示图片
#canvas = tk.Canvas(root, width=400, height=400, bg="white")
#canvas.pack()
frames =[]
sliders = []
lable_index = 0
# for group in range(5):
for group in range(6):
    frame = tk.Frame(root,borderwidth=2,relief='groove')
    frame.grid(row=0,column=group,padx=10,pady=10)
    frames.append(frame)
    frame_sliders = []
    # todo 01
    # 前5列每列10个滑块，第6列1个滑块
    num_sliders = 10 if group < 5 else 1  # 前5列10个，第6列1个
    for j in range(num_sliders):
        slider = tk.Scale(frame, from_=0,to =100 ,orient='horizontal',label=lable[lable_index],command=lambda _:update_image,length=150)
        lable_index = lable_index + 1
        slider.grid(row=j,column=0,padx=5,pady=5)
        frame_sliders.append(slider)
    sliders.append(frame_sliders)
# 按钮点击显示当前图片
img_lable = tk.Label(root)
# img_lable.grid(row=1, column=0,columnspan=4, pady=10)
img_lable.grid(row=1, column=0,columnspan=6, pady=10)

role = tk.Scale(root, from_=0,to =2 ,orient='horizontal',label='role',command=lambda _:update_image)
role.grid(row=2,column=3,columnspan=2,pady=10)

button = tk.Button(root, text="显示图片", command=update_image)
# button.grid(row=2,column=0,columnspan=2,pady=10)
button.grid(row=2, column=0, columnspan=3, pady=10)  # 调整列跨度
role.grid(row=2, column=5, columnspan=2, pady=10)  # 调整角色滑块位置

reset_button = tk.Button(root, text="Reset Sliders", command=reset_sliders) 
# reset_button.grid(row=1, column=2, columnspan=4, pady=10)
reset_button.grid(row=2, column=3, columnspan=3, pady=10)  # 调整列位置

# 初始化Canvas内容
update_image()
auto_update_image()
# 启动主循环
root.mainloop()