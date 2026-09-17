"""
Demo code to load the FLAME Layer and visualise the 3D landmarks on the Face

Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.

For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""

import os
import cv2
import numpy as np
import pyrender
import torch
import trimesh
import matplotlib.pyplot as plt
import tempfile
from tqdm import tqdm
from subprocess import call
import pickle

from flame_pytorch import FLAME, get_config

from psbody.mesh import Mesh
'''
pose mode 6loss 预测结果渲染 区别是15072维3dmesh exp（ ：15069）flame pose（15070：15072）
(***with pose***)
mead 数据
flame pose + 3Dmesh expression
3Dmesh+audio 直接渲染
批量渲染
        修改3处path：
                         01 data_list（3dmesh）
                         02 np path（3dmesh）
                         03 output_path（render）
                         04 file_name_wav（audio）
'''

config = get_config()
radian = np.pi / 180.0
flamelayer = FLAME(config)

# Creating a batch of mean shapes
shape_params = torch.zeros(1, 100).cuda()

# Creating a batch of different global poses
# pose_params_numpy[:, :3] : global rotaation
# pose_params_numpy[:, 3:] : jaw rotaation
pose_params_numpy = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    dtype=np.float32,
)
pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()

# Cerating a batch of neutral expressions
expression_params = torch.zeros(1, 50, dtype=torch.float32).cuda()
flamelayer.cuda()
vertice, landmark = flamelayer(shape_params, expression_params, pose_params)

print(shape_params.shape, expression_params.shape, pose_params.shape)


def main():
    # data_list = '/data/WX/FLAME_PyTorch/FDM/test'
    # data_list = '/home/china/Databak/FDM/testmead/flame/'
    # data_list = '/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss/3dmesh_flame02/'
    # data_list = '/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss/3dmesh_flame_lossset04_posestyle02/'
    # data_list = '/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss/3dmesh_flame_lossset04/'

    # data_list = '/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss_2gru/3dmesh_flame_posestyle_2gru/'
    # data_list = '/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss_2gru/3dmesh_flame_posestyle02_2gru_noKF/'

    # emposetalk render
    # data_list = '/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss_2gru/3dmesh_flame_E-facediff02/' #E-facediff
    # data_list = '/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss_2gru/3dmesh_flame_lossset17/' #lossset17

    # data_list = '/media/china/DATA1/Result-work2/exp_result/exp_pose_talk/3d_flame_lossset17/' #lossset17
    # data_list = '/media/china/DATA1/Result-work2/exp_result/exp_pose_talk/3d_flame_lossset04/' #lossset04
    data_list = '/media/china/DATA1/Result-work2/exp_result/exp_pose_talk/3d_flame_E-facediff02/' #E-facediff02

    list = os.listdir(data_list)

    for data in list:

        my_data = np.load(os.path.join(data_list, data))

        # express = torch.tensor(my_data[:,:50,:,:].reshape((550, -1))).cuda()
        # pose = torch.tensor(my_data[:,50:,:,:].reshape((550, -1))).cuda()
        # print('my_data.shape: ', my_data.shape)
        # express = torch.tensor(my_data['expression']).cuda()

        # pose = torch.tensor(my_data['pose'][:,3:]).cuda() #取后三维度 下颚
        # pose = torch.cat([torch.zeros_like(pose), pose], dim=1)

        # pose = torch.tensor(my_data['pose'][:,:]).cuda() # 取全部维度 头部姿势 + 下颚

        pose = torch.tensor(my_data[:, -3:]).cuda()  # 取头部姿势(1-3)  下颚置0(4-6)
        pose = torch.cat([pose, torch.zeros_like(pose)], dim=1)#

        # pose = torch.tensor(my_data['pose']).cuda()
        # express = torch.tensor(my_data[:,:50]).cuda()
        # pose = torch.tensor(my_data[:,50:]).cuda()
        len = pose.shape[0]
        express = torch.zeros(len, 50).cuda()
        shape = torch.zeros(len, 100).cuda()

        # pose_params_numpy = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],dtype=np.float32,)
        # pose_params_numpy = np.zeros((len,8),dtype=np.float32)
        # pose = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()
        # pose = torch.zeros(len, 6).cuda()

        print('shape: ', shape.size())
        print('express: ', express.shape)
        print('pose: ', pose.shape)

        # pose = torch.cat((pose, torch.zeros_like(pose)), dim=1)
        out = []
        for i in range(len):
            # Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework
            # vertice, landmark = flamelayer(pose_params,
            #                                expression_params,
            #                                pose_params)  # For RingNet project
            # vertice, landmark = flamelayer(shape_params=shape[i,...].unsqueeze(0),
            #                             expression_params=express[i,...].unsqueeze(0),
            #                             pose_params=pose)
            vertice, landmark = flamelayer(shape_params=shape[i,...].unsqueeze(0),
                                        expression_params=express[i,...].unsqueeze(0),
                                        pose_params=pose[i,...].unsqueeze(0))  # For RingNet project

            print(vertice.size(), landmark.size())
            out.append(vertice.squeeze(0).cpu().numpy())

        out = np.array(out)

        print('out.shape: ', out.shape)

        '''
        以下是渲染代码
        '''
        print('开始渲染结果...')

        # template_file = os.path.join('/data/WX/BIWI_dataset/templates/FLAME_sample.ply')
        template_file = os.path.join('/home/china/zxwork/FLAME_PyTorch-master/model/FLAME_sample.ply')
        print("rendering: ", data)
        print('template_file: ', template_file)

        template = Mesh(filename=template_file)

        # predicted_vertices = out

        # print("predicted_vertices.shape", predicted_vertices.shape)
        #
        # # np.save(f'/data/WX/FLAME_PyTorch/FDM/test/{data[:-4]}.npy', predicted_vertices)
        # # np.save(f'/home/china/Databak/FDM/testmead/3dmesh/{data[:-4]}.npy', predicted_vertices) #mesh 存储路径
        # np.save(f'/home/china/Databak/FDM/testmead/3dmesh_pose/{data[:-4]}.npy', predicted_vertices)

        # predicted_vertices = np.reshape(predicted_vertices,(-1,args.vertice_dim//3,3))
        # predicted_vertices_path = '/data/WX/FLAME_PyTorch/FDM/test/' + data
        # output_path = '/data/WX/FLAME_PyTorch/FDM/test/render'
        # predicted_vertices=np.load(f'/home/china/Databak/FDM/testmead/3dmesh_pose_datatest/{data[:-4]}.npy')
        # predicted_vertices=np.load(f'/home/china/Databak/FDM/testmead/3dmesh_pose_datatest/kf_deal/{data[:-4]}.npy')
        # predicted_vertices=np.load(f'/home/china/Databak/FDM/testmead/3dmesh_pose_datatest/kf_deal/{data[:-4]}.npy')
        # face_mesh=np.load(f'/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss/3dmesh_flame01/{data[:-4]}.npy')
        mesh_template = np.load('/home/china/zxwork/FaceDiffuser-main/mead/FLAME_template.npy')
        # template = np.zeros(self.args.vertice_dim)# load oniy flame
        mesh_template = mesh_template.reshape((-1))
        # mesh_template = torch.FloatTensor(mesh_template)

        face_mesh = my_data[:, :15069] - mesh_template # ***face mesh add
        face_mesh = np.reshape(face_mesh, (-1, 15069 // 3, 3))  # ***add
        pose_mesh=out
        predicted_vertices = face_mesh + pose_mesh
        # predicted_vertices=pose_mesh
        print("predicted_vertices.shape", predicted_vertices.shape)
        # np.save(os.path.join('/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss/3dmesh_flame_mesh02', data), predicted_vertices)
        # np.save(os.path.join('/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss/3dmesh_flame_mesh_lossset04_posestyle02', data), predicted_vertices)
        # np.save(os.path.join('/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss/3dmesh_flame_mesh_lossset04', data), predicted_vertices)

        # np.save(os.path.join('/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss_2gru/3dmesh_posestyle_2gru', data), predicted_vertices)
        # np.save(os.path.join('/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss_2gru/3dmesh_posestyle02_2gru_noKF', data), predicted_vertices)

        #  emoposetalk render
        # np.save(os.path.join('/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss_2gru/3dmesh_E-facediff02', data), predicted_vertices)
        # np.save(os.path.join('/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss_2gru/3dmesh_lossset17', data), predicted_vertices)
        # np.save(os.path.join('/media/china/DATA1/Result-work2/exp_result/exp_pose_talk/3dmesh_lossset17', data), predicted_vertices)
        # np.save(os.path.join('/media/china/DATA1/Result-work2/exp_result/exp_pose_talk/3dmesh_lossset04', data), predicted_vertices)
        np.save(os.path.join('/media/china/DATA1/Result-work2/exp_result/exp_pose_talk/3dmesh_E-facediff02', data), predicted_vertices)

        predicted_vertices_path = data_list + data
        # output_path = '/home/china/Databak/FDM/testmead/render1' #渲染视频存储路径
        # output_path = '/home/china/Databak/FDM/testmead/render_pose'
        # output_path = '/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss/render_pose02'
        # output_path = '/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss/render_pose_lossset04_posestyle02'
        # output_path = '/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss/render_pose_lossset04'

        # output_path = '/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss_2gru/render_posestyle_2gru'
        # output_path = '/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss_2gru/render_posestyle02_2gru_noKF'

        #lunwen shiyan
        # output_path = '/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss_2gru/render_E-facediff02'
        # output_path = '/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss_2gru/render_lossset17'

        # output_path = '/media/china/DATA1/Result-work2/exp_result/exp_pose_talk/render_lossset17' # best-loss-017
        # output_path = '/media/china/DATA1/Result-work2/exp_result/exp_pose_talk/render_lossset04' # loss04
        output_path = '/media/china/DATA1/Result-work2/exp_result/exp_pose_talk/render_E-facediff02' # E-facediff02



        if not os.path.exists(output_path):
            os.makedirs(output_path)    

        render_sequence_meshes(predicted_vertices, template, output_path, predicted_vertices_path, vt=None, ft=None, tex_img=None)

def render_sequence_meshes(sequence_vertices, template, out_path, predicted_vertices_path, vt, ft, tex_img):

    # sequence_vertices.shape #(126, 5023, 3)

    num_frames = sequence_vertices.shape[0] #126
    file_name_pred = predicted_vertices_path.split('/')[-1].split('.')[0]
    # file_name_wav = os.path.join('/home/china/Databak/FDM/testmead/audio/', file_name_pred.split('_0')[0]+'.wav')
    # file_name_wav = os.path.join('/media/china/DATA1/Result-work2/exp_result/mesh_flame_gru_6loss/audio/', file_name_pred.split('_0')[0]+'.wav')

    file_name_wav = os.path.join('/media/china/DATA1/Result-work2/exp_result/exp_pose_talk/audio/', file_name_pred.split('_0')[0]+'.wav')

    tmp_video_file_pred = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
    # writer_pred = cv2.VideoWriter(tmp_video_file_pred.name, cv2.VideoWriter_fourcc(*'mp4v'), 25, (800, 800), True)
    writer_pred = cv2.VideoWriter(tmp_video_file_pred.name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (800, 800), True)

    center = np.mean(sequence_vertices[0], axis=0)
    video_fname_pred = os.path.join(out_path, file_name_pred+'.mp4')
    for i_frame in tqdm(range(num_frames)):
        # print("sequence_vertices[i_frame]",sequence_vertices[i_frame])
        # print("sequence_vertices[i_frame].shape", sequence_vertices[i_frame].shape) #(5023, 3)

        render_mesh = Mesh(sequence_vertices[i_frame], template.f)
        # render_mesh.write_obj(os.path.join(out_path, file_name_pred + '.obj'))
        # if(i_frame==num_frames-1):
        #     render_mesh.write_obj(os.path.join(out_path, file_name_pred+'.obj'))
        if vt is not None and ft is not None:
            render_mesh.vt, render_mesh.ft = vt, ft
        pred_img = render_mesh_helper(render_mesh, center, tex_img=tex_img)
        pred_img = pred_img.astype(np.uint8)
        img = pred_img
        writer_pred.write(img) #把图片资源写入视频中，< cv2.VideoWriter 0x7f66c8a0ea30>
    
    writer_pred.release() #释放资源'''
    cmd = ('ffmpeg' + ' -i {0} -pix_fmt yuv420p -qscale 0 {1}'.format(
       tmp_video_file_pred.name, video_fname_pred)).split()
    call(cmd)

    # file_name_wav = '/data/WX/FLAME_PyTorch/FDM/002.wav'
    # file_name_wav = '/home/china/Databak/FDM/test/audio/M003-angry-level_3-002.wav'
    # render with audio
    cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -qscale 0 {2}'.format(
       file_name_wav, video_fname_pred, video_fname_pred.replace('.mp4', '_audio.mp4'))).split()
    call(cmd)

    if os.path.exists(video_fname_pred):
        os.remove(video_fname_pred)


def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None,  z_offset=0):
    camera_params = {'c': np.array([400, 400]),
                        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                        'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center

    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8,
                roughnessFactor=0.8
            )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)

    # if args.background_black:
    #     scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])#[0, 0, 0] black,[255, 255, 255] white
    # else:
    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])#[0, 0, 0] black,[255, 255, 255] white

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

if __name__ == '__main__':
    main()