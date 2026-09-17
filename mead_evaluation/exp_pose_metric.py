import sys
sys.path.append('.')
import numpy as np
import torch
import argparse
import os
import time

# from FLAME_PyTorch.FLAME import FLAME
# from FLAME_PyTorch.config import get_config
# from utiles.flame_utils import torch2mesh

from flame_pytorch.flame import FLAME
from flame_pytorch.config import get_config
# import torch2mesh


# def smooth_sequence(sequence, window_size):
#     smoothed_sequence = np.zeros_like(sequence)
#     for i in range(sequence.shape[0]):
#         start = max(0, i - window_size // 2)
#         end = min(sequence.shape[0], i + window_size // 2 + 1)
#         smoothed_sequence[i] = np.mean(sequence[start:end], axis=0)
#     return smoothed_sequence
'''
评估**** exp + pose ****数据的 面部表情评估指标
'''

def main():
    dev = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    # parser.add_argument("--pred_path", type=str, default="/data/WX/fdm/other_result_mead/EMOTE/npy")
    # parser.add_argument("--gt_path", type=str, default="/data/WX/MEAD/FLAME_ALL")
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead21-3740-1000g512")
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead21-3740-500")
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead3740-transform0-temp1-1000f512-l8-epoch100")
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead3740-transformDIT_XL-1000f512epoch150")
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead3740-transform2cross-1000f512-l4-8-epoch100")

    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead3740-transform2cross-1000f512-l4-8-epoch50")
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset06") #train data
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset03") #错误train data
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset02") #**较好
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-1000f256_epoch50-facediff01_noKF") #**E-facediff
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-1000f256_epoch50-facediff01_KFdeal") #**E-facediff
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-500f512_epoch50-facediff02_noKF") #**E-facediff

    parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset04_withstyle02") #**较好
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset04_posestyle02") #Abla single gru
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-Abla01_posestyle_2gru_noKF") #Abla 01 1+5+0.001+000
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-Abla02_posestyle_2gru_noKF") #Abla 01 1+0+0 0.001+0+0

    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch100-lossset04_posestyle_2gru_noKF") #

    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch100-lossset04_posestyle_2gru_noKF") #

    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset17_posestyle_2gru_noKF") #

    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-init01")

    # parser.add_argument("--pred_path", type=str, default="/home/china/Databak/emodiff-result/faceformer-emo")

    parser.add_argument("--gt_path", type=str, default="/home/china/Databak/MEAD/MEADL3_FLAME_MESH")
    # parser.add_argument("--gt_path", type=str, default="/home/china/Databak/MEAD/MEADL3_FLAME_MESH_KFdeal_POSE")

    parser.add_argument("--region_path", type=str, default="BIWI/regions/")
    parser.add_argument("--templates_path", type=str, default="BIWI/templates.pkl")
    parser.add_argument("--templates_mead_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/mead/FLAME_template.npy")#mead template
    args = parser.parse_args()

    # todo flametomesh 需要
    # flame_config = get_config()
    # flame = FLAME(flame_config)  # 加载FLAME模型
    # flame.eval()
    # flame.to(dev)

    # with open(args.templates_path, 'rb') as fin:
    #     templates = pickle.load(fin,encoding='latin1')

    # with open(os.path.join(args.region_path, "lve.txt")) as f:
    #     maps = f.read().split(", ")
    #     mouth_map = [int(i) for i in maps]

    # with open(os.path.join(args.region_path, "fdd.txt")) as f:
    #     maps = f.read().split(", ")
    #     upper_map = [int(i) for i in maps]

    # face_vertex_path = '/data/WX/fdm/region/face_vertices.npy'
    face_vertex_path = '/home/china/zxwork/FLAME_PyTorch-master/mead_metric/face_vertices.npy'
    face_vertex = np.load(face_vertex_path)

    # lip_vertex_path = '/data/WX/fdm/region/lip_vertices.npy'
    lip_vertex_path = '/home/china/zxwork/FLAME_PyTorch-master/mead_metric/lip_vertices.npy'
    lip_vertex = np.load(lip_vertex_path)

    # emotion_vertex_path = '/data/WX/fdm/region/emotion_vertices.npy'
    emotion_vertex_path = '/home/china/zxwork/FLAME_PyTorch-master/mead_metric/emotion_vertices.npy'
    emotion_vertex = np.load(emotion_vertex_path)
    

    cnt = 0
    vertices_gt_all = []
    vertices_pred_all = []
    motion_std_difference = []

    pred_list = os.listdir(args.pred_path)  # 读取所有的预测结果

    for pred in pred_list:
        # if 'angry' in pred:
        # pred_path = pred.split('_ConditionEmotion_')[0] + '.npy'
        pred_path = pred.split('_condition_')[0] + '.npy'
        print('Processing {}'.format(pred))
        # gt_name = pred.replace('_', '-')[:-10] + '_' + pred.replace('_', '-')[-9:-4] + '.npy' #获取GT name
        gt_name = pred.split('_condition_')[0] + '.npy' #获取GT name
        # if not os.path.exists(os.path.join(args.gt_path, pred.split('_')[0], gt_name)):
        if not os.path.exists(os.path.join(args.gt_path, pred.split('-')[0], gt_name)):
            print('GT not found')
            continue
        # flame_gt = np.load(os.path.join(args.gt_path, pred_path.split('_')[0], gt_name))
        # a=pred_path.split('-')[0]
        flamemesh_gt = np.load(os.path.join(args.gt_path, pred_path.split('-')[0], gt_name)) #GT MESH PATH
        # expression = torch.from_numpy(flame_gt['expression'])  # (T, 50)
        # pose = torch.from_numpy(flame_gt['pose'])[:, 3:]  # (T, 3)
        # pose = torch.cat([torch.zeros_like(pose), pose], dim=1)  # (T, 6)])

        # vertices_pred = torch.from_numpy(np.load(os.path.join(args.pred_path, pred)).reshape(-1, 5023, 3)).to(dev)  # frame*5023*3
        # vertices_gt = torch2mesh(flame, expression.to(dev), pose.to(dev)).reshape(-1, 5023, 3)  # frame*5023*3
        vertices_pred = np.load(os.path.join(args.pred_path, pred)) # frame*15069
        vertices_pred=vertices_pred[:, :15069]

        vertices_pred = vertices_pred.reshape((-1, 5023, 3))# frame*5023*3
        vertices_gt = flamemesh_gt.reshape(-1, 5023, 3)  # frame*5023*3

        max_frame = min(vertices_pred.shape[0], vertices_gt.shape[0])
        vertices_pred = vertices_pred[:max_frame]
        vertices_gt = vertices_gt[:max_frame]

        # template = torch2mesh(flame, torch.zeros((1,1,50)).to(dev), torch.zeros((1,1,6)).to(dev)).reshape(-1, 5023, 3)  # 1*5023*3
        # template = np.load('/home/china/zxwork/FaceDiffuser-main/mead/FLAME_template.npy')
        template = np.load(args.templates_mead_path)
        template = template.reshape(-1, 5023, 3)


        motion_gt = vertices_gt - template
        motion_pred = vertices_pred - template
        # vertices_gt = vertices_gt.cpu().numpy()
        # vertices_pred = vertices_pred.cpu().numpy()

        vertices_gt_all.extend(list(vertices_gt))
        vertices_pred_all.extend(list(vertices_pred))

        # motion_gt = motion_gt.cpu().numpy()
        # motion_pred = motion_pred.cpu().numpy()

        L2_dis_upper = np.array([np.square(motion_gt[:,v, :]) for v in emotion_vertex])
        L2_dis_upper = np.transpose(L2_dis_upper, (1,0,2))
        L2_dis_upper = np.sum(L2_dis_upper,axis=2)
        L2_dis_upper = np.std(L2_dis_upper, axis=0)
        gt_motion_std = np.mean(L2_dis_upper)

        L2_dis_upper = np.array([np.square(motion_pred[:,v, :]) for v in emotion_vertex])
        L2_dis_upper = np.transpose(L2_dis_upper, (1,0,2))
        L2_dis_upper = np.sum(L2_dis_upper,axis=2)
        L2_dis_upper = np.std(L2_dis_upper, axis=0)
        pred_motion_std = np.mean(L2_dis_upper)

        motion_std_difference.append(gt_motion_std - pred_motion_std)

    print('Frame Number: {}'.format(cnt))

    vertices_gt_all = np.array(vertices_gt_all)
    vertices_pred_all = np.array(vertices_pred_all)

    L2_dis_face_max = np.array([np.square(vertices_gt_all[:,v, :]-vertices_pred_all[:,v,:]) for v in face_vertex])
    L2_dis_face_max = np.transpose(L2_dis_face_max, (1,0,2))
    L2_dis_face_max = np.sum(L2_dis_face_max,axis=2)  # 三个方向的和加起来

    L2_dis_face_mean = np.mean(L2_dis_face_max,axis=1) # 平均值
    L2_dis_face_max = np.max(L2_dis_face_max,axis=1)  # 每一帧的最大值

    L2_dis_lip_max = np.array([np.square(vertices_gt_all[:,v, :]-vertices_pred_all[:,v,:]) for v in lip_vertex])
    L2_dis_lip_max = np.transpose(L2_dis_lip_max, (1,0,2))
    L2_dis_lip_max = np.sum(L2_dis_lip_max,axis=2)  # 三个方向的和加起来

    L2_dis_lip_mean=np.mean(L2_dis_lip_max,axis=1)  # 平均值
    L2_dis_lip_max = np.max(L2_dis_lip_max,axis=1)  # 每一帧的最大值

    L2_dis_all_max = np.array(np.square(vertices_gt_all-vertices_pred_all))
    L2_dis_all_max = np.transpose(L2_dis_all_max, (1,0,2))
    L2_dis_all_max = np.sum(L2_dis_all_max,axis=2)  # 三个方向的和加起来
    L2_dis_all_max = np.max(L2_dis_all_max,axis=1)  # 每一帧的最大值

    L2_dis_emotion_max = np.array([np.square(vertices_gt_all[:,v, :]-vertices_pred_all[:,v,:]) for v in emotion_vertex])
    L2_dis_emotion_max = np.transpose(L2_dis_emotion_max, (1,0,2))
    L2_dis_emotion_max = np.sum(L2_dis_emotion_max,axis=2)  # 三个方向的和加起来
    L2_dis_emotion_max = np.mean(L2_dis_emotion_max,axis=1)  # 每一帧的最大值

    print('Face Vertex Error (FVE): {:.4e}'.format(np.mean(L2_dis_face_max)))
    print('Lip Vertex Error (LVE): {:.4e}'.format(np.mean(L2_dis_lip_max)))
    print('Emotion Vertex Error (EVE): {:.4e}'.format(np.mean(L2_dis_emotion_max)))
    print('All Vertex Error: {:.4e}'.format(np.mean(L2_dis_all_max)))

    print('EDD: {:.5e}'.format(abs(sum(motion_std_difference)/len(motion_std_difference))))

    print('Face Vertex Error (FVE_mean): {:.4e}'.format(np.mean(L2_dis_face_mean)))
    print('Lip Vertex Error (LVE_mean): {:.4e}'.format(np.mean(L2_dis_lip_mean)))
    
if __name__=="__main__":
    main()