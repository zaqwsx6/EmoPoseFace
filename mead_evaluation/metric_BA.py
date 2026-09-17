'''
Work2
评估头部姿态的节拍运动
'''

import sys
sys.path.append('.')
import numpy as np
import torch
import argparse
import os
import time

import numpy as np
import os
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema


def get_flame(flame_path: str):  # only with pose [3:]
    data = np.load(flame_path, allow_pickle=True)

    # expression = torch.from_numpy(data['expression'])  # (T, 50)

    pose = np.array(data['pose'])[:, :3]  # (T, 3) only with 头部姿势

    # pose = torch.from_numpy(data['pose'])[:, 3:]  # (T, 3) no 头部姿势 后3为下鄂
    # pose = torch.cat([torch.zeros_like(pose), pose], dim=1)  # (T, 6)])

    # pose = torch.from_numpy(data['pose'])[:, :]  # (T, 3) 带有头部姿势
    # return torch.cat([expression, pose], dim=1)  # (T, 56)
    return pose  # (T, 3)

"""计算BA"""
def extract_head_motion(keypoints):
    """Extract head motion trajectory (N,3) from FLAME parameters (N, 15072)."""
    return keypoints[:, -3:]  # Last 3 columns contain (x, y, z) head position

def calc_head_beats(head_motion, smooth_window=5):
    """Calculate beats from head motion (N,3)."""
    velocity = np.sqrt(np.sum((head_motion[1:] - head_motion[:-1]) ** 2, axis=1))  # (N-1,)
    velocity = G(velocity, smooth_window)  # Smoothing
    beats = argrelextrema(velocity, np.less)[0]  # Local minima as beats

    print(f"Velocity range: {np.min(velocity):.4f} - {np.max(velocity):.4f}, beats found: {len(beats)}")
    return beats, len(velocity)

def BA(gt_beats, pred_beats):
    """Beat Alignment between GT head beats and predicted head beats."""
    if len(pred_beats) == 0 or len(gt_beats) == 0:  # 如果预测节拍或GT节拍为空，返回0
        return 0.0

    ba = 0
    for bb in gt_beats:
        ba += np.exp(-np.min((pred_beats - bb) ** 2) / 2 / 9)  # Gaussian penalty
    return ba / len(gt_beats)

def calc_ba_score(gt_root, pred_root):
    """Compute BA score between GT and predicted head beats."""
    ba_scores = []
    for pkl in os.listdir(pred_root):
        if os.path.isdir(os.path.join(pred_root, pkl)):
            continue

        # Load predicted motion (N, 15072)
        pred_data = np.load(os.path.join(pred_root, pkl), allow_pickle=True).item()
        pred_keypoints = pred_data['pred_position']  # Shape (N, 15072)
        pred_head = extract_head_motion(pred_keypoints)  # (N, 3)

        # Load GT motion (N, 15072)
        gt_data = np.load(os.path.join(gt_root, pkl), allow_pickle=True).item()
        gt_keypoints = gt_data['gt_position']  # Shape (N, 15072)
        gt_head = extract_head_motion(gt_keypoints)  # (N, 3)

        # Compute beats
        gt_beats, _ = calc_head_beats(gt_head)
        pred_beats, _ = calc_head_beats(pred_head)

        ba_scores.append(BA(gt_beats, pred_beats))

    return np.mean(ba_scores)


def calc_frame_displacements(head_motion):
    """
    计算每一帧相对于第一帧的位移均值
    Args:
        head_motion: (N, 3) 的数组，N帧的头部位置（x, y, z）
    Returns:
        mean_displacement: 标量，所有帧相对于第一帧的位移均值
        displacements: (N,) 数组，每一帧相对于第一帧的欧氏距离
    """
    if len(head_motion) == 0:
        return 0.0, np.array([])

    first_frame = head_motion[0]  # 第一帧作为参考点
    displacements = np.sqrt(np.sum((head_motion - first_frame) ** 2, axis=1))  # 欧氏距离
    mean_displacement = np.mean(displacements[1:])  # 排除第一帧自身（距离为0）

    print(
        f"Displacement relative to first frame: "
        f"mean={mean_displacement:.4f}, "
        f"max={np.max(displacements):.4f}, "
        f"min={np.min(displacements[1:]):.4f}"
    )
    return mean_displacement, displacements


def main():
    """Compute BA score between GT and predicted head beats."""
    dev = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    # parser.add_argument("--pred_path", type=str, default="/data/WX/fdm/other_result_mead/EMOTE/npy")
    # parser.add_argument("--gt_path", type=str, default="/data/WX/MEAD/FLAME_ALL")

    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead3740-transform2cross-1000f512-l4-8-epoch50")
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset06") #train data
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset03") #错误train data
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset02") #**较好

    #0.317/0.3275
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-1000f256_epoch50-facediff01_noKF") #**E-facediff0.31
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-500f512_epoch50-facediff02_KFdeal") #**E-facediff0.301/0.315
    # **E-facediff GT0.317/deal0.343  + #0.0366/0.0365
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-1000f256_epoch50-facediff01_KFdeal")

    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset04_withstyle") #**较好
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset04_withstyle") #**较好
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset04_posestyle02") #
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset06_posestyle_2gru_noKF") #0.089

    # ***best***
    parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset17_posestyle_2gru_noKF") #0.322/0.3275
    #Abla
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset04_posestyle02") #0.2933/0.3075
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-Abla02_posestyle_2gru_noKF") #0.2933/0.3075

    # 0.275/0.2877 +  F0.0220/S0.0217
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset04_posestyle_2gru_noKF")


    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset04_posestyle_2gru") #0.280/kf0.2897
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch100-lossset04_posestyle_2gru_noKF") #0.2994/kf0.2988
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset10_posestyle_2gru_noKF") #0.128
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/Facediff-pose/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset04_posestyle_2gru") #no_KF 0.28
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-init01")

    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset05") #0.301/kf0.322
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset02") #0.2969
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset04") #0.2875/ kf0.2899
    # parser.add_argument("--pred_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/result/mead(3Dmesh_exp+pose)-gru2-1000f512-loss6_epoch50-lossset06")

    # parser.add_argument("--pred_path", type=str, default="/home/china/Databak/emodiff-result/faceformer-emo")

    # parser.add_argument("--gt_path", type=str, default="/home/china/Databak/MEAD/MEADL3_FLAME_MESH")
    # parser.add_argument("--gt_path", type=str, default="/home/china/Databak/MEAD/MEADL3_FLAME")
    # parser.add_argument("--gt_path", type=str, default="/home/china/Databak/MEAD/MEADL3_FLAME")
    parser.add_argument("--gt_path", type=str, default="/home/china/Databak/MEAD/MEADL3_FLAME_KFdeal_POSE")


    # parser.add_argument("--region_path", type=str, default="BIWI/regions/")
    # parser.add_argument("--templates_path", type=str, default="BIWI/templates.pkl")
    # parser.add_argument("--templates_mead_path", type=str, default="/home/china/zxwork/FaceDiffuser-main/mead/FLAME_template.npy")#mead template
    args = parser.parse_args()

    cnt = 0
    vertices_gt_all = []
    vertices_pred_all = []
    motion_std_difference = []

    ba_scores = []

    all_means = [] #存储每个序列的均值
    all_displacements = []#存储所有帧的位移

    pred_list = os.listdir(args.pred_path)  # 读取所有的预测结果

    for pred in pred_list:
        # if 'angry' in pred:
        # pred_path = pred.split('_ConditionEmotion_')[0] + '.npy'
        pred_path = pred.split('_condition_')[0] + '.npy'
        print('Processing {}'.format(pred))
        # gt_name = pred.replace('_', '-')[:-10] + '_' + pred.replace('_', '-')[-9:-4] + '.npy' #获取GT name
        gt_name = pred.split('_condition_')[0] + '.npz'  # 获取GT name
        # if not os.path.exists(os.path.join(args.gt_path, pred.split('_')[0], gt_name)):
        gt_path_flame =os.path.join(args.gt_path, pred.split('-')[0], gt_name)
        if not gt_path_flame:
            print('GT not found')
            continue
        # flame_gt = np.load(os.path.join(args.gt_path, pred_path.split('_')[0], gt_name))
        # a=pred_path.split('-')[0]
        # flamemesh_gt = np.load(os.path.join(args.gt_path, pred_path.split('-')[0], gt_name))  # GT MESH PATH
        flame_gt_head = get_flame(gt_path_flame)  # GT flame PATH

        # expression = torch.from_numpy(flame_gt['expression'])  # (T, 50)
        # pose = torch.from_numpy(flame_gt['pose'])[:, 3:]  # (T, 3)
        # pose = torch.cat([torch.zeros_like(pose), pose], dim=1)  # (T, 6)])

        # vertices_pred = torch.from_numpy(np.load(os.path.join(args.pred_path, pred)).reshape(-1, 5023, 3)).to(dev)  # frame*5023*3
        # vertices_gt = torch2mesh(flame, expression.to(dev), pose.to(dev)).reshape(-1, 5023, 3)  # frame*5023*3
        vertices_pred = np.load(os.path.join(args.pred_path, pred))  # frame*15069
        vertices_pred_head = vertices_pred[:, -3:]

        gt_beats, _ = calc_head_beats(flame_gt_head)
        pred_beats, _ = calc_head_beats(vertices_pred_head)

        ba_scores.append(BA(gt_beats, pred_beats))


        #metric02 计算相对于初始帧位移
        mean_disp, displacements = calc_frame_displacements(vertices_pred_head)
        all_means.append(mean_disp)
        all_displacements.extend(displacements[1:])  # 排除每个序列的第一帧

    BA_Score = np.mean(ba_scores)

    # metric02 计算相对于初始帧位移
    seq_mean = np.mean(all_means)
    frame_mean=np.mean(all_displacements)

    print('*BA*: {}'.format(BA_Score))

    print('*frame_mean*:{}'.format(frame_mean))
    print('seq_mean:{}'.format(seq_mean))

    # print('EDD: {:.5e}'.format(abs(sum(motion_std_difference) / len(motion_std_difference))))

    # print('Face Vertex Error (FVE_mean): {:.4e}'.format(1))
    # print('Lip Vertex Error (LVE_mean): {:.4e}'.format(2))


if __name__ == "__main__":
    main()