import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
# from transformers import AutoProcessor
from transformers import Wav2Vec2Processor
import librosa
from torch.nn.utils.rnn import pad_sequence

import pandas as pd

'''
带有头部pose
加载3D MESH + Flame 参数
'''
def reparse_emotion_name(emotion):
    emotion_dict = {
        'angry': ['angry', 'anger'],
        'contempt': ['contempt'],
        'disgusted': ['disgusted', 'disgust'],
        'fear': ['fear', 'fearful'],
        'happy': ['happy', 'happiness'],
        'sad': ['sad', 'sadness'],
        'surprised': ['surprised', 'surprise'],
        'neutral': ['neutral']
    }
    for k, v in emotion_dict.items():
        if emotion in v:
            return k
    else:
        raise ValueError('emotion name {} not found'.format(emotion))


def label_to_idx(label_type: str, label: str, one_hot: bool = False):
    if label_type == 'emotion':
        label = reparse_emotion_name(label)
        if label == 'neutral':
            if one_hot:
                return torch.zeros(7)
            else:
                return -1
        # emotion_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
        # todo
        # emotion_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised', 'neutral']
        emotion_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised'] # 1x7
        assert label in emotion_list, 'emotion label should be in {}, found {}'.format(emotion_list, label)
        if one_hot:
            return torch.eye(len(emotion_list))[emotion_list.index(label)]
        else:
            return emotion_list.index(label)
    elif label_type == 'speaker':
        speaker_list = ['M003', 'M005', 'M007', 'M009', 'M011', 'M012', 'M013', 'M019', 'M022', 'M023',
                        'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 'M030', 'M031', 'M032', 'M033',
                        'M034', 'M035', 'M037', 'M039', 'M040']
        assert label in speaker_list, 'speaker label should be in {}, found {}'.format(speaker_list, label)
    
        if one_hot:
            return torch.eye(len(speaker_list))[speaker_list.index(label)]
        else:
            return speaker_list.index(label)
    
    else:
        raise ValueError('label type {} not found'.format(label_type))

# add args\ param
class MEADDataset(data.Dataset):
    def __init__(self, df, args, is_train: bool, p_general: float = 0.3, read_audio: bool = False):
        super().__init__()
        self.args = args
        # self.dataset_root = '/data/WX/MEAD' #mesh data path
        self.dataset_root = '/home/china/Databak/MEAD'
        self.subjects_dict = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
        # todo
        # self.subjects_dict = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
        self.is_train = is_train
        self.p_general = p_general
        self.read_audio = read_audio
        self.df = df
        # self.df_mesh =df_mesh
        self.one_hot_labels = np.eye(len(self.subjects_dict))

    def __getitem__(self, index ):
        row = self.df.iloc[index]  # [pid, emotion, intensity, flame_id, audio_id]
        if row['intensity'] == 'level_3':
            audio_path = os.path.join(self.dataset_root, 'AUDIO', row['pid'], row['emotion'],
                                    row['intensity'], row['audio_id'])

            #***init 01 and 02  *** without deal data ***
            flame_path = os.path.join(self.dataset_root, 'MEADL3_FLAME', row['pid'],
                                      f"{row['pid']}-{row['emotion']}-{row['intensity']}-{row['flame_id']}")

            Mesh_path = os.path.join(self.dataset_root, 'MEADL3_FLAME_MESH', row['pid'],
                                      f"{row['pid']}-{row['emotion']}-{row['intensity']}-{row['flame_id'][:-4]}.npy")
            #***init 04 05 06
            # flame_path = os.path.join(self.dataset_root, 'MEADL3_FLAME_KFdeal_POSE', row['pid'],
            #                           f"{row['pid']}-{row['emotion']}-{row['intensity']}-{row['flame_id']}")
            #
            # Mesh_path = os.path.join(self.dataset_root, 'MEADL3_FLAME_MESH', row['pid'],
            #                          f"{row['pid']}-{row['emotion']}-{row['intensity']}-{row['flame_id'][:-4]}.npy")
            if self.read_audio:

                processor = Wav2Vec2Processor.from_pretrained(
                    "facebook/hubert-xlarge-ls960-ft")  # HuBERT uses the processor of Wav2Vec 2.0

                # *** 更换 audio deal
                speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
                audio_values = np.squeeze(processor(speech_array, return_tensors="pt", padding="longest",
                                                    sampling_rate=sampling_rate).input_values)
                audio_values=torch.FloatTensor(audio_values)

            file_name = row['pid'] + '_' + row['emotion'] + '_' + row['intensity'] + '_' + row['audio_id']# audio path
            # file_name = row['audio_id']

            # get flame
            # flame = self.get_flame(flame_path) #exp + pose (1,N,56)
            flame = self.get_flame(flame_path)  #pose[3：] （N,3）

            vertice = np.load(Mesh_path, allow_pickle=True)
            vertice = vertice.astype(np.float32)
            vertice = torch.from_numpy(vertice)
            vertice = vertice.reshape(vertice.shape[0], -1)
            flame_3Dmesh = torch.cat([vertice, flame], dim=1) #(N,15069) cat (N,3)——》(N,15072)


            if self.is_train:
                emotion_label = label_to_idx('emotion', row['emotion'], one_hot=True)
            else:
                # emotion_label = self.one_hot_labels
                emotion_label = label_to_idx('emotion', row['emotion'], one_hot=True)

            # vertice = np.load(flame, allow_pickle=True)
            # vertice = vertice[0]
            # template = torch.zeros((1, 56))

            template = np.load('/media/china/solidrepo1/zxwork/EmoPoseFace-main/data/FLAME_template.npy')
            # template = np.zeros(self.args.vertice_dim)# load oniy flame
            template = template.reshape((-1))
            template = torch.FloatTensor(template)

            if self.read_audio:
                # return audio_values, flame, template, emotion_label, file_name, flame_path
                return audio_values, flame_3Dmesh, template, emotion_label, file_name, flame_path
            else:
                return flame_3Dmesh, template, emotion_label, file_name, flame_path

    def __len__(self):
        return len(self.df)

    @staticmethod
    def get_audio(audio_path: str):
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        audio = torch.from_numpy(audio)
        length = torch.tensor(audio.shape[0])
        return audio, length

    @staticmethod
    def get_flame(flame_path: str): # only with pose [3:]
        data = np.load(flame_path, allow_pickle=True)

        # expression = torch.from_numpy(data['expression'])  # (T, 50)

        pose = torch.from_numpy(data['pose'])[:, :3]  # (T, 3) only with HEAD POSE

        # pose = torch.from_numpy(data['pose'])[:, 3:]  # (T, 3) no HEAD POSE
        # pose = torch.cat([torch.zeros_like(pose), pose], dim=1)  # (T, 6)])

        # pose = torch.from_numpy(data['pose'])[:, :]  # (T, 3) With HEAD POSE
        # return torch.cat([expression, pose], dim=1)  # (T, 56)
        return pose  # (T, 3)


def padding_collate_fn(batch):
    batch_audio_list = [item[0] for item in batch]
    batch_motion_list = [item[1] for item in batch]
    batch_template_list = [item[2] for item in batch]
    batch_onehot_list = [item[3] for item in batch]
    # batch_filename_list = [item[4] for item in batch]
    padding_audio = pad_sequence(batch_audio_list, batch_first=True, padding_value=0)
    padding_motion = pad_sequence(batch_motion_list, batch_first=True, padding_value=0)
    padding_template = pad_sequence(batch_template_list, batch_first=True, padding_value=0)
    padding_onehot = pad_sequence(batch_onehot_list, batch_first=True, padding_value=0)

    result = []
    result.append(padding_audio)
    result.append(padding_motion)
    result.append(padding_template)
    result.append(padding_onehot)
    # result.append(batch_filename_list)

    return result


def get_dataloaders(args, batch_size=1, workers=10, read_audio=True): #chushi read_audio = Flase  workes==10*** \ 20 \ 30

    df = pd.read_csv('/home/china/Databak/MEAD/mead_v2.csv')
    # df_mesh = pd.read_csv('/home/china/Databak/MEAD/mead_v2_Mesh.csv') # todo **02


    # val_df = df[(df['pid'] == 'M023')|(df['pid'] == 'M029')]  # 验证集 2230-500/1000
    # test_df = df[(df['pid'] == 'M022')|(df['pid'] == 'M030')]  # 测试集
    val_df = df[(df['pid'] == 'M039') | (df['pid'] == 'M035')]  # 验证集
    test_df = df[(df['pid'] == 'M040') | (df['pid'] == 'M037')]  # 测试集

    # pids = ['M003', 'M005', 'M007', 'M009', 'M011', 'W009', 'W011', 'W014', 'W015', 'W016']
    # train_df = pd.concat([df[df['pid'] == pid] for pid in pids])

    # todo 02**
    # val_df_mesh = df_mesh[(df_mesh['pid'] == 'M039') | (df_mesh['pid'] == 'M035')]  # 验证集
    # test_df_mesh = df_mesh[(df_mesh['pid'] == 'M040') | (df_mesh['pid'] == 'M037')]  # 测试集

    # todo 02**
    # train_df_mesh = df_mesh.drop(val_df_mesh.index).drop(test_df_mesh.index)
    # train_df_mesh = train_df_mesh.sample(frac=1, random_state=1)  # 从训练集中抽取10%作为训练集
    # val_df_mesh = val_df_mesh.sample(frac=1, random_state=1)  # 从验证集中抽取10%作为验证集


    train_df = df.drop(val_df.index).drop(test_df.index)  # 将测试集和验证集从训练集中去除
    # train_df = df[(df['pid'] == 'M003')|(df['pid'] == 'M005')| (df['pid'] == 'M007')|(df['pid'] == 'M009')|(df['pid'] == 'M011')|(df['pid'] == 'M012')|(df['pid'] == 'M013')|(df['pid'] == 'M019') |(df['pid'] == 'M023')|(df['pid'] == 'M024')|(df['pid'] == 'M025')|(df['pid'] == 'M026')|(df['pid'] == 'M027')|(df['pid'] == 'M028')|(df['pid'] == 'M040') | (df['pid'] == 'M037')] # 将测试集和验证集从训练集中去除

    train_df = train_df.sample(frac=1, random_state=1)  # frac=0.1 从训练集中抽取10%作为训练集
    val_df = val_df.sample(frac=1, random_state=1)  # frac=0.1 从验证集中抽取10%作为验证集

    dataset = {
        'train': data.DataLoader(MEADDataset(train_df, args, is_train=True, read_audio=read_audio),
                                 batch_size=batch_size, shuffle=True,
                                 drop_last=True, num_workers=workers),
        'valid': data.DataLoader(MEADDataset(val_df, args, is_train=False, read_audio=read_audio),
                                 batch_size=1, shuffle=False,
                                 drop_last=True),
        'test': data.DataLoader(MEADDataset(test_df, args, is_train=False, read_audio=read_audio),
                                batch_size=1, shuffle=False)
    }
    return dataset

if __name__ == "__main__":
    get_dataloaders()
    
