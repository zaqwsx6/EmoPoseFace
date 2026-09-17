import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from hubert.modeling_hubert import HubertModel
from torch import Tensor

'''
test-04 model
use FaceDiff
exp 3D mesh(15069) + pose flame(3)
with style enconder
flame pose 和 3dmesh face使用2个gru分开生成 
'''
def adjust_input_representation(audio_embedding_matrix, vertex_matrix, ifps, ofps):
    """
    Brings audio embeddings and visual frames to the same frame rate.

    Args:
        audio_embedding_matrix: The audio embeddings extracted by the audio encoder
        vertex_matrix: The animation sequence represented as a series of vertex positions (or blendshape controls)
        ifps: The input frame rate (it is 50 for the HuBERT encoder)
        ofps: The output frame rate
    """
    if ifps % ofps == 0:
        factor = -1 * (-ifps // ofps)
        if audio_embedding_matrix.shape[1] % 2 != 0:
            audio_embedding_matrix = audio_embedding_matrix[:, :audio_embedding_matrix.shape[1] - 1]

        if audio_embedding_matrix.shape[1] > vertex_matrix.shape[1] * 2:
            audio_embedding_matrix = audio_embedding_matrix[:, :vertex_matrix.shape[1] * 2]

        elif audio_embedding_matrix.shape[1] < vertex_matrix.shape[1] * 2:
            vertex_matrix = vertex_matrix[:, :audio_embedding_matrix.shape[1] // 2]
    elif ifps > ofps:
        factor = -1 * (-ifps // ofps)
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True, mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
    else:
        factor = 1
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True, mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)

    frame_num = vertex_matrix.shape[1]
    audio_embedding_matrix = torch.reshape(audio_embedding_matrix, (1, audio_embedding_matrix.shape[1] // factor, audio_embedding_matrix.shape[2] * factor))
    return audio_embedding_matrix, vertex_matrix, frame_num


# dropout mask for speech conditioning
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


''' 
    网络 exp 3Dmesh + flame pose：
    基于(pose+exp)(T,56)flame参数 +gru* or transformer
'''

class FaceDiff(nn.Module):
    def __init__(
            self,
            args,
            vertice_dim: int,
            pose_dim: int,
            latent_dim: int = 512,
            cond_feature_dim: int = 1536,
            diffusion_steps: int = 500,
            gru_latent_dim: int = 512,
            num_layers: int = 2,


    ) -> None:

        super().__init__()
        self.i_fps = args.input_fps # audio fps (input to the network)
        self.o_fps = args.output_fps # 4D Scan fps (output or target)
        self.one_hot_timesteps = np.eye(args.diff_steps)

        # Audio Encoder
        self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.audio_dim = self.audio_encoder.encoder.config.hidden_size
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.device = args.device

        frozen_layers = [0,1]

        for name, param in self.audio_encoder.named_parameters():
            if name.startswith("feature_projection"):
                param.requires_grad = False
            if name.startswith("encoder.layers"):
                layer = int(name.split(".")[2])
                if layer in frozen_layers:
                    param.requires_grad = False

        # conditional projection
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)

        # noised animation projection
        self.input_projection = nn.Sequential(
            nn.Linear(vertice_dim, latent_dim * 2),
            nn.Conv1d(1, 1, kernel_size=9, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.input_projection_pose = nn.Sequential(
            nn.Linear(pose_dim, latent_dim * 2),
            nn.Conv1d(1, 1, kernel_size=9, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        # timestep projection
        self.time_mlp = nn.Sequential(
            nn.Linear(diffusion_steps, latent_dim),
            nn.Mish(),
        )
        # self.norm_cond = nn.LayerNorm(latent_dim * 3) # with style01
        self.norm_cond = nn.LayerNorm(latent_dim * 3 )# with style02
        self.norm_cond_pose = nn.LayerNorm(latent_dim * 4)# with style02

        # facial decoder
        # self.gru = nn.GRU(latent_dim * 3, gru_latent_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.gru = nn.GRU(latent_dim * 3, gru_latent_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.final_layer = nn.Linear(gru_latent_dim, vertice_dim)

        #todo pose gru 01-a

        self.gru_pose = nn.GRU(latent_dim * 4, gru_latent_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
        # todo pose gru 02-b
        # self.gru_pose = nn.GRU(latent_dim * 3, gru_latent_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.final_layer_pose = nn.Linear(gru_latent_dim, pose_dim)

        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

        nn.init.constant_(self.final_layer_pose.weight, 0)
        nn.init.constant_(self.final_layer_pose.bias, 0)

        # Subject embedding, S
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), latent_dim, bias=False)

        self.obj_pose=nn.Linear(latent_dim*2, latent_dim)

    def forward(
            self, x: Tensor,  times: Tensor, cond_embed: Tensor, template, one_hot, emostyle
    ):
        batch_size, device = x.shape[0], x.device
        times = torch.FloatTensor(self.one_hot_timesteps[times])
        times = times.to(device=device)

        template = template.unsqueeze(1)

        #todo 分x
        x_pose = x[:,:,-3:]  # pose flame
        x_face = x[:,:,:-3]  # face mesh

        #todo with style01

        # obj_embedding = self.obj_vector(one_hot)
        # emotion_style_embedding = torch.cat([obj_embedding, emostyle], dim=-1)
        # emotion_style_embedding = self.obj_emo(emotion_style_embedding)
        # emotion_style_embedding = emotion_style_embedding.unsqueeze(1)  # ** add

        # todo with posestyle01-a
        obj_embedding = self.obj_vector(one_hot)
        obj_embedding = obj_embedding.unsqueeze(1)

        # todo with posestyle02-b

        # obj_embedding = self.obj_vector(one_hot)
        # obj_embedding_face = obj_embedding.unsqueeze(1)
        # pose_style_embedding = torch.cat([obj_embedding, emostyle], dim=-1)
        # pose_style_embedding = self.obj_pose(pose_style_embedding)
        # pose_style_embedding = pose_style_embedding.unsqueeze(1)  # ** add

        # project to latent space
        x_face = x_face.permute(1, 0, 2)  # x 表示面部运动 dim 15069
        x_face = self.input_projection(x_face)
        x_face = x_face.permute(1, 0, 2)

        # todo pose
        x_pose =x_pose.permute(1, 0, 2)
        x_pose = self.input_projection_pose(x_pose)
        x_pose = x_pose.permute(1, 0, 2)

        hidden_states = cond_embed # audio
        hidden_states = self.audio_encoder(hidden_states).last_hidden_state
        hidden_states, x, frame_num = adjust_input_representation(hidden_states, x, self.i_fps, self.o_fps)
        cond_embed = hidden_states[:, :frame_num]
        x_face = x_face[:, :frame_num]
        x_pose = x_pose[:, :frame_num]

        # todo with - posestyle01-a
        pose_style_embedding = emostyle.expand(1, frame_num, -1)# withstyle02

        cond_tokens = self.cond_projection(cond_embed)

        # create the diffusion timestep embedding
        t_tokens = self.time_mlp(times)
        t_tokens = t_tokens.repeat(frame_num, 1, 1)
        t_tokens = t_tokens.permute(1, 0, 2)

        # full conditioning tokens
        # full_cond_tokens = torch.cat([cond_tokens, x, t_tokens], dim=-1)# withstyle01
        full_cond_tokens = torch.cat([cond_tokens, x_face, t_tokens], dim=-1) # 面部 不需要posestyle
        full_cond_tokens = self.norm_cond(full_cond_tokens)
        # todo posestyle 01-a
        full_cond_tokens_pose = torch.cat([cond_tokens, x_pose, t_tokens, pose_style_embedding], dim=-1)  # 面部 不需要posestyle
        full_cond_tokens_pose = self.norm_cond_pose(full_cond_tokens_pose)

        #todo posestyle 02-b

        # full_cond_tokens_pose = torch.cat([cond_tokens, x_pose, t_tokens], dim=-1)  # 面部 不需要posestyle
        # full_cond_tokens_pose = self.norm_cond(full_cond_tokens_pose)

        output_face, _ = self.gru(full_cond_tokens)
        output_pose, _ =self.gru_pose(full_cond_tokens_pose)
        # output = output * obj_embedding

        #todo posestyle 01-a

        output_face = output_face * obj_embedding
        output_pose = output_pose * obj_embedding

        #todo posestyle 02-b

        # output_face = output_face * obj_embedding_face
        # output_pose = output_pose * pose_style_embedding

        output_face = self.final_layer(output_face)
        output_pose = self.final_layer_pose(output_pose)
        # output = output + template

        output_face = output_face + template
        # output_pose= output_pose[:,:,-3:]
        output_final = torch.cat([output_face,output_pose], dim=-1)

        # return output
        return output_final
