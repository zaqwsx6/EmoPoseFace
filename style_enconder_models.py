import torch.nn as nn
import torch
# from .common import PositionalEncoding
import math
'''
运动风格编码器
'''
class StyleEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Model parameters
        # self.motion_coef_dim = 15072 #01 all face style
        self.motion_coef_dim = 3 #02 pose style
        # if args.rot_repr == 'aa':
        #     self.motion_coef_dim += 1 if args.no_head_pose else 4
        # else:
        #     raise ValueError(f'Unknown rotation representation {args.rot_repr}!')

        # self.feature_dim = args.feature_dim
        # self.n_heads = args.n_heads
        # self.n_layers = args.n_layers
        # self.mlp_ratio = args.mlp_ratio
        self.feature_dim = 512
        self.n_heads = 4
        self.n_layers = 4
        self.mlp_ratio = 4

        # Transformer for feature extraction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=self.n_heads, dim_feedforward=self.mlp_ratio * self.feature_dim,
            activation='gelu', batch_first=True
        )

        self.PE = PositionalEncoding(self.feature_dim)
        self.encoder = nn.ModuleDict({
            'motion_proj': nn.Linear(self.motion_coef_dim, self.feature_dim),
            'transformer': nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers),
        })

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, motion_coef):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :param audio: (batch_size, seq_len)
        :return: (batch_size, feature_dim)
        """
        batch_size, seq_len, _ = motion_coef.shape

        motion_coef=motion_coef[:,:,-3:] #todo 02 add

        # Motion
        motion_feat = self.encoder['motion_proj'](motion_coef)
        motion_feat = self.PE(motion_feat)

        feat = self.encoder['transformer'](motion_feat)  # (N, L, feat_dim)
        #todo **（转换为非时序的特征）
        feat = feat.mean(dim=1)  # Pooling to (N, feat_dim)

        return feat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)