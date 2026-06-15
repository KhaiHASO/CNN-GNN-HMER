import math
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from torch import FloatTensor, LongTensor

from .pos_enc import ImgPosEnc
from .gat import GAT


# DenseNet-B
class _Bottleneck(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super(_Bottleneck, self).__init__()
        interChannels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(n_channels, interChannels,
                               kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(
            interChannels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# single layer
class _SingleLayer(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super(_SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(
            n_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# transition layer
class _Transition(nn.Module):
    def __init__(self, n_channels: int, n_out_channels: int, use_dropout: bool):
        super(_Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_out_channels)
        self.conv1 = nn.Conv2d(n_channels, n_out_channels,
                               kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int,
        num_layers: int,
        reduction: float = 0.5,
        bottleneck: bool = True,
        use_dropout: bool = True,
    ):
        super(DenseNet, self).__init__()
        n_dense_blocks = num_layers
        n_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(
            1, n_channels, kernel_size=7, padding=3, stride=2, bias=False
        )
        self.norm1 = nn.BatchNorm2d(n_channels)
        self.dense1 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense2 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )

        self.out_channels = n_channels + n_dense_blocks * growth_rate
        self.post_norm = nn.BatchNorm2d(self.out_channels)

    @staticmethod
    def _make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout):
        layers = []
        for _ in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(_Bottleneck(
                    n_channels, growth_rate, use_dropout))
            else:
                layers.append(_SingleLayer(
                    n_channels, growth_rate, use_dropout))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, x_mask):
        out = self.conv1(x)
        out = self.norm1(out)
        out_mask = x_mask[:, 0::2, 0::2]
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense1(out)
        out = self.trans1(out)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense2(out)
        out = self.trans2(out)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense3(out)
        out = self.post_norm(out)
        return out, out_mask


class Encoder(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        use_gat: bool = False,
        gat_num_layers: int = 2,
        gat_num_heads: int = 8,
        gat_hidden_dim: int = None,
        gat_dropout: float = 0.1,
    ):
        super().__init__()

        self.model = DenseNet(growth_rate=growth_rate, num_layers=num_layers)

        self.feature_proj = nn.Conv2d(
            self.model.out_channels, d_model, kernel_size=1)

        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)

        self.use_gat = use_gat
        if use_gat:
            gat_hidden_dim = gat_hidden_dim or d_model
            self.gat = GAT(
                in_features=d_model,
                hidden_features=gat_hidden_dim,
                out_features=d_model,
                num_layers=gat_num_layers,
                num_heads=gat_num_heads,
                dropout=gat_dropout,
            )

        self.norm = nn.LayerNorm(d_model)

    def _build_grid_adjacency(self, mask: LongTensor) -> LongTensor:
        """Build grid adjacency matrix from mask
        
        Parameters
        ----------
        mask : LongTensor
            [b, h, w] where 0 is valid, 1 is padding
        
        Returns
        -------
        LongTensor
            [b, n_nodes, n_nodes] adjacency matrix (1 for connected, 0 for not)
        """
        b, h, w = mask.shape
        n_nodes = h * w
        
        # Create grid adjacency (8-connected: up, down, left, right, diagonals)
        indices = torch.arange(n_nodes, device=mask.device)
        
        # Right neighbor
        has_right = (indices % w) < (w - 1)
        right_src = indices[has_right]
        right_dst = right_src + 1
        
        # Down neighbor
        has_down = indices < (h - 1) * w
        down_src = indices[has_down]
        down_dst = down_src + w
        
        # Diagonal Down-Right neighbor
        has_down_right = has_down & has_right
        dr_src = indices[has_down_right]
        dr_dst = dr_src + w + 1
        
        # Diagonal Down-Left neighbor
        has_down_left = has_down & ((indices % w) > 0)
        dl_src = indices[has_down_left]
        dl_dst = dl_src + w - 1
        
        # Build base adjacency matrix for a single graph
        adj_single = torch.zeros(n_nodes, n_nodes, dtype=torch.long, device=mask.device)
        adj_single[right_src, right_dst] = 1
        adj_single[right_dst, right_src] = 1
        adj_single[down_src, down_dst] = 1
        adj_single[down_dst, down_src] = 1
        adj_single[dr_src, dr_dst] = 1
        adj_single[dr_dst, dr_src] = 1
        adj_single[dl_src, dl_dst] = 1
        adj_single[dl_dst, dl_src] = 1
        
        # Self-connections (prevents NaN in softmax)
        adj_single[indices, indices] = 1
        
        # Expand to batch size
        adj = adj_single.unsqueeze(0).expand(b, -1, -1).clone()
        
        # Mask out padding nodes (set their connections to 0)
        mask_flat = mask.reshape(b, n_nodes)  # [b, n_nodes]
        padding_mask = (mask_flat == 1)  # True for padding
        adj.masked_fill_(padding_mask.unsqueeze(1), 0)
        adj.masked_fill_(padding_mask.unsqueeze(2), 0)
        
        # Re-apply self-connections for all nodes (including padding nodes)
        batch_indices = torch.arange(b, device=mask.device).unsqueeze(1)
        adj[batch_indices, indices, indices] = 1
        
        return adj

    def forward(
        self, img: FloatTensor, img_mask: LongTensor
    ) -> Tuple[FloatTensor, LongTensor]:
        """encode image to feature

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']

        Returns
        -------
        Tuple[FloatTensor, LongTensor]
            [b, h, w, d], [b, h, w]
        """
        # extract feature
        feature, mask = self.model(img, img_mask)
        feature = self.feature_proj(feature)

        # proj
        feature = rearrange(feature, "b d h w -> b h w d")

        # Apply GAT if enabled (on pure visual features, before adding positional encoding)
        if self.use_gat:
            b, h, w, d = feature.shape
            # Flatten to [b, n_nodes, d]
            feature_flat = feature.view(b, h * w, d)
            # Build adjacency matrix
            adj = self._build_grid_adjacency(mask)
            # Apply GAT with a residual skip connection
            feature_flat = feature_flat + self.gat(feature_flat, adj)
            # Reshape back to [b, h, w, d]
            feature = feature_flat.view(b, h, w, d)
            feature = self.norm(feature)

        # positional encoding (added after GAT to keep coordinates sharp and unblurred)
        feature = self.pos_enc_2d(feature, mask)
        feature = self.norm(feature)

        # flat to 1-D
        return feature, mask
