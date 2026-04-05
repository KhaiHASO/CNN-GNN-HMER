import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase_2_gnn_design.log_utils import get_logger
from phase_2_gnn_design.graph_utils import (
    build_feature_knn_edge_index,
    build_grid_edge_index,
    build_hybrid_edge_index,
    build_topk_similarity_edge_index,
    edge_index_summary,
    flatten_feature_map,
    unflatten_node_tensor,
)


class SimpleImgPosEnc(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = feature.shape
        if channels != self.d_model:
            raise ValueError(f"Expected {self.d_model} channels, got {channels}")

        y = torch.linspace(0.0, 1.0, steps=height, device=feature.device)
        x = torch.linspace(0.0, 1.0, steps=width, device=feature.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        pos = torch.zeros(height, width, channels, device=feature.device)
        pos[..., 0::2] = yy.unsqueeze(-1)
        pos[..., 1::2] = xx.unsqueeze(-1)

        return feature + pos.unsqueeze(0) * mask.unsqueeze(-1).float()


class _Bottleneck(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super().__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv1 = nn.Conv2d(n_channels, inter_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(
            inter_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        return torch.cat((x, out), dim=1)


class _SingleLayer(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super().__init__()
        self.conv1 = nn.Conv2d(
            n_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        return torch.cat((x, out), dim=1)


class _Transition(nn.Module):
    def __init__(self, n_channels: int, n_out_channels: int, use_dropout: bool):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(n_out_channels)
        self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        return F.avg_pool2d(out, 2, ceil_mode=True)


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int,
        num_layers: int,
        reduction: float = 0.5,
        bottleneck: bool = True,
        use_dropout: bool = True,
    ):
        super().__init__()
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
    def _make_dense(
        n_channels: int,
        growth_rate: int,
        n_dense_blocks: int,
        bottleneck: bool,
        use_dropout: bool,
    ) -> nn.Sequential:
        layers = []
        for _ in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(_Bottleneck(n_channels, growth_rate, use_dropout))
            else:
                layers.append(_SingleLayer(n_channels, growth_rate, use_dropout))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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


class GraphRefinementBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        edge_mode: str = "hybrid",
        grid_connectivity: int = 4,
        knn_k: int = 8,
        topk_ratio: float = 0.15,
    ):
        super().__init__()
        self.d_model = d_model
        self.edge_mode = edge_mode
        self.grid_connectivity = grid_connectivity
        self.knn_k = knn_k
        self.topk_ratio = topk_ratio
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size, channels, height, width = feature_map.shape
        if channels != self.d_model:
            raise ValueError(f"Expected {self.d_model} channels, got {channels}")

        nodes = flatten_feature_map(feature_map)
        edge_indices = [self._build_edge_index(nodes[batch_idx], height, width) for batch_idx in range(batch_size)]

        refined_batches = [self._message_pass(nodes[batch_idx], edge_indices[batch_idx]) for batch_idx in range(batch_size)]
        refined_nodes = torch.stack(refined_batches, dim=0)
        refined_nodes = self.norm(refined_nodes + nodes)
        refined_map = unflatten_node_tensor(refined_nodes, height=height, width=width)

        return refined_map, edge_index_summary(edge_indices[0], num_nodes=height * width)

    def _build_edge_index(self, batch_nodes: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if self.edge_mode == "grid":
            return build_grid_edge_index(height, width, connectivity=self.grid_connectivity)
        if self.edge_mode == "knn":
            return build_feature_knn_edge_index(batch_nodes, k=self.knn_k)
        if self.edge_mode == "topk":
            return build_topk_similarity_edge_index(batch_nodes, topk_ratio=self.topk_ratio)
        if self.edge_mode == "hybrid":
            return build_hybrid_edge_index(
                batch_nodes,
                height=height,
                width=width,
                k=self.knn_k,
                grid_connectivity=self.grid_connectivity,
            )
        raise ValueError(f"Unsupported edge_mode: {self.edge_mode}")

    @staticmethod
    def _message_pass(nodes: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        try:
            from torch_geometric.nn import GATConv  # type: ignore

            channels = nodes.size(-1)
            gat_layer = GATConv(channels, channels, heads=1, concat=False)
            return gat_layer(nodes, edge_index)
        except ImportError:
            source, target = edge_index
            aggregated = torch.zeros_like(nodes)
            counts = torch.zeros(nodes.size(0), 1, device=nodes.device)
            aggregated.index_add_(0, target, nodes[source])
            counts.index_add_(0, target, torch.ones_like(target, dtype=nodes.dtype).unsqueeze(1))
            counts = counts.clamp_min(1.0)
            return aggregated / counts


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        growth_rate: int = 24,
        num_layers: int = 16,
        use_graph_refinement: bool = False,
        edge_mode: str = "hybrid",
        grid_connectivity: int = 4,
        knn_k: int = 8,
        topk_ratio: float = 0.15,
    ):
        super().__init__()
        self.model = DenseNet(growth_rate=growth_rate, num_layers=num_layers)
        self.feature_proj = nn.Conv2d(self.model.out_channels, d_model, kernel_size=1)
        self.use_graph_refinement = use_graph_refinement
        if use_graph_refinement:
            self.graph_block = GraphRefinementBlock(
                d_model=d_model,
                edge_mode=edge_mode,
                grid_connectivity=grid_connectivity,
                knn_k=knn_k,
                topk_ratio=topk_ratio,
            )
        self.pos_enc_2d = SimpleImgPosEnc(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, img: torch.Tensor, img_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature, mask = self.model(img, img_mask)
        feature = self.feature_proj(feature)
        if self.use_graph_refinement:
            feature, _edge_stats = self.graph_block(feature)
        feature = feature.permute(0, 2, 3, 1)
        feature = self.pos_enc_2d(feature, mask)
        feature = self.norm(feature)
        return feature, mask


def run_encoder_demo(
    batch_size: int = 2,
    height: int = 128,
    width: int = 128,
    d_model: int = 256,
    use_graph_refinement: bool = True,
    edge_mode: str = "hybrid",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    img = torch.randn(batch_size, 1, height, width)
    img_mask = torch.ones(batch_size, height, width, dtype=torch.long)
    encoder = Encoder(
        d_model=d_model,
        use_graph_refinement=use_graph_refinement,
        edge_mode=edge_mode,
        knn_k=4,
        grid_connectivity=4,
        num_layers=4,
    )
    with torch.no_grad():
        cnn_feature, mask = encoder.model(img, img_mask)
        projected = encoder.feature_proj(cnn_feature)
        edge_stats = {}
        if encoder.use_graph_refinement:
            projected, edge_stats = encoder.graph_block(projected)
        feature = projected.permute(0, 2, 3, 1)
        feature = encoder.pos_enc_2d(feature, mask)
        feature = encoder.norm(feature)
    return feature, mask, edge_stats


if __name__ == "__main__":
    logger = get_logger("sandbox_encoder")
    feature, mask, edge_stats = run_encoder_demo()
    logger.info("=== sandbox_encoder.py started ===")
    logger.info("Graph refinement edge stats: %s", edge_stats)
    logger.info("Encoder feature shape: %s", tuple(feature.shape))
    logger.info("Encoder mask shape: %s", tuple(mask.shape))
    logger.info("=== sandbox_encoder.py finished ===")
