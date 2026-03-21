import argparse
import math
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Actor, WebKB, WikipediaNetwork
from torch_geometric.datasets.heterophilous_graph_dataset import HeterophilousGraphDataset
from torch_geometric.utils import remove_self_loops, to_undirected


def load_dataset(data_id: int):
    if data_id == 0:
        dataset = HeterophilousGraphDataset(root='/tmp/RomanEmpire', name='Roman-empire')
    elif data_id == 1:
        dataset = HeterophilousGraphDataset(root='/tmp/Minesweeper', name='Minesweeper')
    elif data_id == 2:
        dataset = HeterophilousGraphDataset(root='/tmp/AmazonRatings', name='Amazon-ratings')
    elif data_id == 3:
        dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon')
    elif data_id == 4:
        dataset = WikipediaNetwork(root='/tmp/Squirrel', name='squirrel')
    elif data_id == 5:
        dataset = Actor(root='/tmp/Actor')
    elif data_id == 6:
        dataset = WebKB(root='/tmp/Cornell', name='Cornell')
    elif data_id == 7:
        dataset = WebKB(root='/tmp/Texas', name='Texas')
    else:
        dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')
    return dataset, dataset.num_classes


DATASET_NAMES = {
    0: 'Roman-empire',
    1: 'Minesweeper',
    2: 'Amazon-ratings',
    3: 'Chameleon',
    4: 'Squirrel',
    5: 'Actor',
    6: 'Cornell',
    7: 'Texas',
    8: 'Wisconsin',
}


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg: str = 'auto') -> torch.device:
    device_arg = device_arg.lower()

    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    if device_arg == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        print('[Warning] CUDA requested but not available. Falling back to CPU.')
        return torch.device('cpu')

    if device_arg == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        print('[Warning] MPS requested but not available. Falling back to CPU.')
        return torch.device('cpu')

    return torch.device(device_arg)


def pick_single_split(data, split_idx: int = 0):
    if hasattr(data, 'train_mask') and data.train_mask.dim() == 2:
        split_idx = max(0, min(split_idx, data.train_mask.size(1) - 1))
        data.train_mask = data.train_mask[:, split_idx]
        data.val_mask = data.val_mask[:, split_idx]
        data.test_mask = data.test_mask[:, split_idx]
    return data


def unique_directed_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index

    row, col = edge_index
    key = row.to(torch.long) * int(num_nodes) + col.to(torch.long)
    perm = torch.argsort(key)

    row = row[perm]
    col = col[perm]
    key = key[perm]

    keep = torch.ones_like(key, dtype=torch.bool)
    keep[1:] = key[1:] != key[:-1]
    return torch.stack([row[keep], col[keep]], dim=0)


def prepare_graph(data):
    num_nodes = data.num_nodes

    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index = unique_directed_edges(edge_index, num_nodes=num_nodes)

    row, col = edge_index
    oriented_mask = row < col
    edge_index_solver = torch.stack([row[oriented_mask], col[oriented_mask]], dim=0)

    if edge_index_solver.numel() == 0:
        max_degree = 0
    else:
        deg = torch.bincount(
            torch.cat([edge_index_solver[0], edge_index_solver[1]], dim=0),
            minlength=num_nodes,
        )
        max_degree = int(deg.max().item())

    deg_mp = torch.bincount(col, minlength=num_nodes).to(torch.float32).clamp_min_(1.0)

    data.edge_index_mp = edge_index
    data.edge_index_solver = edge_index_solver
    data.deg_mp = deg_mp
    data.max_degree = max_degree
    return data


def inverse_softplus(x: float) -> float:
    return math.log(math.expm1(x)) if x > 0 else -20.0


class BoundedInfluencePotential(nn.Module):
    def __init__(self, num_basis: int = 8, beta_eps: float = 1e-4):
        super().__init__()
        self.num_basis = num_basis
        self.beta_eps = beta_eps
        self.raw_b0 = nn.Parameter(torch.tensor(-2.25))
        self.raw_a = nn.Parameter(torch.full((num_basis,), -5.0))
        self.raw_beta = nn.Parameter(torch.full((num_basis,), -1.0))
        self.c = nn.Parameter(torch.linspace(-3.0, 3.0, steps=num_basis))

    def constrained_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a = F.softplus(self.raw_a)
        beta = F.softplus(self.raw_beta) + self.beta_eps
        b0 = F.softplus(self.raw_b0)
        c = self.c
        return a, beta, c, b0

    def psi(self, t: torch.Tensor) -> torch.Tensor:
        t = t.clamp_min(0.0)
        a, beta, c, b0 = self.constrained_parameters()
        u = t.unsqueeze(-1) * beta + c
        sig = torch.sigmoid(u)
        return b0 + (a * sig).sum(dim=-1)

    def psi_prime(self, t: torch.Tensor) -> torch.Tensor:
        t = t.clamp_min(0.0)
        a, beta, c, _ = self.constrained_parameters()
        u = t.unsqueeze(-1) * beta + c
        sig = torch.sigmoid(u)
        return (a * beta * sig * (1.0 - sig)).sum(dim=-1)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        t = t.clamp_min(0.0)
        a, beta, c, b0 = self.constrained_parameters()
        u = t.unsqueeze(-1) * beta + c
        base = F.softplus(c)
        return b0 * t + ((a / beta) * (F.softplus(u) - base)).sum(dim=-1)


class InputEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MeanAggregator(nn.Module):
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x.new_zeros(x.shape)
        src, dst = edge_index
        out = x.new_zeros(x.shape)
        out.index_add_(0, dst, x[src])
        return out / deg.unsqueeze(-1).to(x.dtype)


class HeterophilyLinearAgg(nn.Module):
    """
    Heterophily-aware linear pre-activation.

    It keeps ego and neighbor channels separate, includes a high-pass branch (x - mean_nb),
    and adds a 2-hop mean branch. All branches remain linear in the node features, so this
    is still compatible with the methodology's flexible linear pre-activation block.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.agg = MeanAggregator()
        self.self_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.nb1_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.hp_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.nb2_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.branch_logits = nn.Parameter(torch.zeros(4))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, edge_index_mp: torch.Tensor, deg_mp: torch.Tensor) -> torch.Tensor:
        nb1 = self.agg(h, edge_index_mp, deg_mp)
        nb2 = self.agg(nb1, edge_index_mp, deg_mp)
        hp = h - nb1

        scales = 2.0 * torch.sigmoid(self.branch_logits)
        z = (
            scales[0] * self.self_lin(h)
            + scales[1] * self.nb1_lin(nb1)
            + scales[2] * self.hp_lin(hp)
            + scales[3] * self.nb2_lin(nb2)
            + self.bias
        )
        return self.norm(z)


def symmetric_edge_features(h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
    return torch.cat([h_src + h_dst, torch.abs(h_src - h_dst), h_src * h_dst], dim=-1)


def directed_edge_features(h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
    return torch.cat([h_src + h_dst, h_src - h_dst, h_src * h_dst], dim=-1)


class EdgeWeightNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, init_scale: float = 0.08):
        super().__init__()
        feat_dim = 3 * in_dim
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.raw_scale = nn.Parameter(torch.tensor(inverse_softplus(init_scale), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        final_linear = self.mlp[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(self, h: torch.Tensor, edge_index_solver: torch.Tensor) -> torch.Tensor:
        if edge_index_solver.numel() == 0:
            return h.new_zeros((0,))

        row, col = edge_index_solver
        pair = symmetric_edge_features(h[row], h[col])
        score = self.mlp(pair).squeeze(-1)
        raw_w = F.softplus(score) + 1e-8
        raw_w = raw_w / raw_w.detach().mean().clamp_min(1e-6)
        scale = F.softplus(self.raw_scale)
        return scale * raw_w


class OffsetNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_relations: int = 8,
        mu_max: float = 1.0,
        init_scale: float = 0.12,
    ):
        super().__init__()
        self.num_relations = num_relations
        self.mu_max = float(mu_max)

        self.relations = nn.Parameter(torch.zeros(num_relations, out_dim))
        self.mlp = nn.Sequential(
            nn.Linear(3 * in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_relations),
        )
        self.diff_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.raw_scale = nn.Parameter(torch.tensor(math.log(init_scale / (1.0 - init_scale)), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.relations)
        final_linear = self.mlp[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        nn.init.zeros_(self.diff_proj.weight)

    def directed_candidate(self, h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(directed_edge_features(h_src, h_dst))
        pi = F.softmax(logits, dim=-1)
        return pi @ self.relations

    def forward(self, h: torch.Tensor, edge_index_solver: torch.Tensor) -> torch.Tensor:
        if edge_index_solver.numel() == 0:
            return h.new_zeros((0, self.relations.size(-1)))

        row, col = edge_index_solver
        mu_tilde_ij = self.directed_candidate(h[row], h[col])
        mu_tilde_ji = self.directed_candidate(h[col], h[row])
        mu_dict = 0.5 * (mu_tilde_ij - mu_tilde_ji)
        mu_local = self.diff_proj(h[row] - h[col])
        mu = mu_dict + mu_local

        scale = torch.sigmoid(self.raw_scale)
        mu = scale * mu
        if self.mu_max > 0.0:
            mu = self.mu_max * torch.tanh(mu / self.mu_max)
        return mu


class ShiftedProxActivation(nn.Module):
    def __init__(
        self,
        potential: BoundedInfluencePotential,
        alpha: float = 1.0,
        kappa: float = 0.9,
        num_pd_iter: int = 12,
        num_newton: int = 8,
        xi: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.potential = potential
        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self.num_pd_iter = int(num_pd_iter)
        self.num_newton = int(num_newton)
        self.xi = float(xi)
        self.eps = float(eps)

    @staticmethod
    def incidence_forward(u: torch.Tensor, edge_index_solver: torch.Tensor) -> torch.Tensor:
        row, col = edge_index_solver
        return u[row] - u[col]

    @staticmethod
    def incidence_adjoint(y: torch.Tensor, edge_index_solver: torch.Tensor, num_nodes: int) -> torch.Tensor:
        row, col = edge_index_solver
        out = y.new_zeros((num_nodes, y.size(-1)))
        out.index_add_(0, row, y)
        out.index_add_(0, col, -y)
        return out

    def step_sizes(self, max_degree: int) -> Tuple[float, float]:
        if max_degree <= 0:
            return self.kappa, self.kappa
        denom = math.sqrt(2.0 * float(max_degree))
        tau = self.kappa / denom
        sigma = self.kappa / denom
        return tau, sigma

    def edgewise_prox_unshifted(self, v: torch.Tensor, weights: torch.Tensor, sigma: float) -> torch.Tensor:
        if v.numel() == 0:
            return v

        r = torch.norm(v, p=2, dim=-1)
        lam = weights / sigma

        s = torch.zeros_like(r)
        psi0 = self.potential.psi(torch.zeros_like(r))
        f0 = lam * psi0 - r

        active = (r > self.eps) & (f0 < 0.0)
        if active.any():
            s[active] = r[active]
            for _ in range(self.num_newton):
                s_active = s[active]
                r_active = r[active]
                lam_active = lam[active]

                psi_val = self.potential.psi(s_active)
                psi_prime_val = self.potential.psi_prime(s_active)

                numer = s_active + lam_active * psi_val - r_active
                denom = 1.0 + lam_active * psi_prime_val
                s_next = s_active - numer / denom
                s_next = torch.minimum(torch.maximum(s_next, torch.zeros_like(s_next)), r_active)
                s[active] = s_next

        scale = torch.zeros_like(r)
        nz = r > self.eps
        scale[nz] = s[nz] / r[nz]
        return scale.unsqueeze(-1) * v

    def forward(
        self,
        z: torch.Tensor,
        edge_index_solver: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
        max_degree: int,
    ) -> torch.Tensor:
        if edge_index_solver.numel() == 0:
            return z

        tau, sigma = self.step_sizes(max_degree)

        u = z
        u_bar = u
        y = z.new_zeros((edge_index_solver.size(1), z.size(1)))

        z_coeff = tau / self.alpha
        prox_g_denom = 1.0 + z_coeff

        for _ in range(self.num_pd_iter):
            y_tilde = y + sigma * (self.incidence_forward(u_bar, edge_index_solver) - offsets)
            y_hat = self.edgewise_prox_unshifted(y_tilde / sigma, weights, sigma)
            y_next = y_tilde - sigma * y_hat

            q = u - tau * self.incidence_adjoint(y_next, edge_index_solver, z.size(0))
            u_next = (q + z_coeff * z) / prox_g_denom
            u_bar = u_next + self.xi * (u_next - u)

            u = u_next
            y = y_next

        return u


class ShiftedProxActLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_hidden_dim: int = 128,
        num_relations: int = 8,
        num_basis: int = 8,
        alpha: float = 1.0,
        kappa: float = 0.9,
        num_pd_iter: int = 12,
        num_newton: int = 8,
        xi: float = 1.0,
        mu_max: float = 0.75,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.lin_agg = HeterophilyLinearAgg(in_dim=in_dim, out_dim=out_dim)
        self.edge_weight_net = EdgeWeightNet(in_dim=in_dim, hidden_dim=edge_hidden_dim)
        self.offset_net = OffsetNet(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=edge_hidden_dim,
            num_relations=num_relations,
            mu_max=mu_max,
        )
        self.potential = BoundedInfluencePotential(num_basis=num_basis)
        self.prox_act = ShiftedProxActivation(
            potential=self.potential,
            alpha=alpha,
            kappa=kappa,
            num_pd_iter=num_pd_iter,
            num_newton=num_newton,
            xi=xi,
        )
        self.raw_prox_gate = nn.Parameter(torch.tensor(-2.0))
        self.residual_proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = float(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index_mp: torch.Tensor,
        edge_index_solver: torch.Tensor,
        deg_mp: torch.Tensor,
        max_degree: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = self.lin_agg(h, edge_index_mp, deg_mp)
        weights = self.edge_weight_net(h, edge_index_solver)
        offsets = self.offset_net(h, edge_index_solver)
        u = self.prox_act(z, edge_index_solver, weights, offsets, max_degree)

        prox_gate = torch.sigmoid(self.raw_prox_gate)
        update = z + prox_gate * (u - z)
        update = F.dropout(update, p=self.dropout, training=self.training)
        out = self.norm(self.residual_proj(h) + update)

        aux = {
            'z': z,
            'weights': weights,
            'offsets': offsets,
            'prox_gate': prox_gate.detach(),
        }
        return out, aux


class ShiftedProxGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        edge_hidden_dim: int = 128,
        num_relations: int = 8,
        num_basis: int = 8,
        alpha: float = 1.0,
        kappa: float = 0.9,
        num_pd_iter: int = 12,
        num_newton: int = 8,
        xi: float = 1.0,
        mu_max: float = 0.75,
        dropout: float = 0.6,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.input_encoder = InputEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)

        layers = []
        for _ in range(num_layers):
            layers.append(
                ShiftedProxActLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    edge_hidden_dim=edge_hidden_dim,
                    num_relations=num_relations,
                    num_basis=num_basis,
                    alpha=alpha,
                    kappa=kappa,
                    num_pd_iter=num_pd_iter,
                    num_newton=num_newton,
                    xi=xi,
                    mu_max=mu_max,
                    dropout=dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

        total_dim = hidden_dim * (num_layers + 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data) -> Dict[str, torch.Tensor]:
        h = self.input_encoder(data.x)

        reps = [h]
        aux_per_layer = []
        for layer in self.layers:
            h, aux = layer(
                h,
                data.edge_index_mp,
                data.edge_index_solver,
                data.deg_mp,
                data.max_degree,
            )
            reps.append(h)
            aux_per_layer.append(aux)

        final_rep = torch.cat(reps, dim=-1)
        logits = self.classifier(final_rep)
        return {
            'logits': logits,
            'embeddings': final_rep,
            'aux': aux_per_layer,
        }


@torch.no_grad()
def masked_accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return pred[mask].eq(y[mask]).float().mean().item()


@torch.no_grad()
def summarize_aux(aux_per_layer):
    if not aux_per_layer:
        return 0.0, 0.0, 0.0
    last = aux_per_layer[-1]
    weights = last['weights']
    offsets = last['offsets']
    mean_w = weights.mean().item() if weights.numel() > 0 else 0.0
    mean_mu = offsets.norm(dim=-1).mean().item() if offsets.numel() > 0 else 0.0
    prox_gate = float(last['prox_gate'].item()) if isinstance(last['prox_gate'], torch.Tensor) else float(last['prox_gate'])
    return mean_w, mean_mu, prox_gate


def auxiliary_regularization(aux_per_layer, weight_reg: float = 0.0, offset_reg: float = 0.0) -> torch.Tensor:
    reg = None
    for aux in aux_per_layer:
        if weight_reg > 0.0 and aux['weights'].numel() > 0:
            term = weight_reg * aux['weights'].mean()
            reg = term if reg is None else reg + term
        if offset_reg > 0.0 and aux['offsets'].numel() > 0:
            term = offset_reg * aux['offsets'].pow(2).mean()
            reg = term if reg is None else reg + term
    if reg is None:
        reg = torch.tensor(0.0)
    return reg


def train_one_epoch(
    model,
    data,
    optimizer,
    grad_clip: float = 1.0,
    weight_reg: float = 1e-4,
    offset_reg: float = 5e-5,
    label_smoothing: float = 0.0,
):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    logits = out['logits']
    loss = F.cross_entropy(
        logits[data.train_mask],
        data.y[data.train_mask],
        label_smoothing=label_smoothing,
    )
    loss = loss + auxiliary_regularization(out['aux'], weight_reg=weight_reg, offset_reg=offset_reg).to(logits.device)
    loss.backward()
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()

    return {
        'loss': loss.item(),
        'logits': logits.detach(),
        'aux': out['aux'],
    }


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data)
    logits = out['logits']
    return {
        'val_acc': masked_accuracy(logits, data.y, data.val_mask),
        'test_acc': masked_accuracy(logits, data.y, data.test_mask),
        'logits': logits,
        'aux': out['aux'],
    }


def main():
    parser = argparse.ArgumentParser(description='Heterophily-boosted Shifted ProxAct GNN')
    parser.add_argument('data', type=int, help='dataset id: 0..8')
    parser.add_argument('--device', type=str, default='auto', help='auto, cuda, mps, cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_idx', type=int, default=0)

    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--edge_hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_relations', type=int, default=8)
    parser.add_argument('--num_basis', type=int, default=8)

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--kappa', type=float, default=0.9)
    parser.add_argument('--num_pd_iter', type=int, default=12)
    parser.add_argument('--num_newton', type=int, default=8)
    parser.add_argument('--xi', type=float, default=1.0)
    parser.add_argument('--mu_max', type=float, default=0.75)

    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--weight_reg', type=float, default=1e-4)
    parser.add_argument('--offset_reg', type=float, default=5e-5)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--log_every', type=int, default=50)

    args = parser.parse_args()
    set_seed(args.seed)

    dataset, num_classes = load_dataset(args.data)
    data = dataset[0]
    data = pick_single_split(data, split_idx=args.split_idx)
    data = prepare_graph(data)

    device = get_device(args.device)

    data.x = data.x.to(torch.float32)
    data.x = F.normalize(data.x, p=2, dim=-1)
    data = data.to(device)

    model = ShiftedProxGNN(
        in_dim=dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        edge_hidden_dim=args.edge_hidden_dim,
        num_relations=args.num_relations,
        num_basis=args.num_basis,
        alpha=args.alpha,
        kappa=args.kappa,
        num_pd_iter=args.num_pd_iter,
        num_newton=args.num_newton,
        xi=args.xi,
        mu_max=args.mu_max,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=50,
        min_lr=1e-5,
    )

    best_val = -1.0
    best_test = -1.0
    best_state = None
    patience_ctr = 0

    dataset_name = DATASET_NAMES.get(args.data, f'data_{args.data}')
    print(f'Dataset: {dataset_name}')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')
        print(f'CUDA capability: {torch.cuda.get_device_capability(0)}')
        mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
        mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f'Initial GPU memory | allocated: {mem_alloc:.2f} MB | reserved: {mem_reserved:.2f} MB')
    print(f'Nodes: {data.num_nodes} | Solver edges: {data.edge_index_solver.size(1)} | Max degree: {data.max_degree}')
    print(f'Model device: {next(model.parameters()).device}')
    print(f'x device: {data.x.device}')
    print(f'edge_index_mp device: {data.edge_index_mp.device}')
    print(f'edge_index_solver device: {data.edge_index_solver.device}')

    for epoch in range(1, args.max_epochs + 1):
        train_info = train_one_epoch(
            model,
            data,
            optimizer,
            grad_clip=args.grad_clip,
            weight_reg=args.weight_reg,
            offset_reg=args.offset_reg,
            label_smoothing=args.label_smoothing,
        )
        eval_info = evaluate(model, data)
        val_acc = eval_info['val_acc']
        test_acc = eval_info['test_acc']
        scheduler.step(val_acc)

        improved = val_acc > best_val
        if improved:
            best_val = val_acc
            best_test = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % args.log_every == 0 or improved:
            mean_w, mean_mu, prox_gate = summarize_aux(eval_info['aux'])
            log_msg = (
                f'Epoch {epoch:04d} | '
                f'Loss: {train_info["loss"]:.4f} | '
                f'Val: {val_acc:.4f} | Test: {test_acc:.4f} | '
                f'Best Val: {best_val:.4f} | Best Test: {best_test:.4f} | '
                f'Mean w: {mean_w:.4f} | Mean ||mu||: {mean_mu:.4f} | Prox gate: {prox_gate:.4f}'
            )
            if device.type == 'cuda':
                mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
                log_msg += f' | GPU Mem: {mem_alloc:.2f} MB'
            print(log_msg)

        if patience_ctr > args.patience:
            print(f'Early stopping at epoch {epoch}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    eval_info = evaluate(model, data)
    mean_w, mean_mu, prox_gate = summarize_aux(eval_info['aux'])
    print(
        f'Final Best-Val Model | Val: {eval_info["val_acc"]:.4f} | '
        f'Test: {eval_info["test_acc"]:.4f} | '
        f'Mean w: {mean_w:.4f} | Mean ||mu||: {mean_mu:.4f} | Prox gate: {prox_gate:.4f}'
    )


if __name__ == '__main__':
    main()
