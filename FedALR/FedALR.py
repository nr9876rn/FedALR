import argparse
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------- 依赖自 FedCBD --------------
from evaluate import compute_local_test_accuracy
from Data import (
    cifar_dataset,
    fashionmnist_dataset,
    yahoo_dataset,
    svhn_dataset,
    dbpedia_dataset,
)

# ======================= 1. FactorizedLinear =======================
class FactorizedLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, rank_m: int):
        super().__init__()
        r = min(rank_m, in_f, out_f)
        self.A = nn.Parameter(torch.randn(out_f, r))
        self.B = nn.Parameter(torch.randn(r, in_f))
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.B.t(), a=np.sqrt(5))

    def forward(self, x): return F.linear(x, self.A @ self.B)
    def a_params(self): return [self.A]
    def b_params(self): return [self.B]
    def set_AB(self, A, B): self.A.data.copy_(A); self.B.data.copy_(B)


# ======================= 2. Backbone =======================
class SimpleCNN(nn.Module):
    """
    - conv1 / conv2：普通卷积（全局 FedAvg 聚合）
    - fc1 / fc2：低秩 FactorizedLinear
    - fc3：普通全连接（完全本地）
    """
    def __init__(self, in_ch, num_classes, rank_m):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 6, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)

        self.pool, self.relu = nn.MaxPool2d(2, 2), nn.ReLU(inplace=True)

        self.fc1 = FactorizedLinear(16 * 5 * 5, 120, rank_m)
        self.fc2 = FactorizedLinear(120, 84,      rank_m)
        self.fc3 = nn.Linear(84, num_classes)

    # 仅返回可分解层参数（fc1 / fc2）
    def a_params(self):
        return [p for m in (self.fc1, self.fc2) for p in m.a_params()]

    def b_params(self):
        return [p for m in (self.fc1, self.fc2) for p in m.b_params()]

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# ======================= 3. Utils =======================
@torch.no_grad()
def params_to_vec(params):
    return torch.cat([p.view(-1) for p in params])

@torch.no_grad()
def vec_to_params(vec, params):
    idx = 0
    for p in params:
        num = p.numel()
        p.copy_(vec[idx: idx + num].view_as(p))
        idx += num

def B_to_vec(model): return params_to_vec(model.b_params())
def vec_to_B(model, vec): vec_to_params(vec, model.b_params())

@torch.no_grad()
def choose_m_by_energy(S, energy, max_rank):
    cum = torch.cumsum(S, 0) / S.sum()
    return min(int(torch.searchsorted(cum, energy)) + 1, max_rank)

@torch.no_grad()
def stacked_svd_factorization(global_model, local_Ws,
                              energy, max_rank, fixed_rank):
    """只对 fc1 / fc2 做一次 stacked‑SVD 初始化"""
    for name in ['fc1', 'fc2']:
        Ws = torch.cat([w[name] for w in local_Ws], dim=0)
        _, S, Vt = torch.linalg.svd(Ws, full_matrices=False)

        m0 = fixed_rank if fixed_rank > 0 else choose_m_by_energy(S, energy, max_rank)
        rank_target = getattr(global_model, name).A.size(1)
        if m0 > rank_target:
            print(f"[Warn] {name}: m={m0} clipped to {rank_target}")
        m = min(m0, rank_target)

        B_new  = torch.sqrt(S[:m])[:, None] * Vt[:m, :]
        B_pinv = torch.linalg.pinv(B_new)

        for cid, Wdict in enumerate(local_Ws):
            Ak = Wdict[name] @ B_pinv
            layer = getattr(global_model, name)

            if m < rank_target:
                B_full = torch.zeros(rank_target, B_new.shape[1], device=B_new.device)
                A_full = torch.zeros(Ak.shape[0],   rank_target, device=Ak.device)
                B_full[:m] = B_new;  A_full[:, :m] = Ak
            else:
                B_full, A_full = B_new, Ak

            layer.set_AB(A_full, B_full)
            Wdict[name] = (A_full @ B_full).clone()

        print(f"[StackedSVD] {name}: rank={m}, energy={float(S[:m].sum()/S.sum()):.3f}")


# ======================= 4. Federated class =======================
class pFedLatent:
    def __init__(self, args, cfg):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.node_num = args.n_parties

        if args.dataset == 'cifar10':
            ds, in_ch = cifar_dataset, 3
        elif args.dataset == 'fashionmnist':
            ds, in_ch = fashionmnist_dataset, 1
        elif args.dataset == 'svhn':
            ds, in_ch = svhn_dataset, 3
        else:
            raise ValueError('dataset not supported')

        (self.train_dls, _, self.test_dl,
         self.net_dataidx_map, _, self.data_dist) = ds.dataset_read(
            args.dataset, args.datadir, args.batch_size, args.n_parties,
            args.partition, args.beta, args.skew_class, args.ratio)

        self.global_model = SimpleCNN(in_ch, cfg['classes_size'], args.max_rank).to(self.device)
        self.clients = [copy.deepcopy(self.global_model).to(self.device)
                        for _ in range(self.node_num)]

        # 仅缓存 fc1 / fc2 全权重
        self.local_Ws = [dict() for _ in range(self.node_num)]
        self.best_acc = [0.0] * self.node_num

        self.lr_A, self.lr_B = args.lr_A, args.lr_B

    # ---------- Warm‑up ----------
    def warmup(self):
        for cid, (model, loader) in enumerate(zip(self.clients, self.train_dls)):
            opt = optim.SGD(model.parameters(), lr=self.lr_A, momentum=0.9, weight_decay=self.args.reg)
            model.train()
            for _ in range(self.args.warmup_local):
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device, dtype=torch.long)
                    opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()

            for name in ['fc1', 'fc2']:
                layer = getattr(model, name)
                self.local_Ws[cid][name] = (layer.A @ layer.B).detach().cpu()
        print("[Warm‑up] finished")

    # ---------- SVD init ----------
    def stacked_svd_init(self):
        stacked_svd_factorization(self.global_model, self.local_Ws,
                                  self.args.energy, self.args.max_rank, self.args.rank_m)
        self.global_B_vec = B_to_vec(self.global_model).clone()
        for C in self.clients: vec_to_B(C, self.global_B_vec)

    # ---------- 评估 ----------
    def evaluate(self, tag):
        for cid, net in enumerate(self.clients):
            acc, _ = compute_local_test_accuracy(net, self.test_dl, self.data_dist[cid])
            self.best_acc[cid] = max(self.best_acc[cid], acc)
            print(f'>> Client {cid:2d} | Cur Personal acc: {acc:.4f}')
        print(f'>> {tag} | Mean best personal = {np.mean(self.best_acc):.4f}')

    # ---------- FedAvg for conv weights ----------
    @torch.no_grad()
    def _aggregate_conv(self):
        for name in ['conv1', 'conv2']:                          ### Changed
            w_sum = torch.zeros_like(getattr(self.global_model, name).weight)
            b_sum = torch.zeros_like(getattr(self.global_model, name).bias)
            for C in self.clients:
                layer = getattr(C, name)
                w_sum += layer.weight.data
                b_sum += layer.bias.data
            w_avg, b_avg = w_sum / self.node_num, b_sum / self.node_num
            # 更新 global
            g_layer = getattr(self.global_model, name)
            g_layer.weight.data.copy_(w_avg);  g_layer.bias.data.copy_(b_avg)
            # 广播到客户端
            for C in self.clients:
                layer = getattr(C, name)
                layer.weight.data.copy_(w_avg); layer.bias.data.copy_(b_avg)

    # ---------- 在线训练 ----------
    def online_training(self):
        for r in range(self.args.rounds):
            pinv_cache = {n: torch.linalg.pinv(getattr(self.global_model, n).B.detach().cpu())
                          for n in ['fc1', 'fc2']}              ### Changed

            dB_sum = torch.zeros_like(self.global_B_vec);  tot_w = 0.0

            # ---- local updates ----
            for cid, (model, loader) in enumerate(zip(self.clients, self.train_dls)):
                # 同步全球 B、conv
                vec_to_B(model, self.global_B_vec)
                for n in ['conv1', 'conv2']:
                    getattr(model, n).load_state_dict(getattr(self.global_model, n).state_dict())

                # 重算 A_k
                for n in ['fc1', 'fc2']:
                    layer = getattr(model, n)
                    Ak = self.local_Ws[cid][n].to(self.device) @ pinv_cache[n].to(self.device)
                    layer.A.data.copy_(Ak)

                # A‑phase
                for p in model.b_params(): p.requires_grad_(False)
                for p in model.a_params(): p.requires_grad_(True)
                optA = optim.SGD(list(model.a_params()) +                ### Added conv & fc3
                                 [getattr(model, 'conv1').weight, getattr(model, 'conv1').bias,
                                  getattr(model, 'conv2').weight, getattr(model, 'conv2').bias,
                                  *model.fc3.parameters()],
                                 lr=self.lr_A, momentum=0.9)
                model.train()
                for _ in range(self.args.local_epochs):
                    for x, y in loader:
                        x, y = x.to(self.device), y.to(self.device, dtype=torch.long)
                        optA.zero_grad(); F.cross_entropy(model(x), y).backward(); optA.step()

                # B‑phase（只更新 B）
                for p in model.a_params(): p.requires_grad_(False)
                for p in model.b_params(): p.requires_grad_(True)
                optB = optim.SGD(model.b_params(), lr=self.lr_B, momentum=0.9)
                for _ in range(self.args.local_epochs):
                    for x, y in loader:
                        x, y = x.to(self.device), y.to(self.device, dtype=torch.long)
                        optB.zero_grad(); F.cross_entropy(model(x), y).backward(); optB.step()

                # ΔB 聚合
                dB_i = B_to_vec(model) - self.global_B_vec
                w_i  = min(dB_i.norm().item() ** self.args.gamma,
                           self.args.clip_tau if self.args.clip_tau > 0 else dB_i.norm().item() ** self.args.gamma)
                dB_sum += w_i * dB_i;  tot_w += w_i

                for n in ['fc1', 'fc2']:
                    self.local_Ws[cid][n] = (getattr(model, n).A @ getattr(model, n).B).detach().cpu()

            # ---- server aggregation ----
            self.global_B_vec += dB_sum / tot_w
            vec_to_B(self.global_model, self.global_B_vec)
            for C in self.clients: vec_to_B(C, self.global_B_vec)

            self._aggregate_conv()

            self.evaluate(f"Round {r + 1}")

    # ---------- run ----------
    def run(self):
        self.warmup()
        self.stacked_svd_init()
        self.online_training()
        self.evaluate("Final")


# ======================= 5. CLI =======================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--datadir', type=str, required=False, default="../Data/Dataset/", help="Data directory")
    parser.add_argument('--partition', type=str, default='noniid')
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--skew_class', type=int, default=2)
    parser.add_argument('--n_parties', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--warmup_local', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=70)
    parser.add_argument('--lr_A', type=float, default=1e-3)
    parser.add_argument('--lr_B', type=float, default=1e-3)
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--energy', type=float, default=0.95)
    parser.add_argument('--max_rank', type=int, default=64)
    parser.add_argument('--rank_m', type=int, default=-1)
    parser.add_argument('--gamma', type=float, default=4)
    parser.add_argument('--clip_tau', type=float, default=0.0)
    parser.add_argument('--ratio', type=float, default=0.8)
    return parser.parse_args()


def main():
    seed = 37
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    args = get_args()
    cfg = {'classes_size': 10}
    pFedLatent(args, cfg).run()


if __name__ == '__main__':
    main()
