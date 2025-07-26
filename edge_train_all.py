import argparse, os, itertools
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.transforms import NormalizeFeatures, RandomLinkSplit
from torch_geometric.utils import negative_sampling

# ---------- 全局 ----------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['Cora', 'Citeseer', 'Flickr'], default='Cora')
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)          # GAT
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--fanouts', type=lambda s: list(map(int, s.split(','))), default='15,10')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--neg_ratio', type=int, default=1)      # 负样本数量=正样本*neg_ratio
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

# ---------- 数据集 ----------
def load_dataset(name, root='data'):
    path = os.path.join(root, name.lower(), "raw")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found!")
    if name in ['Cora', 'Citeseer']:
        dataset = Planetoid(root=root, name=name, transform=NormalizeFeatures())
    elif name == 'Flickr':
        dataset = Flickr(root=os.path.join(root, name))
    data = dataset[0]
    # 拆分边
    transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=False,
                                val=0.05, test=0.1)
    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data, data

# ---------- 模型 ----------
class GCN(nn.Module):
    def __init__(self, in_dim, hid, out_layers):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hid)])
        for _ in range(out_layers-1):
            self.convs.append(GCNConv(hid, hid))
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x

class GAT(nn.Module):
    def __init__(self, in_dim, hid, out_layers, heads=4):
        super().__init__()
        self.convs = nn.ModuleList([GATConv(in_dim, hid, heads=heads)])
        for _ in range(out_layers-1):
            self.convs.append(GATConv(hid*heads, hid, heads=heads))
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x

class SAGE(nn.Module):
    def __init__(self, in_dim, hid, out_layers):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim, hid)])
        for _ in range(out_layers-1):
            self.convs.append(SAGEConv(hid, hid))
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x

class GIN(nn.Module):
    def __init__(self, in_dim, hid, out_layers):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.convs = nn.ModuleList([GINConv(nn1)])
        for _ in range(out_layers-1):
            nnk = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid))
            self.convs.append(GINConv(nnk))
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x

MODELS = {'GCN': GCN, 'GAT': GAT, 'GraphSAGE': SAGE, 'GIN': GIN}

# ---------- 工具 ----------
def decode(z, edge_index):
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

# ---------- 全图训练 ----------
def train_once_full(model, data, optimizer, device, neg_ratio):
    model.train()
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    z = model(x, edge_index)
    pos_out = decode(z, data.edge_label_index.to(device))
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.edge_label_index.size(1)*neg_ratio,
        method='sparse'
    ).to(device)
    neg_out = decode(z, neg_edge_index)
    out = torch.cat([pos_out, neg_out])
    y = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
    loss = F.binary_cross_entropy_with_logits(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ---------- 评估 ----------
@torch.no_grad()
def evaluate(model, data, device, Ks=[10, 20, 50]):
    model.eval()
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    z = model(x, edge_index)
    pos_edge = data.edge_label_index
    neg_edge = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge.size(1)
    )
    pos_score = torch.sigmoid(decode(z, pos_edge.to(device)))
    neg_score = torch.sigmoid(decode(z, neg_edge.to(device)))
    scores = torch.cat([pos_score, neg_score]).cpu()
    labels = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))])

    auc = roc_auc_score(labels, scores)
    ap  = average_precision_score(labels, scores)

    # MRR & Hits@K
    pos_size, neg_size = pos_score.size(0), neg_score.size(0)
    ranks = []
    for p, n in zip(pos_score, neg_score):
        rank = (n >= p).sum().item() + 1
        ranks.append(rank)
    ranks = torch.tensor(ranks, dtype=torch.float)
    mrr = (1. / ranks).mean().item()
    hits = {f'Hits@{k}': (ranks <= k).float().mean().item() for k in Ks}

    return {'AUC': auc, 'AP': ap, 'MRR': mrr, **hits}

# ---------- 主 ----------
def main():
    args = get_args()
    dataset_names = ['Cora', 'Flickr', 'Citeseer']
    for dname in dataset_names:
        train_data, val_data, test_data, full_data = load_dataset(dname)

        records = []
        for mname in MODELS:
            print(f'\n========== {mname} ==========' )
            Model = MODELS[mname]
            extra = {'heads': args.heads} if mname == 'GAT' else {}
            model = Model(
                in_dim=full_data.num_features,
                hid=args.hidden,
                out_layers=args.num_layers,
                **extra
            ).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            epoch_times = []
            for epoch in range(1, args.epochs + 1):
                import time
                start_time = time.time()
                loss = train_once_full(model, train_data, optimizer, args.device, args.neg_ratio)
                elapsed = time.time() - start_time
                epoch_times.append(elapsed)
                if epoch % 20 == 0 or epoch == args.epochs:
                    print(f'Epoch {epoch:03d}  loss={loss:.4f}  time={elapsed:.2f}s')

            # 评估
            val_res  = evaluate(model, val_data,  args.device)
            test_res = evaluate(model, test_data, args.device)
            # 记录train split每个epoch的耗时
            for i, t in enumerate(epoch_times, 1):
                row = {'dataset': dname, 'model': mname, 'split': 'train', 'epoch': i, 'epoch_time': t}
                records.append(row)
            for split_name, res in [('val', val_res), ('test', test_res)]:
                row = {'dataset': dname, 'model': mname, 'split': split_name, **res}
                records.append(row)

        # 保存 Excel
        df = pd.DataFrame(records)
        out_file = f"result/link_full_{dname}_results.xlsx"
        df.to_excel(out_file, index=False)
        print(f'\n已保存结果到 {out_file}')
        print(df)

if __name__ == '__main__':
    main()
