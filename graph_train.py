import argparse, os, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.datasets import TUDataset, ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.transforms import NormalizeFeatures
import pandas as pd
import itertools

# ---------- 参数 ----------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['MUTAG', 'ZINC'], default='MUTAG')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--heads', type=int, default=4)      # GAT
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

# ---------- 数据集 ----------
def load_dataset(name, root='data'):
    if name == 'MUTAG':
        path = os.path.join(root, 'TUDataset', name)
        if not os.path.exists(os.path.join(path, 'raw')):
            raise FileNotFoundError(f"本地 {path}/raw 不存在")
        dataset = TUDataset(root=path, name=name, use_node_attr=True)
    elif name == 'ZINC':
        path = os.path.join(root, name)
        if not os.path.exists(os.path.join(path, 'train')):
            raise FileNotFoundError(f"本地 {path} 未准备好")
        dataset = ZINC(root=path, subset=True, split='train')
    else:
        raise ValueError(name)
    return dataset

# ---------- 池化 ----------
# global_min_pool 官方没有，自定义一行即可
def global_min_pool(x, batch):
    # x: [N, F], batch: [N]
    return global_add_pool(-x, batch) * -1   # 取负号再取相反数即得最小值

POOLS = {
    'avg': global_mean_pool,
    'max': global_max_pool,
    'min': global_min_pool
}

# ---------- 模型 ----------
class GCN(nn.Module):
    def __init__(self, in_dim, hid, layers):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hid)])
        for _ in range(layers-1):
            self.convs.append(GCNConv(hid, hid))
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x

class GAT(nn.Module):
    def __init__(self, in_dim, hid, layers, heads=4):
        super().__init__()
        self.convs = nn.ModuleList([GATConv(in_dim, hid, heads=heads)])
        for _ in range(layers-1):
            self.convs.append(GATConv(hid*heads, hid, heads=heads))
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x

class SAGE(nn.Module):
    def __init__(self, in_dim, hid, layers):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim, hid)])
        for _ in range(layers-1):
            self.convs.append(SAGEConv(hid, hid))
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x

class GIN(nn.Module):
    def __init__(self, in_dim, hid, layers):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.convs = nn.ModuleList([GINConv(nn1)])
        for _ in range(layers-1):
            nnk = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid))
            self.convs.append(GINConv(nnk))
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x

MODELS = {'GCN': GCN, 'GAT': GAT, 'GraphSAGE': SAGE, 'GIN': GIN}

# ---------- 图分类网络 ----------
class GraphClassifier(nn.Module):
    def __init__(self, model_cls, pool, in_dim, hid, layers, out_dim, **kw):
        super().__init__()
        self.encoder = model_cls(in_dim, hid, layers, **kw)
        self.pool = pool
        self.fc = nn.Linear(hid if model_cls != GAT else hid*kw.get('heads', 1), out_dim)

    def forward(self, x, edge_index, batch):
        x = self.encoder(x, edge_index)
        x = self.pool(x, batch)
        return self.fc(x).squeeze(-1)   # squeeze 兼容 ZINC 回归

# ---------- 训练/评估 ----------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        total_loss += loss.item() * data.num_graphs
        if criterion.__class__.__name__ == 'CrossEntropyLoss':
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
    if criterion.__class__.__name__ == 'CrossEntropyLoss':
        return total_loss / len(loader.dataset), correct / len(loader.dataset)
    else:
        return total_loss / len(loader.dataset), None   # MAE

# ---------- 主 ----------
def main():
    args = get_args()
    dataset = load_dataset(args.dataset)
    is_classification = args.dataset == 'MUTAG'
    task_metric = 'Accuracy' if is_classification else 'MAE'

    # 划分
    if is_classification:
        from torch.utils.data import random_split
        n = len(dataset)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        n_test = n - n_train - n_val
        train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_set, batch_size=args.batch_size)
        test_loader  = DataLoader(test_set, batch_size=args.batch_size)
    else:  # ZINC 官方已有 split
        train_loader = DataLoader(ZINC(root='data/zinc', subset=True, split='train'), batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(ZINC(root='data/zinc', subset=True, split='val'),   batch_size=args.batch_size)
        test_loader  = DataLoader(ZINC(root='data/zinc', subset=True, split='test'),  batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss() if is_classification else nn.L1Loss()

    records = []
    for mname, pool_name in itertools.product(MODELS, POOLS):
        print(f'\n=== {mname} + {pool_name}Pool ===')
        pool = POOLS[pool_name]
        extra = {'heads': args.heads} if mname == 'GAT' else {}
        model = GraphClassifier(
            MODELS[mname], pool,
            in_dim=dataset.num_features,
            hid=args.hidden,
            layers=args.num_layers,
            out_dim=dataset.num_classes if is_classification else 1,
            **extra
        ).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_metric = float('inf') if not is_classification else 0.
        best_test_metric = 0.
        for epoch in range(1, args.epochs+1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
            val_loss, val_metric = test_epoch(model, val_loader, criterion, args.device)
            test_loss, test_metric = test_epoch(model, test_loader, criterion, args.device)

            if is_classification:
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_test_metric = test_metric
            else:
                if val_loss < best_val_metric:
                    best_val_metric = val_loss
                    best_test_metric = test_metric

        records.append({
            'model': mname,
            'pool': pool_name,
            task_metric: best_test_metric
        })

    # 保存
    df = pd.DataFrame(records)
    out_file = f"result/graph_{args.dataset}_results.xlsx"
    df.to_excel(out_file, index=False)
    print(f'\n结果已保存到 {out_file}')
    print(df)

if __name__ == '__main__':
    main()