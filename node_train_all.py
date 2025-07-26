import argparse, os, time
import torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import global_mean_pool   # 仅 GIN 需要

# ---------- 全局参数 ----------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'Citeseer', 'Flickr'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)      # only for GAT
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

# ---------- 数据集 ----------
def load_dataset(name, root='data'):
    path = os.path.join(root, name.lower(), "raw")
    if not os.path.exists(path):
        raise FileNotFoundError(f"本地数据集 {path} 不存在！")
    if name in ['Cora', 'Citeseer']:
        dataset = Planetoid(root=root, name=name, transform=NormalizeFeatures())
    elif name == 'Flickr':
        dataset = Flickr(root=os.path.join(root, name))
    else:
        raise ValueError(name)
    return dataset[0]

# ---------- 模型 ----------
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](x, edge_index)

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1))
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](x, edge_index).squeeze(-2)

class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        from torch_geometric.nn import GraphSAGE as GS
        self.model = GS(in_channels, hidden_channels, num_layers, out_channels)
    def forward(self, x, edge_index):
        return self.model(x, edge_index)

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))))
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))))
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))))
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](x, edge_index)

MODELS = {'GCN': GCN, 'GAT': GAT, 'GraphSAGE': SAGE, 'GIN': GIN}

# ---------- 评估 ----------
@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index).cpu()
    y_true = data.y.cpu()
    masks = {'train': data.train_mask, 'val': data.val_mask, 'test': data.test_mask}
    results = {}
    for split, mask in masks.items():
        mask = mask.cpu()
        y_true_split = y_true[mask]
        y_prob_split = torch.softmax(out[mask], dim=1)
        y_pred_split = y_prob_split.argmax(dim=1)
        acc = accuracy_score(y_true_split, y_pred_split)
        f1_ma = f1_score(y_true_split, y_pred_split, average='macro')
        f1_mi = f1_score(y_true_split, y_pred_split, average='micro')
        # AUROC (多分类 OvR)
        if y_prob_split.size(1) == 2:
            auroc = roc_auc_score(y_true_split, y_prob_split[:, 1])
        else:
            auroc = roc_auc_score(y_true_split, y_prob_split, multi_class='ovr')
        results[split] = {'acc': acc, 'f1_macro': f1_ma, 'f1_micro': f1_mi, 'auroc': auroc}
    return results

# ---------- 全图训练 ----------
def train_once_full(model, data, optimizer, device):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    y = data.y[data.train_mask].to(device)
    loss = F.cross_entropy(out[data.train_mask], y)
    loss.backward()
    optimizer.step()
    return loss.item()

def run_model_full(model_name, data, args):
    Model = MODELS[model_name]
    extra = {'heads': args.heads} if model_name == 'GAT' else {}
    model = Model(
        in_channels=data.x.size(-1),
        hidden_channels=args.hidden,
        out_channels=data.y.max().item() + 1,
        num_layers=args.num_layers,
        **extra
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_once_full(model, data, optimizer, args.device)
        elapsed = time.time() - start_time
        epoch_times.append(elapsed)
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f'{model_name}  Epoch {epoch:03d}  loss {train_loss:.4f}  time {elapsed:.2f}s')
    results = evaluate(model, data)
    return results, epoch_times

# ---------- 主函数 ----------
def main():
    args = get_args()
    dataset_names = ['Cora', 'Flickr', 'Citeseer']
    for name in dataset_names:
        data = load_dataset(name).to(args.device)
        records = []

        for mname in MODELS:
            print(f'\n========== {mname} ==========' )
            results, epoch_times = run_model_full(mname, data, args)
            for split, metrics in results.items():
                row = {'model': mname, 'split': split, **metrics}
                if split == 'train':
                    for i, t in enumerate(epoch_times, 1):
                        row_time = row.copy()
                        row_time['epoch'] = i
                        row_time['epoch_time'] = t
                        records.append(row_time)
                else:
                    records.append(row)

        # 写 Excel
        df = pd.DataFrame(records)
        out_file = f"result/node_full_{name}_results.xlsx"
        df.to_excel(out_file, index=False)
        print(f'\n已保存结果到 {out_file}')
        print(df)

if __name__ == '__main__':
    main() 