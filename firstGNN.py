import cugraph
import torch

def louvain(dgl_g):
    cugraph_g = dgl_g.to_cugraph().to_undirected()
    df, _ = cugraph.louvain(cugraph_g, resolution=3)
    # revert the node ID renumbering by cugraph
    df = cugraph_g.unrenumber(df, 'vertex').sort_values('vertex')
    return torch.utils.dlpack.from_dlpack(df['partition'].to_dlpack()).long()

def core_number(dgl_g):
    cugraph_g = dgl_g.to_cugraph().to_undirected()
    df = cugraph.core_number(cugraph_g)
    # revert the node ID renumbering by cugraph
    df = cugraph_g.unrenumber(df, 'vertex').sort_values('vertex')
    return torch.utils.dlpack.from_dlpack(df['core_number'].to_dlpack()).long()

import dgl.transforms as T
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import SAGEConv
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

device = torch.device('cuda')
dataset = DglNodePropPredDataset(name='ogbn-arxiv')
g, label = dataset[0]
transform = T.Compose([
    T.AddReverse(),
    T.AddSelfLoop(),
    T.ToSimple()
])
g = transform(g).int().to(device)


feat1 = louvain(g)
feat2 = core_number(g)
# convert to one-hot
feat1 = F.one_hot(feat1, feat1.max() + 1)
feat2 = F.one_hot(feat2, feat2.max() + 1)
# concat feat1 and feat2
x = torch.cat([feat1, feat2], dim=1).float()

class GraphSAGE(nn.Module):
    def __init__(self, in_size, num_classes, num_hidden=256, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.layers.append(SAGEConv(in_size, num_hidden, 'mean'))
        self.layers.append(SAGEConv(num_hidden, num_hidden, 'mean'))
        self.layers.append(SAGEConv(num_hidden, num_classes, 'mean'))

        for _ in range(2):
            self.bns.append(nn.BatchNorm1d(num_hidden))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, x):
        h = x

        for i, layer in enumerate(self.layers[:-1]):
            h = layer(g, h)
            h = self.bns[i](h)
            h = F.relu(h)
            h = self.dropout(h)
        h = self.layers[-1](g, h)
        return h.log_softmax(dim=-1)

split_idx = dataset.get_idx_split()
label = label.to(device)
train_idx = split_idx['train'].to(device)
model = GraphSAGE(x.shape[1], dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
evaluator = Evaluator(name='ogbn-arxiv')

best_val_acc = 0
final_test_acc = 0
for epoch in range(300):
    # train
    model.train()
    out = model(g, x)[train_idx]
    loss = F.nll_loss(out, label.squeeze(1)[train_idx])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # eval
    model.eval()
    with torch.no_grad():
        out = model(g, x)
    pred = out.argmax(dim=-1, keepdim=True)
    valid_acc = evaluator.eval({
        'y_true': label[split_idx['valid']],
        'y_pred': pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': label[split_idx['test']],
        'y_pred': pred[split_idx['test']],
    })['acc']
    if valid_acc > best_val_acc:
        best_val_acc = valid_acc
        final_test_acc = test_acc
        print('Epoch {:d} | Best Val Acc {:.4f}'.format(epoch, best_val_acc))
print('Test Acc {:.4f}'.format(test_acc))