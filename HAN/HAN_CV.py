import dgl
import copy
import torch
import random
import numpy as np
import pandas as pd
from model import HAN
from sklearn.metrics import accuracy_score


def setup_seed(seed):
    """
    fix the random seed
    :param seed: the random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None

setup_seed(0)

# attribute
name2id = {}
id2name = {}

x = pd.read_csv("data/features.csv")

for i in range(len(x)):
    name = x.loc[i][0]
    id2name[i] = name
    name2id[name] = i

x = x.iloc[:, 1:]
x = x.to_numpy()

# attribute

adj = pd.read_csv("data/Network.csv")

max_id = x.shape[0]
edges_1 = []
edges_2 = []
for i in range(len(adj)):
    start = adj.loc[i][0]
    mid = adj.loc[i][1]
    end = adj.loc[i][2]

    if 1:
        try:
            edges_1.append([name2id[start], name2id[mid]])
        except:
            pass
        try:
            edges_1.append([name2id[mid], name2id[start]])
        except:
            pass
    if 1:
        try:
            edges_2.append([name2id[mid], name2id[end]])
        except:
            pass
        try:
            edges_2.append([name2id[end], name2id[mid]])
        except:
            pass

label = pd.read_csv("data/label.csv")

y = np.zeros((x.shape[0], ))
for i in range(len(label["node"])):
    try:
        index = name2id[label["node"].iloc[i]]
        y[index] = label["label"].iloc[i]
    except:
        pass

y = torch.tensor(y).float()

# x = np.concatenate([x, np.zeros((len(name2id)-x.shape[0], 5))], axis=0)

x = torch.tensor(x).float()
# print(x.shape)
# print(len(id2name))
# print(len(name2id))
g = dgl.heterograph(
    {
        ('rna', 'type1', 'rna'): edges_1,
        ('rna', 'type2', 'rna'): edges_2
    }
)

# print(x.shape)
print(g)


# logits = rgcn_net(None, None)
# y_pred = (logits >= 0.5).int().squeeze(dim=-1)

# acc = accuracy_score(y[test_id], y_pred[test_id])
# print(acc)

# print(emb["inc_rna"].shape)
# print(emb["m_rna"].shape)

#
# HAN
criterion = torch.nn.BCELoss()

# print(g)

han_net = HAN(
    meta_paths=[["type1", "type2"], ["type2", "type1"]],
    in_size=x.shape[1],
    hidden_size=100,
    out_size=1,
    num_heads=[4],
    dropout=0,
)

optimizer = torch.optim.Adam(han_net.parameters(), lr=1e-3)

idx = list(range(x.shape[0]))
random.shuffle(idx)
idx = np.array(idx)

best_acc = 0

from sklearn.model_selection import KFold
import numpy as np
kf = KFold(n_splits=5,shuffle=False)  # 初始化KFold

for train_id, test_id in kf.split(x):
    # print('train_index:%s , test_index: %s ' %(train_id, test_id))

    optimizer = torch.optim.Adam(han_net.parameters(), lr=1e-3)
    best_acc = 0
    for epoch in range(100):
        logits = han_net(g, x)
        loss = criterion(logits.squeeze(dim=-1)[train_id], y[train_id])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logits = han_net(g, x)
        y_pred = (logits >= 0.2).int().squeeze(dim=-1)
        acc = accuracy_score(y[test_id], y_pred[test_id])
        # print(acc)
        if best_acc < acc:
            best_acc = acc
            best_model = copy.deepcopy(han_net)

    logits = best_model(g, x)
    y_pred = (logits >= 0.2).int().squeeze(dim=-1)

    logits_list = logits[test_id][:, 0].data
    index = sorted(range(len(logits_list)), key=lambda k: logits_list[k], reverse=True)

    acc = accuracy_score(y[test_id], y_pred[test_id])
    print(acc)

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y[test_id], y_pred[test_id])
    print(cm)

    acc = accuracy_score(y[test_id], y_pred[test_id])
    print("final", acc)

    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report

    y_true = y[test_id]
    y_pred = y_pred[test_id]

    logits = logits[test_id]
    # print('logits', logits)
    from sklearn import metrics

    fpr, tpr, thresholds = metrics.roc_curve(y_true, logits.detach().numpy())
    auc = metrics.auc(fpr, tpr)

    #
    # # 2.计算accuracy
    # print('accuracy_score', accuracy_score(y_true, y_pred))
    #
    # # 3.计算多分类的precision、recall、f1-score分数
    print("final", acc)
    print('precision', precision_score(y_true, y_pred))
    print('recall', recall_score(y_true, y_pred))
    print('f1-score', f1_score(y_true, y_pred))
    print('auc', auc)
    #

