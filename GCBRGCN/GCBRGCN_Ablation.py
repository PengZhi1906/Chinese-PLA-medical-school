import dgl
import copy
import torch
import random
import numpy as np
import pandas as pd
from model import RGCN_NET
from sklearn.metrics import accuracy_score
from sklearn import metrics


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


x = pd.read_csv("模型输入数据/features.csv")

for i in range(len(x)):
    name = x.loc[i][0]
    id2name[i] = name
    name2id[name] = i

x = x.iloc[:, 1:]
x = x.to_numpy()
print(x)

# attribute
adj = pd.read_csv("模型输入数据/Network.csv")

max_id = x.shape[0]
edges_1 = []
edges_2 = []
for i in range(len(adj)):
    start = adj.loc[i][0]
    mid = adj.loc[i][1]
    end = adj.loc[i][2]
    # if start != "NA":
    #     try:
    #         name2id[start]
    #     except:
    #         name2id[start] = max_id
    #         id2name[max_id] = start
    #         max_id += 1
    #
    # if mid != "NA":
    #     try:
    #         name2id[mid]
    #     except:
    #         name2id[mid] = max_id
    #         id2name[max_id] = mid
    #         max_id += 1
    #
    # if end != "NA":
    #     try:
    #         name2id[end]
    #     except:
    #         name2id[end] = max_id
    #         id2name[max_id] = end
    #         max_id += 1

    if 1:
        try:
            # 1
            edges_1.append([name2id[start], name2id[mid]])
        except:
            pass
        # try:
        #     edges_1.append([name2id[mid], name2id[start]])
        # except:
        #     pass
    if 1:
        try:
            edges_2.append([name2id[mid], name2id[end]])
        except:
            pass
        # try:
        #     edges_2.append([name2id[end], name2id[mid]])
        # except:
        #     pass


label = pd.read_csv("模型输入数据/label.csv")

label = pd.read_csv("./模型输入数据/消融实验/miRNA+lncRNA/节点label.csv")





y = np.zeros((x.shape[0], ))
for i in range(len(label["code"])):
    try:
        index = name2id[label["code"].iloc[i]]
        y[index] = label["label"].iloc[i]
    except:
        pass

y = torch.tensor(y).float()
print(y)

# x = np.concatenate([x, np.zeros((len(name2id)-x.shape[0], 5))], axis=0)
x = torch.tensor(x).float()

print(edges_1)
g = dgl.heterograph(
    {
        ('rna', 'type1', 'rna'): edges_1,
        ('rna', 'type2', 'rna'): edges_2
    }
)

print(g)

# logits = rgcn_net(None, None)
# y_pred = (logits >= 0.5).int().squeeze(dim=-1)

# acc = accuracy_score(y[test_id], y_pred[test_id])
# print(acc)

# print(emb["inc_rna"].shape)
# print(emb["m_rna"].shape)

#
# GCN
criterion = torch.nn.BCELoss()
rgcn_net = RGCN_NET(g, 100, 1)
optimizer = torch.optim.Adam(rgcn_net.parameters(), lr=1e-2)

idx = list(range(x.shape[0]))
random.shuffle(idx)
idx = np.array(idx)

train_id = idx[:int(x.shape[0]*0.7)]
test_id = idx[int(x.shape[0]*0.7):]

best_acc = 0
best_model = None
threshold = 0.2
best_logit = 0
for epoch in range(100):
    rgcn_net.eval()
    logits = rgcn_net(None, None)

    y_pred = (logits >= threshold).int().squeeze(dim=-1)
    acc = accuracy_score(y[test_id], y_pred[test_id])
    print(acc)
    if best_acc < acc:
        best_acc = acc
        best_model = copy.deepcopy(rgcn_net)

    rgcn_net.train()
    logits = rgcn_net(None, None)
    loss = criterion(logits.squeeze(dim=-1)[train_id], y[train_id])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print(best_acc)
best_model.eval()
logits = best_model(None, None)

y_pred = (logits >= threshold).int().squeeze(dim=-1)
acc = accuracy_score(y[test_id], y_pred[test_id])
print(acc)


# import pdb
# pdb.set_trace()

# for name, parameters in best_model.named_parameters():
#     print(name, ':', parameters.data.shape)
# for name, parameters in model.named_parameters():
#     print(name, ':', parameters.data.shape)

logits = best_model(None, None)
y_pred = (logits >= threshold).int().squeeze(dim=-1)
logits_list = logits[test_id][:, 0].data
index = sorted(range(len(logits_list)), key=lambda k: logits_list[k], reverse=True)
# print(y_pred[index])
print(test_id[index])

for i in test_id[index]:
    print(id2name[i])


for i in test_id[index]:
    print(y_pred[i].detach().numpy())

print(logits_list[index])

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
# 1.计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)
cm = np.array([[41, 23],
 [ 8, 39]])

conf_matrix = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])  # 数据有5个类别

# 画出混淆矩阵
fig, ax = plt.subplots(figsize=(4.5, 4.5))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 14}, cmap="Blues")
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
#
# # 2.计算accuracy
# print('accuracy_score', accuracy_score(y_true, y_pred))
#
# # 3.计算多分类的precision、recall、f1-score分数
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
auc = metrics.auc(fpr, tpr)

print('precision', precision_score(y_true, y_pred))
print('recall', recall_score(y_true, y_pred))
print('f1-score', f1_score(y_true, y_pred))
print('auc', auc)

#
# # 下面这个可以显示出每个类别的precision、recall、f1-score。
# print('classification_report\n', classification_report(y_true, y_pred))
