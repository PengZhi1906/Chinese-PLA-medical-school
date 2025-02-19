import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    roc_curve, auc
import matplotlib.pyplot as plt

# 设置工作目录
os.chdir("F:\\图神经网\\机器学习模型评估")

# 创建保存图表和数据的文件夹
output_folder = "F:\\图神经网络\\机器学习模型评估"
os.makedirs(output_folder, exist_ok=True)

# 读取数据文件路径
input_file_path = "STAD.csv"

# 输出文件路径
output_file_path = "model_metrics_summary.csv"

# 读取数据
data = pd.read_csv(input_file_path)

# One-Hot Encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# 提取特征和标签
X = data_encoded.drop("label", axis=1)  # 特征
y = data_encoded["label"]  # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化所有模型
svm_model = SVC(probability=True)
rf_model = RandomForestClassifier()
xgb_model = XGBClassifier()
lr_model = LogisticRegression()
dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()
nb_model = GaussianNB()

# 存储模型及其名称
models = {
    "SVM": svm_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "Logistic Regression": lr_model,
    "Decision Tree": dt_model,
    "KNN": knn_model,
    "Naive Bayes": nb_model
}

# 存储评价指标和图表
metrics_dict = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": [],
    "AUC": []
}

# 绘制所有模型的 ROC 曲线
plt.figure(figsize=(10, 8))

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # 计算各个模型的性能指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_value = roc_auc_score(y_test, y_proba[:, 1])

    # 存储评价指标
    metrics_dict["Model"].append(model_name)
    metrics_dict["Accuracy"].append(accuracy)
    metrics_dict["Precision"].append(precision)
    metrics_dict["Recall"].append(recall)
    metrics_dict["F1 Score"].append(f1)
    metrics_dict["AUC"].append(auc_value)

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - All Models')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_folder, "all_models_roc_curve.png"))
plt.close()

# 创建汇总表格
summary_df = pd.DataFrame(metrics_dict)

# 输出到CSV文件
summary_df.to_csv(os.path.join(output_folder, output_file_path), index=False)
print(f"Model metrics summary saved to {output_folder}/{output_file_path}")
