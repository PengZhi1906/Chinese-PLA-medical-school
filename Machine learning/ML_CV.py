import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc

# 设置工作目录
os.chdir("F:\\图神经网络\\机器学习模型评估")

# 创建保存结果的文件夹
output_folder = "F:\\图神经网络\\机器学习模型评估\\"
os.makedirs(output_folder, exist_ok=True)

# 读取数据
input_file_path = "STAD.csv"
data = pd.read_csv(input_file_path)

# One-Hot Encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# 提取特征和标签
X = data_encoded.drop("label", axis=1).values
y = data_encoded["label"].values

# 确保数据是 C-contiguous
X = np.ascontiguousarray(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化模型
models = {
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# 定义交叉验证评估指标
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# 对每个模型进行 5 折交叉验证并保存结果
for model_name, model in models.items():
    print(f"Running 5-fold cross-validation for {model_name}...")

    # 进行交叉验证
    cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring, return_train_score=False)

    # 创建 DataFrame
    cv_results_df = pd.DataFrame({
        "Fold": [1, 2, 3, 4, 5],  # 5 折
        "Accuracy": cv_results['test_accuracy'],
        "Precision": cv_results['test_precision'],
        "Recall": cv_results['test_recall'],
        "F1-score": cv_results['test_f1'],
        "AUC": cv_results['test_roc_auc']
    })



    # 组合最终表格
    cv_results_df = pd.concat([cv_results_df,], ignore_index=True)

    # 保存交叉验证结果到 CSV 文件
    model_results_file = os.path.join(output_folder, f"{model_name}_cv_results.csv")
    cv_results_df.to_csv(model_results_file, index=False)

    print(f"Cross-validation results for {model_name} saved to {model_results_file}")
