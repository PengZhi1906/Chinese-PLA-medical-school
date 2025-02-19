import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# 数据文件路径
data_files = [
    'F:\\图神经网络\\dignose0.7.csv',
    'F:\\图神经网络\\dignose0.8.csv'
]
data_labels = ['7 biomarkers', '3 biomarkers']
# 主输出文件夹
main_output_folder = 'F:\\图神经网络\\test'
os.makedirs(main_output_folder, exist_ok=True)

# 创建蓝色渐变色
blue_colors = plt.cm.Blues(np.linspace(0.5, 1, len(data_files)))

# 绘制ROC曲线
plt.figure()

for file_index, file_path in enumerate(data_files):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 使用SMOTE进行数据增强
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # 定义交叉验证策略
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = LogisticRegression(class_weight='balanced', solver='saga', max_iter=10000, penalty='l2', C=0.01)

    # 使用交叉验证评估模型
    cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring='roc_auc')
    mean_auc = np.mean(cv_scores)
    std_auc = np.std(cv_scores)

    # 在整个数据集上训练模型
    model.fit(X_resampled, y_resampled)

    # 在测试集上进行预测
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_pred = model.predict(X_test)

    # 计算评价指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # 保存评价指标
    metrics_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_evaluation_metrics.txt"
    metrics_file = os.path.join(main_output_folder, metrics_filename)
    with open(metrics_file, 'w') as f:
        f.write(f"Cross-validated AUC: {mean_auc:.4f} ± {std_auc:.4f}\n")
        f.write(
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Test AUC: {auc:.4f}\n")

    # 保存模型
    model_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_model.pkl"
    model_file = os.path.join(main_output_folder, model_filename)
    joblib.dump(model, model_file)

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, color=blue_colors[file_index], label=f'{data_labels[file_index]}(AUC = {auc:.2f})')

# 设置图表
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Dignose Model')
plt.legend()

# 保存ROC曲线图
roc_curve_filename = 'combined_roc_curve.png'
roc_curve_file = os.path.join(main_output_folder, roc_curve_filename)
plt.savefig(roc_curve_file)
plt.close()
