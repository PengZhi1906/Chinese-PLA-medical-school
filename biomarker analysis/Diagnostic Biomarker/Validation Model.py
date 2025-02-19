import pandas as pd
import joblib
import os
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 定义蓝色渐变色列表
blue_gradients = ['#42A5F5', '#0D47A1']
# 模型和测试数据的路径

model_and_data = {
'7 biomarkers': {'model_path': 'F:\\图神经网络\\test7.csv'},
    '3 biomarkers': {'model_path': 'F:\\图神经网络\\test3.csv'}
}

# 输出文件夹
output_folder = 'F:\\图神经网络\\验证'
os.makedirs(output_folder, exist_ok=True)

# 准备绘制ROC曲线
plt.figure()

for (model_name, paths), color in zip(model_and_data.items(), blue_gradients):
    # 加载模型
    model = joblib.load(paths['model_path'])

    # 加载测试数据
    test_data = pd.read_csv(paths['data_path'])
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # 进行预测
    prediction_probabilities = model.predict_proba(X_test)[:, 1]

    # 保存预测结果
    output_file_path = os.path.join(output_folder, f'{model_name}_predictions.csv')
    pd.DataFrame(prediction_probabilities, columns=['Prediction Probabilities']).to_csv(output_file_path, index=False)

    # 计算并绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_test, prediction_probabilities)
    auc = roc_auc_score(y_test, prediction_probabilities)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})', color=color)

# 完成ROC曲线的绘制
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Validation Model')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_folder, 'combined_roc_curve.png'))
plt.show()

print("预测结果和ROC曲线图已保存。")
