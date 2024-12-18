from collections import defaultdict
from sklearn.metrics import classification_report

# 读取 label.txt
with open("labels.txt") as f:
    labels = [line.strip().split() for line in f.readlines()]

# 读取 predict.txt
with open("prediction_results.txt") as f:
    predictions = [line.strip().split(", Predicted: ") for line in f.readlines()]

# 创建标签字典
label_dict = {line[0]: line[1] for line in labels}

# 创建预测字典
predict_dict = {line[0].split(":")[1].split(".")[0].strip(): line[1] for line in predictions}
# 获取标签和预测列表，确保文件名一致
true_labels = [label_dict[key] for key in label_dict]
predicted_labels = [predict_dict[key] for key in label_dict]
# 计算 F1 score
report = classification_report(true_labels, predicted_labels, labels=["baseline", "spike", "ripple", "rons"])
print(report)
