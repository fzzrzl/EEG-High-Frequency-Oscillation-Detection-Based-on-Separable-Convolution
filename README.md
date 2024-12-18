# 项目名称

## 介绍

为保护知识产权，本项目中涉及模型及数据处理的代码已打包为 `.so` 文件。

## 脚本说明

### 1. 训练模型脚本 (`train_model.py`)

用于训练模型的脚本。
### 示例
```bash
python train_model.py --train_path=r'/home/train/*.csv' --val_path=r'/home/validation/*.csv' --save_path='model_conv_kernel17_9459.pth' --device='cuda:0' 
```
* train_path: 训练数据路径（CSV 格式）。
* val_path: 验证数据路径（CSV 格式）。
* save_path: 模型保存路径。

### 2. 获取预测结果脚本 (`get_results.py`)
获取预测结果prediction_lables.txt。
### 示例
```bash
python get_results.py --test_path=r'/home/test/' --save_path='model_conv_kernel17_9459.pth' --device='cuda:0'
```
* test_path: 测试数据集文件夹地址。
* save_path: 模型保存地址。

### 3. 计算 F1 分数脚本 (`cal_F1.py`)
输出classification report
### 示例
```bash
python cal_F1.py
```

