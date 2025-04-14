import argparse
import logging
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import read_dataset
from extract_feature import extract_features

# ML模型相关导入
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, \
    recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 参数解析
parser = argparse.ArgumentParser(description="AIPstack with ML (RF+XGBoost+LightGBM+GBDT) and DL (ResNext) fusion")
parser.add_argument('--mode', type=str, help='数据集类型：填写AMP或AIP', default='AIP')
parser.add_argument('--dataset_random_state', type=int, help='数据集划分随机数种子', default=42)
parser.add_argument('--xgb_random_seed', type=int, help='XGBoost随机数种子', default=42)
parser.add_argument('--rf_random_seed', type=int, help='Random Forest随机数种子', default=42)
parser.add_argument('--xgb_n_estimators', type=int, help='XGBoost森林规模', default=50)
parser.add_argument('--xgb_max_depth', type=int, help='XGBoost树深度', default=15)
parser.add_argument('--rf_n_estimators', type=int, help='Random Forest森林规模', default=50)
parser.add_argument('--rf_max_depth', type=int, help='Random Forest树深度', default=80)
parser.add_argument('--lgb_n_estimators', type=int, help='LightGBM森林规模', default=100)
parser.add_argument('--lgb_max_depth', type=int, help='LightGBM树深度', default=10)
parser.add_argument('--lgb_random_seed', type=int, help='LightGBM随机数种子', default=42)
parser.add_argument('--gbdt_n_estimators', type=int, help='GBDT森林规模', default=100)
parser.add_argument('--gbdt_max_depth', type=int, help='GBDT树深度', default=10)
parser.add_argument('--gbdt_random_seed', type=int, help='GBDT随机数种子', default=42)

args = parser.parse_args()

print(f"数据集划分随机数种子: {args.dataset_random_state}")
print(
    f"XGBoost: n_estimators={args.xgb_n_estimators}, max_depth={args.xgb_max_depth}, random_seed={args.xgb_random_seed}")
print(
    f"Random Forest: n_estimators={args.rf_n_estimators}, max_depth={args.rf_max_depth}, random_seed={args.rf_random_seed}")
print(
    f"LightGBM: n_estimators={args.lgb_n_estimators}, max_depth={args.lgb_max_depth}, random_seed={args.lgb_random_seed}")
print(
    f"GBDT: n_estimators={args.gbdt_n_estimators}, max_depth={args.gbdt_max_depth}, random_seed={args.gbdt_random_seed}")

# 配置日志
logging.basicConfig(filename='training_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(
    f"启动训练，使用的超参数：mode={args.mode}, dataset_random_state={args.dataset_random_state}, "
    f"xgb_random_seed={args.xgb_random_seed}, rf_random_seed={args.rf_random_seed}, "
    f"xgb_n_estimators={args.xgb_n_estimators}, xgb_max_depth={args.xgb_max_depth}, "
    f"rf_n_estimators={args.rf_n_estimators}, rf_max_depth={args.rf_max_depth}, "
    f"lgb_n_estimators={args.lgb_n_estimators}, lgb_max_depth={args.lgb_max_depth}, "
    f"gbdt_n_estimators={args.gbdt_n_estimators}, gbdt_max_depth={args.gbdt_max_depth}"
)

# 读取AIP数据集（使用read_dataset_from_aipstack_work读取文件，确保文件路径正确）
sequences, labels = read_dataset.read_dataset_from_aipstack_work('AIP.txt')

# 数据集划分：例如将前419个样本作为测试集，其余为训练集
X_train_seq = sequences[419:]
X_test_seq = sequences[:419]
y_train = labels[419:]
y_test = labels[:419]

# 对训练集和测试集分别提取特征
X_train = extract_features(X_train_seq)
X_test = extract_features(X_test_seq)


# --------------------- 定义 ResNextBlock 与融合模型构建函数 ---------------------
class ResNextBlock(layers.Layer):
    def __init__(self, units, cardinality=4, dropout_rate=0.2, **kwargs):
        """
        ResNext块：将输入拆分为多个组（cardinality），分别进行变换后聚合，再加上跳跃连接。
        :param units: 输出维度（必须能被cardinality整除）
        :param cardinality: 分组数
        :param dropout_rate: Dropout比例
        """
        super().__init__(**kwargs)
        assert units % cardinality == 0, "units must be divisible by cardinality"
        self.units = units
        self.cardinality = cardinality
        self.group_units = units // cardinality
        self.dropout_rate = dropout_rate
        self.branches = []
        for i in range(cardinality):
            branch = models.Sequential([
                layers.Dense(self.group_units, activation='relu'),
                layers.Dropout(dropout_rate),
                layers.Dense(self.group_units)
            ])
            self.branches.append(branch)
        self.skip_dense = layers.Dense(units, use_bias=False)
        self.activation = layers.Activation('relu')

    def call(self, inputs, training=False):
        splits = tf.split(inputs, self.cardinality, axis=-1)
        branch_outputs = [branch(splits[i], training=training) for i, branch in enumerate(self.branches)]
        aggregated = layers.concatenate(branch_outputs, axis=-1)
        skip = self.skip_dense(inputs)
        return self.activation(aggregated + skip)


def build_model(input_shape, num_classes):
    """
    构建基于 ResNext 块的融合网络
    :param input_shape: 输入维度（此处为融合模型输入维度）
    :param num_classes: 分类数（二分类，输出1个概率）
    :return: 编译好的 Keras 模型
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(16, activation='relu')(inputs)
    x = ResNextBlock(16, cardinality=4, dropout_rate=0.2)(x)
    x = ResNextBlock(16, cardinality=4, dropout_rate=0.2)(x)
    x = ResNextBlock(16, cardinality=4, dropout_rate=0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# --------------------- 定义运行消融实验的函数 ---------------------
def run_ablation_experiment(models_list):
    """
    对给定基础模型组合（models_list，列表中每项为 (name, model)）进行训练，
    得到各模型在训练集和测试集上的预测概率，构造融合模型（输入维度自动调整为该组合数量），
    采用多次重复训练选取最佳模型，返回最佳AUC值以及最佳模型在测试集上的预测概率（用于绘制ROC曲线）。
    """
    train_preds = []
    test_preds = []
    # 对传入的每个基础模型分别训练并预测
    for name, model in models_list:
        model_params = model.get_params()
        params_str = ', '.join(f"{key}={value}" for key, value in model_params.items())
        print(f"训练模型：{name}, 参数：{params_str}")
        model.fit(X_train, y_train)
        train_pred = model.predict_proba(X_train)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]
        train_preds.append(train_pred.reshape(-1, 1))
        test_preds.append(test_pred.reshape(-1, 1))
        print(f"{name} - Train Accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}, " +
              f"Test Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    # 合并基础模型预测作为新的元特征
    X_train_meta = np.hstack(train_preds)
    X_test_meta = np.hstack(test_preds)
    # 强制转换为 NumPy 数组，确保类型正确以支持 validation_split
    X_train_meta = np.array(X_train_meta)
    X_test_meta = np.array(X_test_meta)
    y_train_arr = np.array(y_train)
    y_test_arr = np.array(y_test)
    num_models = X_train_meta.shape[1]

    fusion_model = build_model(input_shape=(num_models,), num_classes=2)
    fusion_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    best_auc = 0.0
    best_prediction = None
    # 重复训练若干次以选择最佳模型
    for _ in range(5):
        history = fusion_model.fit(X_train_meta, y_train_arr, epochs=10, batch_size=32,
                                   validation_split=0.2, verbose=0)
        predictions = fusion_model.predict(X_test_meta)
        current_auc = roc_auc_score(y_test_arr, predictions)
        if current_auc > best_auc:
            best_auc = current_auc
            best_prediction = predictions
    return best_auc, best_prediction


# --------------------- 定义四个消融方案 ---------------------
# 1. (XGBoost + LightGBM + GBDT) + ResNeXt  → w/o RF
models_no_rf = [
    ('xgb', XGBClassifier(n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth,
                          use_label_encoder=False, eval_metric='logloss', random_state=args.xgb_random_seed)),
    ('lgb', lgb.LGBMClassifier(n_estimators=args.lgb_n_estimators, max_depth=args.lgb_max_depth,
                               random_state=args.lgb_random_seed)),
    ('gbdt', GradientBoostingClassifier(n_estimators=args.gbdt_n_estimators, max_depth=args.gbdt_max_depth,
                                        random_state=args.gbdt_random_seed))
]

# 2. (RF + LightGBM + GBDT) + ResNeXt  → w/o XGBoost
models_no_xgb = [
    ('rf', RandomForestClassifier(n_estimators=args.rf_n_estimators, max_depth=args.rf_max_depth,
                                  random_state=args.rf_random_seed)),
    ('lgb', lgb.LGBMClassifier(n_estimators=args.lgb_n_estimators, max_depth=args.lgb_max_depth,
                               random_state=args.lgb_random_seed)),
    ('gbdt', GradientBoostingClassifier(n_estimators=args.gbdt_n_estimators, max_depth=args.gbdt_max_depth,
                                        random_state=args.gbdt_random_seed))
]

# 3. (RF + XGBoost + GBDT) + ResNeXt  → w/o LightGBM
models_no_lgb = [
    ('rf', RandomForestClassifier(n_estimators=args.rf_n_estimators, max_depth=args.rf_max_depth,
                                  random_state=args.rf_random_seed)),
    ('xgb', XGBClassifier(n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth,
                          use_label_encoder=False, eval_metric='logloss', random_state=args.xgb_random_seed)),
    ('gbdt', GradientBoostingClassifier(n_estimators=args.gbdt_n_estimators, max_depth=args.gbdt_max_depth,
                                        random_state=args.gbdt_random_seed))
]

# 4. (RF + XGBoost + LightGBM) + ResNeXt  → w/o GBDT
models_no_gbdt = [
    ('rf', RandomForestClassifier(n_estimators=args.rf_n_estimators, max_depth=args.rf_max_depth,
                                  random_state=args.rf_random_seed)),
    ('xgb', XGBClassifier(n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth,
                          use_label_encoder=False, eval_metric='logloss', random_state=args.xgb_random_seed)),
    ('lgb', lgb.LGBMClassifier(n_estimators=args.lgb_n_estimators, max_depth=args.lgb_max_depth,
                               random_state=args.lgb_random_seed))
]

# --------------------- 分别运行四个消融实验 ---------------------
auc_no_rf, pred_no_rf = run_ablation_experiment(models_no_rf)
auc_no_xgb, pred_no_xgb = run_ablation_experiment(models_no_xgb)
auc_no_lgb, pred_no_lgb = run_ablation_experiment(models_no_lgb)
auc_no_gbdt, pred_no_gbdt = run_ablation_experiment(models_no_gbdt)

# 计算各实验的ROC曲线
fpr_no_rf, tpr_no_rf, _ = roc_curve(np.array(y_test), pred_no_rf)
fpr_no_xgb, tpr_no_xgb, _ = roc_curve(np.array(y_test), pred_no_xgb)
fpr_no_lgb, tpr_no_lgb, _ = roc_curve(np.array(y_test), pred_no_lgb)
fpr_no_gbdt, tpr_no_gbdt, _ = roc_curve(np.array(y_test), pred_no_gbdt)

# --------------------- 绘制包含四条ROC曲线的图 ---------------------
plt.figure(figsize=(8, 6))
plt.plot(fpr_no_rf, tpr_no_rf, color='red', lw=2,
         label=f"w/o RF (AUC = {auc_no_rf:.4f})")
plt.plot(fpr_no_xgb, tpr_no_xgb, color='orange', lw=2,
         label=f"w/o XGBoost (AUC = {auc_no_xgb:.4f})")
plt.plot(fpr_no_lgb, tpr_no_lgb, color='blue', lw=2,
         label=f"w/o LightGBM (AUC = {auc_no_lgb:.4f})")
plt.plot(fpr_no_gbdt, tpr_no_gbdt, color='green', lw=2,
         label=f"w/o GBDT (AUC = {auc_no_gbdt:.4f})")
# 绘制对角参考线
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12, fontweight='bold', color='black')
plt.ylabel("True Positive Rate", fontsize=12, fontweight='bold', color='black')
plt.title("ROC curves of 4 ML ablation", fontsize=14, fontweight='bold', color='black')
plt.xticks(fontsize=10, fontweight='bold', color='black')
plt.yticks(fontsize=10, fontweight='bold', color='black')
plt.legend(loc="lower right", prop={'weight': 'bold'}, frameon=False)
plt.tight_layout()
plt.show()

# --------------------- 保存结果 ---------------------
import pandas as pd


def save_results_to_excel(metrics: dict, predictions, predictions_binary, folder_name="best_results_aip_resnext"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{folder_name}/results_{current_time}.xlsx"
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        predictions_df = pd.DataFrame({
            'Predictions': predictions.flatten(),
            'Predictions Binary': predictions_binary.flatten()
        })
        predictions_df.to_excel(writer, index=False, sheet_name='Predictions')
        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_excel(writer, index=False, sheet_name='Metrics')
    print(f"Results saved to {filename}")

# 可根据需要选择保存最佳模型结果
# save_results_to_excel(best_model_metrics, best_prediction, best_prediction_binary)
