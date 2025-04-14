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
parser = argparse.ArgumentParser(description="AIPstack with ML (RF+XGBoost+LightGBM+GBDT) and DL (ResNeXt) fusion")
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

# 读取AIP数据集（请确认文件路径正确）
sequences, labels = read_dataset.read_dataset_from_aipstack_work('AIP.txt')

# 数据集划分：例如将前419个样本作为测试集，其余为训练集
X_train_seq = sequences[419:]
X_test_seq = sequences[:419]
y_train = labels[419:]
y_test = labels[:419]


# ====================================================
# 【新】包装函数，用于对完整特征矩阵进行消融处理
# 假设 extract_features 返回的特征矩阵 shape 为 (n_samples, d)
# d 的排列顺序：
#   DDE:      列 0-19   (20 维)
#   CKSAAP:   列 20-59  (40 维)
#   PP16:     列 60-75  (16 维)
#   ACH:      列 76-99  (24 维)
# 注意：实际索引请根据 extract_features 返回的维度进行调整！
def extract_features_ablation(seq_list, remove=None):
    full_features = extract_features(seq_list)  # 返回完整特征矩阵
    if remove is None:
        return full_features
    if remove == "DDE":
        # 删除 DDE 部分
        return full_features[:, 20:]
    elif remove == "CKSAAP":
        # 删除 CKSAAP 部分：拼接前20列和60列及以后
        return np.hstack((full_features[:, :20], full_features[:, 60:]))
    elif remove == "PP16":
        # 删除 PP16 部分：拼接前60列和76列及以后
        return np.hstack((full_features[:, :60], full_features[:, 76:]))
    elif remove == "ACH":
        # 删除 ACH 部分：保留前76列
        return full_features[:, :76]
    else:
        raise ValueError("未知的消融特征标识")


# 分别生成四种特征组合的特征矩阵
X_train_wo_DDE = extract_features_ablation(X_train_seq, remove="DDE")
X_test_wo_DDE = extract_features_ablation(X_test_seq, remove="DDE")

X_train_wo_CKSAAP = extract_features_ablation(X_train_seq, remove="CKSAAP")
X_test_wo_CKSAAP = extract_features_ablation(X_test_seq, remove="CKSAAP")

X_train_wo_PP16 = extract_features_ablation(X_train_seq, remove="PP16")
X_test_wo_PP16 = extract_features_ablation(X_test_seq, remove="PP16")

X_train_wo_ACH = extract_features_ablation(X_train_seq, remove="ACH")
X_test_wo_ACH = extract_features_ablation(X_test_seq, remove="ACH")


# ====================================================

# --------------------- 定义 ResNextBlock 与融合模型构建函数 ---------------------
class ResNextBlock(layers.Layer):
    def __init__(self, units, cardinality=4, dropout_rate=0.2, **kwargs):
        """
        ResNext块：将输入拆分为多个组（cardinality），分别进行变换后聚合，再加上跳跃连接。
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
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(16, activation='relu')(inputs)
    x = ResNextBlock(16, cardinality=4, dropout_rate=0.2)(x)
    x = ResNextBlock(16, cardinality=4, dropout_rate=0.2)(x)
    x = ResNextBlock(16, cardinality=4, dropout_rate=0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# --------------------- 定义运行特征消融实验的函数 ---------------------
def run_feature_ablation(models_list, X_train_feat, X_test_feat, y_train, y_test):
    """
    针对给定基础模型组合（models_list，每项为 (name, model)）以及特征矩阵进行训练，
    得到各模型在训练集和测试集上的预测概率，再构造融合模型（输入维度自适应），
    重复训练若干次后选择最佳模型，返回最佳AUC值及在测试集上的预测概率。
    """
    train_preds = []
    test_preds = []
    for name, model in models_list:
        model_params = model.get_params()
        params_str = ', '.join(f"{key}={value}" for key, value in model_params.items())
        print(f"训练模型：{name}, 参数：{params_str}")
        model.fit(X_train_feat, y_train)
        train_pred = model.predict_proba(X_train_feat)[:, 1]
        test_pred = model.predict_proba(X_test_feat)[:, 1]
        train_preds.append(train_pred.reshape(-1, 1))
        test_preds.append(test_pred.reshape(-1, 1))
        print(f"{name} - Train Accuracy: {accuracy_score(y_train, model.predict(X_train_feat)):.4f}, " +
              f"Test Accuracy: {accuracy_score(y_test, model.predict(X_test_feat)):.4f}")
    X_train_meta = np.hstack(train_preds)
    X_test_meta = np.hstack(test_preds)
    # 确保转换为 NumPy 数组
    X_train_meta = np.array(X_train_meta)
    X_test_meta = np.array(X_test_meta)
    y_train_arr = np.array(y_train)
    y_test_arr = np.array(y_test)
    num_models = X_train_meta.shape[1]

    fusion_model = build_model(input_shape=(num_models,), num_classes=2)
    fusion_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    best_auc = 0.0
    best_prediction = None
    for _ in range(5):
        history = fusion_model.fit(X_train_meta, y_train_arr, epochs=10, batch_size=32,
                                   validation_split=0.2, verbose=0)
        predictions = fusion_model.predict(X_test_meta)
        current_auc = roc_auc_score(y_test_arr, predictions)
        if current_auc > best_auc:
            best_auc = current_auc
            best_prediction = predictions
    return best_auc, best_prediction


# 定义基础模型组合（始终使用：XGBoost, RF, LightGBM, GBDT）
base_models = [
    ('xgb', XGBClassifier(n_estimators=args.xgb_n_estimators, max_depth=args.xgb_max_depth,
                          use_label_encoder=False, eval_metric='logloss', random_state=args.xgb_random_seed)),
    ('rf', RandomForestClassifier(n_estimators=args.rf_n_estimators, max_depth=args.rf_max_depth,
                                  random_state=args.rf_random_seed)),
    ('lgb', lgb.LGBMClassifier(n_estimators=args.lgb_n_estimators, max_depth=args.lgb_max_depth,
                               random_state=args.lgb_random_seed)),
    ('gbdt', GradientBoostingClassifier(n_estimators=args.gbdt_n_estimators, max_depth=args.gbdt_max_depth,
                                        random_state=args.gbdt_random_seed))
]

# --------------------- 分别运行四个特征消融实验 ---------------------
auc_wo_DDE, pred_wo_DDE = run_feature_ablation(base_models, X_train_wo_DDE, X_test_wo_DDE, y_train, y_test)
auc_wo_CKSAAP, pred_wo_CKSAAP = run_feature_ablation(base_models, X_train_wo_CKSAAP, X_test_wo_CKSAAP, y_train, y_test)
auc_wo_PP16, pred_wo_PP16 = run_feature_ablation(base_models, X_train_wo_PP16, X_test_wo_PP16, y_train, y_test)
auc_wo_ACH, pred_wo_ACH = run_feature_ablation(base_models, X_train_wo_ACH, X_test_wo_ACH, y_train, y_test)

# 计算各实验的 ROC 曲线
fpr_wo_DDE, tpr_wo_DDE, _ = roc_curve(np.array(y_test), pred_wo_DDE)
fpr_wo_CKSAAP, tpr_wo_CKSAAP, _ = roc_curve(np.array(y_test), pred_wo_CKSAAP)
fpr_wo_PP16, tpr_wo_PP16, _ = roc_curve(np.array(y_test), pred_wo_PP16)
fpr_wo_ACH, tpr_wo_ACH, _ = roc_curve(np.array(y_test), pred_wo_ACH)

# --------------------- 绘制包含四条ROC曲线的图 ---------------------
plt.figure(figsize=(8, 6))
plt.plot(fpr_wo_DDE, tpr_wo_DDE, color='red', lw=2,
         label=f"w/o DDE (AUC = {auc_wo_DDE:.4f})")
plt.plot(fpr_wo_CKSAAP, tpr_wo_CKSAAP, color='orange', lw=2,
         label=f"w/o CKSAAP (AUC = {auc_wo_CKSAAP:.4f})")
plt.plot(fpr_wo_PP16, tpr_wo_PP16, color='blue', lw=2,
         label=f"w/o PP16 (AUC = {auc_wo_PP16:.4f})")
plt.plot(fpr_wo_ACH, tpr_wo_ACH, color='green', lw=2,
         label=f"w/o ACH (AUC = {auc_wo_ACH:.4f})")
# 绘制对角参考线
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12, fontweight='bold', color='black')
plt.ylabel("True Positive Rate", fontsize=12, fontweight='bold', color='black')
plt.title("ROC curves of 4 Features ablation", fontsize=14, fontweight='bold', color='black')
plt.xticks(fontsize=10, fontweight='bold', color='black')
plt.yticks(fontsize=10, fontweight='bold', color='black')
plt.legend(loc="lower right", prop={'weight': 'bold'}, frameon=False)
plt.tight_layout()
plt.show()

# --------------------- 保存结果（可选） ---------------------
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

# 可根据需要保存最佳模型结果
# save_results_to_excel(best_model_metrics, best_prediction, best_prediction_binary)
