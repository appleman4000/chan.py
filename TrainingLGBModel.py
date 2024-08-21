# cython: language_level=3
import csv
import json
import pickle
import threading
from typing import TypedDict

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from optuna_dashboard import run_server
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from ChanModel.Features import CFeatures
from Common.CTime import CTime
from Plot.PlotDriver import CPlotDriver


class T_SAMPLE_INFO(TypedDict):
    feature: CFeatures
    is_buy: bool
    open_time: CTime


def plot(chan, plot_marker):
    plot_config = {
        "plot_kline": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_zs": True,
        "plot_bsp": True,
        "plot_marker": True,
    }
    plot_para = {
        "figure": {
            "x_range": 400,
        },
        "marker": {
            "markers": plot_marker
        }
    }
    plot_driver = CPlotDriver(
        chan,
        plot_config=plot_config,
        plot_para=plot_para,
    )
    plot_driver.save2img("label.png")


alpha = 0.25
gamma = 1

param_grid = {
    'seed': 42,
    'device': 'cpu',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'min_split_gain': 0,
    'min_child_weight': 1e-3,
    'verbose': -1,
    'boosting_type': 'gbdt',
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'reg_alpha': 0.0,
}


# param_grid = {
#     'random_state': 42
# }


def objective(trial):
    # 使用 Optuna 定义超参数的搜索空间

    param_grid.update({
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        # 'reg_alpha': trial.suggest_float('reg_alpha', 0, 500.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 500.0)
    })
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    param_grid.update(
        {
            "class_weight": {
                0: class_weights[0],
                1: class_weights[1],
            }
        }
    )
    param_grid["random_state"] = param_grid["seed"]
    param_grid["bagging_seed"] = param_grid["seed"]
    param_grid["feature_fraction_seed"] = param_grid["seed"]
    param_grid["gpu_device_id"] = 0
    param_grid["gpu_platform_id"] = 0

    model = LGBMClassifier(**param_grid)

    model.fit(X_train, y_train)

    # 在验证集上预测
    y_pred = model.predict(X_val)

    # 计算 F1 分数（适用于二分类）
    score = roc_auc_score(y_val, y_pred)

    # 返回平均 F1 分数
    return score


# def objective(trial):
#     # 使用 Optuna 定义超参数的搜索空间
#     param_grid.update({
#         "max_depth": trial.suggest_int('max_depth', 1, 20),
#         "min_samples_split": trial.suggest_int('min_samples_split', 2, 100),
#         "min_samples_leaf": trial.suggest_int('min_samples_leaf', 2, 100),
#         "criterion": trial.suggest_categorical('criterion', ['gini', 'entropy'])
#     })
#     class_weights = class_weight.compute_class_weight(
#         "balanced", classes=np.unique(train_label), y=train_label
#     )
#     param_grid.update(
#         {
#             "class_weight": {
#                 0: class_weights[0],
#                 1: class_weights[1],
#             }
#         }
#     )
#     X, y = train_data, train_label
#     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, shuffle=False)
#
#     model = DecisionTreeClassifier(**param_grid)
#
#     model.fit(X_train, y_train)
#
#     # 在验证集上预测
#     y_pred = model.predict(X_valid)
#
#     # 计算 F1 分数（适用于二分类）
#     score = f1_score(y_valid, y_pred)
#
#     # 返回平均 F1 分数
#     return score


# def objective(trial):
#     # 使用 Optuna 定义超参数的搜索空间
#     param_grid.update({
#         "n_estimators": trial.suggest_int('n_estimators', 10, 500),
#         "max_depth": trial.suggest_int('max_depth', 1, 50),
#         "min_samples_split": trial.suggest_int('min_samples_split', 2, 50),
#         "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 50),
#         "max_features": trial.suggest_categorical('max_features', ['sqrt', 'log2']),
#         "criterion": trial.suggest_categorical('criterion', ["gini", "entropy", "log_loss"]),
#         "min_weight_fraction_leaf": trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),  # 最小加权样本比例
#         "max_leaf_nodes": trial.suggest_int('max_leaf_nodes', 10, 1000),  # 最大叶节点数
#     })
#     class_weights = class_weight.compute_class_weight(
#         "balanced", classes=np.unique(train_label), y=train_label
#     )
#     param_grid.update(
#         {
#             "class_weight": {
#                 0: class_weights[0],
#                 1: class_weights[1],
#             }
#         }
#     )
#     X, y = train_data, train_label
#     # 数据标准化
#     # 定义 StratifiedKFold 交叉验证
#     folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
#     scores = []
#
#     # 对每个折叠进行训练和验证
#     for train_index, valid_index in folds.split(X, y):
#         X_train, X_valid = X[train_index], X[valid_index]
#         y_train, y_valid = y[train_index], y[valid_index]
#
#         model = RandomForestClassifier(**param_grid)
#
#         model.fit(X_train, y_train)
#
#         # 在验证集上预测
#         y_pred = model.predict(X_valid)
#
#         # 计算 F1 分数（适用于二分类）
#         score = roc_auc_score(y_valid, y_pred)
#         scores.append(score)
#
#     # 返回平均 F1 分数
#     return sum(scores) / len(scores)


def get_factors(obj):
    results = {}
    for attr_name, attr_value in obj.__class__.__dict__.items():
        if callable(attr_value) and attr_name != '__init__':
            results.update(attr_value(obj))
    return results


def load_dataset_from_csv(csv_file, meta, bsp_type):
    images = []
    labels = []
    features = []
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            label = int(row[0])  # Read the label (convert it to int if necessary)
            bs_type = row[1]
            if bs_type not in bsp_type:
                continue
            file_path = row[2]  # Read the filepath

            feature = row[3]
            feature = {int(k): float(v) for k, v in (item.split(':') for item in feature.split())}

            missing = 0
            feature_arr = [missing] * len(meta)
            for feat_name, feat_value in feature.items():
                if feat_name in meta.values():
                    feature_arr[feat_name] = feat_value
            # Process the label and filepath as needed
            print(f"Label: {label}, Filepath: {file_path}")
            labels.append(label)
            features.append(feature_arr)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32),


def get_all_in_one_dataset(codes, bsp_type):
    X_trains, X_vals, y_trains, y_vals = [], [], [], []

    for code in codes:
        meta = json.load(open(f"./TMP/{code}_feature.meta", "r"))
        features, labels = load_dataset_from_csv(f"./TMP/{code}_dataset.csv", bsp_type=bsp_type, meta=meta)
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, shuffle=False,
                                                          random_state=42)
        X_trains.extend(X_train)
        X_vals.extend(X_val)
        y_trains.extend(y_train)
        y_vals.extend(y_val)
        # 追加新数据到数据集中

    return np.array(X_trains, dtype=np.float32), np.array(X_vals, dtype=np.float32), \
        np.array(y_trains, dtype=np.float32), np.array(y_vals, dtype=np.float32)


if __name__ == "__main__":
    symbols = [
        # Major
        "EURUSD",
        "GBPUSD",
        "AUDUSD",
        "NZDUSD",
        "USDJPY",
        "USDCAD",
        "USDCHF",
        # Crosses
        # "AUDCHF",
        # "AUDJPY",
        # "AUDNZD",
        # "CADCHF",
        # "CADJPY",
        # "CHFJPY",
        # "EURAUD",
        # "EURCAD",
        # "AUDCAD",
        # "EURCHF",
        # "GBPNZD",
        # "GBPCAD",
        # "GBPCHF",
        # "GBPJPY",
    ]
    X_train, X_val, y_train, y_val = get_all_in_one_dataset(symbols, bsp_type=["2", "2s"])
    # 创建 Optuna 优化器
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)


    def start_dashboard():
        run_server(storage, host="127.0.0.1", port=8080)


    # 启动一个后台线程来运行 Optuna Dashboard
    dashboard_thread = threading.Thread(target=start_dashboard)
    dashboard_thread.start()
    study.optimize(objective, n_trials=1000, n_jobs=-1)

    # 输出最佳结果
    print('Best trial:', study.best_trial.params)

    # 使用最佳参数训练最终模型
    best_params = study.best_trial.params
    param_grid.update(best_params)
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    param_grid.update(
        {
            "class_weight": {
                0: class_weights[0],
                1: class_weights[1],
            }
        }
    )
    classifier1 = LGBMClassifier(**param_grid)
    # 训练 Pipeline
    classifier1.fit(X_train, y_train)
    meta = json.load(open(f"./TMP/EURUSD_feature.meta", "r"))
    feature_names = meta.keys()

    feature_importances = classifier1.feature_importances_
    # 特征名称

    # 创建特征重要性数据框
    import pandas as pd

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    pd.set_option('display.max_rows', None)
    print(importance_df)

    with open("model.hdf5", "wb") as f:
        pickle.dump(classifier1, f)
        # pickle.dump(scaler, f)
    print("train LogisticRegression model completely!")
