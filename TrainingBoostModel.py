# cython: language_level=3
import csv
import pickle
import threading
from typing import TypedDict

import lightgbm as lgb
import numpy as np
import optuna
import pandas
import pandas as pd
from lightgbm import LGBMClassifier
from optuna_dashboard import run_server
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from ChanModel.Features import CFeatures
from Common.CTime import CTime


class T_SAMPLE_INFO(TypedDict):
    feature: CFeatures
    is_buy: bool
    open_time: CTime


alpha = 0.25
gamma = 1

param_grid = {
    'seed': 42,
    'device': 'cpu',
    'objective': 'binary',
    'min_split_gain': 0,
    'min_child_weight': 1e-3,
    'verbose': -1,
    'boosting_type': 'gbdt',
}


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
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 100.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 100.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),

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
    callbacks = [lgb.log_evaluation(period=1), lgb.early_stopping(stopping_rounds=10, verbose=False)]
    model.fit(X_train, y_train, eval_set=(X_val, y_val), eval_metric="auc", callbacks=callbacks)

    # 在验证集上预测
    y_pred = model.predict_proba(X_val)[:, 1]

    # 计算 F1 分数（适用于二分类）
    score = roc_auc_score(y_val, y_pred)

    # 返回平均 F1 分数
    return score


# param_grid = {
#     'random_state': 42,
#     "objective": "Logloss",
#     "eval_metric": "AUC"
# }
#
#
# def objective(trial):
#     # 使用 Optuna 定义超参数的搜索空间
#     param_grid.update({
#         # "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
#         # "loss_function": "MultiClass",  # trial.suggest_categorical("loss_function", ["MultiClass", "MultiClassOneVsAll"]),
#         # "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
#         "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 1e2, log=True),
#         "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
#         "depth": trial.suggest_int("depth", 1, 10),
#         "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
#         "bootstrap_type": trial.suggest_categorical(
#             "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
#         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20),
#         "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 20),
#     })
#
#     class_weights = class_weight.compute_class_weight(
#         "balanced", classes=np.unique(y_train), y=y_train
#     )
#     param_grid.update(
#         {
#             "class_weights": {
#                 0: class_weights[0],
#                 1: class_weights[1],
#             }
#         }
#     )
#
#     model = CatBoostClassifier(**param_grid)
#
#     model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0, early_stopping_rounds=100)
#
#     # 在验证集上预测
#     y_pred = model.predict(X_val)
#
#     # 计算 F1 分数（适用于二分类）
#     score = roc_auc_score(y_val, y_pred)
#
#     # 返回平均 F1 分数
#     return score


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


def load_dataset_from_csv(csv_file, meta, bsp_type):
    labels = []
    features = []
    df = pandas.DataFrame(csv_file)
    labels = pd["label"]
    features = df.iloc[:, 2:]

    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            label = int(row[0])  # Read the label (convert it to int if necessary)
            bs_types = row[1]
            if np.array([t not in bs_types for t in bsp_type]).all():
                continue
            file_path = row[2]  # Read the filepath

            feature = row[3]
            feature = {int(k): float(v) for k, v in (item.split(':') for item in feature.split())}

            missing = float('nan')
            feature_arr = [missing] * len(meta)
            for feat_name, feat_value in feature.items():
                if feat_name in meta.values():
                    feature_arr[feat_name] = feat_value
            # Process the label and filepath as needed
            # print(f"Label: {label}, Filepath: {file_path}")
            labels.append(label)
            features.append(feature_arr)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32),


def get_all_in_one_dataset(codes, bsp_type):
    df = pd.DataFrame()
    for code in codes:
        new_df = pd.read_csv(f"./TMP/{code}_dataset.csv", index_col=0)
        df = pd.concat([df, new_df])
    df = df[df['bsp_type'].isin(bsp_type)]
    labels = df["label"].to_numpy(dtype=int)
    features = df.iloc[:, 3:].to_numpy(dtype=np.float32)
    feature_names = df.columns[3:].tolist()

    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, shuffle=False,
                                                      random_state=42)

    return X_train, X_val, y_train, y_val, feature_names


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
        "AUDCHF",
        "AUDJPY",
        "AUDNZD",
        "CADCHF",
        "CADJPY",
        "CHFJPY",
        "EURAUD",
        "EURCAD",
        "AUDCAD",
        "EURCHF",
        "GBPNZD",
        "GBPCAD",
        "GBPCHF",
        "GBPJPY",
    ]
    bsp_type = ["1", "1p", "1-1p"]
    # bsp_type = ["2", "2s", "2-2s"]
    X_train, X_val, y_train, y_val, feature_names = get_all_in_one_dataset(symbols, bsp_type=bsp_type)
    with open(f"./TMP/all_in_one_{'_'.join(bsp_type)}_model.meta", "wb") as fid:
        # meta保存下来，实盘预测时特征对齐用
        pickle.dump(feature_names, fid)


    def start_dashboard():
        run_server(storage, host="127.0.0.1", port=8080)


    storage = optuna.storages.InMemoryStorage()
    dashboard_thread = threading.Thread(target=start_dashboard)
    dashboard_thread.start()

    print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
    # 创建 Optuna 优化器

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)

    # 启动一个后台线程来运行 Optuna Dashboard

    study.optimize(objective, n_trials=1000, n_jobs=-1)
    # 输出最佳结果
    print('Best trial:', study.best_trial.params)
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
    classifier = LGBMClassifier(**param_grid)
    # 训练 Pipeline
    callbacks = [lgb.log_evaluation(period=1), lgb.early_stopping(stopping_rounds=100, verbose=False)]
    classifier.fit(X_train, y_train, eval_set=(X_val, y_val), eval_metric='auc', callbacks=callbacks)
    feature_importances = classifier.feature_importances_
    # 特征名称
    # 创建特征重要性数据框
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    pd.set_option('display.max_rows', None)
    print(importance_df)

    with open(f"./TMP/all_in_one_{'_'.join(bsp_type)}_model.hdf5", "wb") as f:
        pickle.dump(classifier, f)
    print("train model completely!")
