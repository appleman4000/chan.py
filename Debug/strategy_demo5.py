# cython: language_level=3
import json
import os
import pickle
import threading
from typing import Dict, TypedDict

import numpy as np
import optuna
import xgboost as xgb
from lightgbm import LGBMClassifier
from optuna_dashboard import run_server
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import class_weight

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, BSP_TYPE
from Common.CTime import CTime
from Debug.FeatureEngineering import FeatureFactors
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
    plot_driver.save2img("label.png0")


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
        "balanced", classes=np.unique(train_label), y=train_label
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
    X, y = train_data, train_label
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LGBMClassifier(**param_grid)

    model.fit(X_train, y_train)

    # 在验证集上预测
    y_pred = model.predict(X_valid)

    # 计算 F1 分数（适用于二分类）
    score = roc_auc_score(y_valid, y_pred)

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


if __name__ == "__main__":
    """
    本demo主要演示如何记录策略产出的买卖点的特征
    然后将这些特征作为样本，训练一个模型(以XGB为demo)
    用于预测买卖点的准确性

    请注意，demo训练预测都用的是同一份数据，这是不合理的，仅仅是为了演示
    """
    symbols = [
        # Major
        "EURUSD",
        # "GBPUSD",
        # "AUDUSD",
        # "NZDUSD",
        # "USDJPY",
        # "USDCAD",
        # "USDCHF",
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
        # "USDCNH",
        # "XAUUSD",
        # "XAGUSD",
    ]
    if os.path.exists("feature.libsvm"):
        os.remove("feature.libsvm")
    if os.path.exists("feature.meta"):
        os.remove("feature.meta")
    if os.path.exists("model.json"):
        os.remove("model.json")
    for code in symbols:
        begin_time = "2010-01-01 00:00:00"
        end_time = "2021-07-10 00:00:00"
        data_src = DATA_SRC.FOREX
        lv_list = [KL_TYPE.K_15M]

        config = CChanConfig({
            "trigger_step": True,  # 打开开关！
            "bi_strict": True,
            "skip_step": 500,
            "divergence_rate": float("inf"),
            "bsp2_follow_1": False,
            "bsp3_follow_1": False,
            "min_zs_cnt": 0,
            "bs1_peak": False,
            "macd_algo": "slope",
            "bs_type": '1,1p',
            "print_warning": True,
            "zs_algo": "normal",
        })

        chan = CChan(
            code=code,
            begin_time=begin_time,
            end_time=end_time,
            data_src=data_src,
            lv_list=lv_list,
            config=config,
            autype=AUTYPE.NONE,
        )

        bsp_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的bsp的特征

        # 跑策略，保存买卖点的特征
        for chan_snapshot in chan.step_load():
            last_klu = chan_snapshot[0][-1][-1]
            bsp_list = chan_snapshot.get_bsp()
            if not bsp_list:
                continue
            last_bsp = bsp_list[-1]

            cur_lv_chan = chan_snapshot[0]
            if last_bsp.klu.idx not in bsp_dict and cur_lv_chan[-1].idx == last_bsp.klu.klc.idx:
                bsp_dict[last_bsp.klu.idx] = {
                    "feature": last_bsp.features,
                    "is_buy": last_bsp.is_buy,
                    "open_time": last_klu.time,
                }
                factors = get_factors(FeatureFactors(chan))
                for key in factors.keys():
                    bsp_dict[last_bsp.klu.idx]['feature'].add_feat(key, factors[key])
                print(last_bsp.klu.time, last_bsp.is_buy)

        # 生成libsvm样本特征
        bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
        feature_meta = {}  # 特征meta
        cur_feature_idx = 0
        plot_marker = {}
        fid = open("feature.libsvm", "a")
        for bsp_klu_idx, feature_info in bsp_dict.items():
            label = int(bsp_klu_idx in bsp_academy)  # 以买卖点识别是否准确为label
            features = []  # List[(idx, value)]
            for feature_name, value in feature_info['feature'].items():
                if feature_name not in feature_meta:
                    feature_meta[feature_name] = cur_feature_idx
                    cur_feature_idx += 1
                features.append((feature_meta[feature_name], value))
            features.sort(key=lambda x: x[0])
            feature_str = " ".join([f"{idx}:{value}" for idx, value in features])
            fid.write(f"{label} {feature_str}\n")
            plot_marker[feature_info["open_time"].to_str()] = (
                "√" if label else "×", "down" if feature_info["is_buy"] else "up")
        fid.close()
        with open("feature.meta", "w") as fid:
            # meta保存下来，实盘预测时特征对齐用
            fid.write(json.dumps(feature_meta))

    # load sample file & train model
    dtrain = xgb.DMatrix("feature.libsvm?format=libsvm")  # load sample
    train_data = csr_matrix(dtrain.get_data()).toarray()
    train_label = np.array(dtrain.get_label())
    print(train_data.shape)
    print(train_label.shape)
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
        "balanced", classes=np.unique(train_label), y=train_label
    )
    param_grid.update(
        {
            "class_weight": {
                0: class_weights[0],
                1: class_weights[1],
            }
        }
    )
    # scaler = StandardScaler()
    # train_data = scaler.fit_transform(train_data)
    classifier1 = LGBMClassifier(**param_grid)
    # classifier1 = DecisionTreeClassifier(**param_grid)
    # classifier1 = RandomForestClassifier(**param_grid)
    # 训练 Pipeline
    classifier1.fit(train_data, train_label)
    feature_names = feature_meta.keys()

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
