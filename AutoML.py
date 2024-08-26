# encoding:utf-8
import logging
import warnings
from typing import Dict

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from BuySellPoint.BS_Point import CBS_Point
from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, KL_TYPE, AUTYPE, BSP_TYPE
from CommonTools import reconnect_mt5
from FeatureEngineering import FeatureFactors
from GenerateDataset import T_SAMPLE_INFO

logging.getLogger('LGBMClassifier').setLevel(logging.CRITICAL)
optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore")


def predict_bsp(model, last_bsp: CBS_Point, feature_names):
    missing = float("nan")
    feature_arr = [missing] * len(feature_names)
    for feat_name, feat_value in last_bsp.features.items():
        i = feature_names.index(feat_name)
        if i >= 0:
            feature_arr[i] = feat_value
    feature_arr = [feature_arr]
    return model.predict_proba(feature_arr)[0][1]


lv_list = [KL_TYPE.K_10M]
data_src = DATA_SRC.FOREX
trade_param_grid = {
}


def trade_from(code, begin_time, end_time, dataset_param_grid, model, feature_names, trade_param_grid):
    config = CChanConfig(conf=dataset_param_grid.copy())
    chan = CChan(
        code=code,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        begin_time=begin_time,
        end_time=end_time
    )
    capital = 10000
    lots = 1
    money = 100000 * lots
    fee = 1.0003
    long_order = 0
    short_order = 0
    for chan_snapshot in chan.step_load():

        lv_chan = chan_snapshot[0]
        profit = 0
        bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]
        if last_bsp.klu.klc.idx == lv_chan[-2].idx and (
                BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
            factors = FeatureFactors(chan[0], MAX_BI=dataset_param_grid["MAX_BI"],
                                     MAX_ZS=dataset_param_grid["MAX_ZS"],
                                     MAX_SEG=dataset_param_grid["MAX_SEG"],
                                     MAX_SEGSEG=dataset_param_grid["MAX_SEGSEG"],
                                     MAX_SEGZS=dataset_param_grid["MAX_SEGZS"]).get_factors()
            for key in factors.keys():
                last_bsp.features.add_feat(key, float(factors[key]))
            bsp1_pred = predict_bsp(model=model, last_bsp=last_bsp, feature_names=feature_names)
        else:
            bsp1_pred = 0.0
        if long_order > 0:
            # 止盈
            close_price = round(lv_chan[-1][-1].close, 5)
            long_profit = close_price / long_order - 1
            exit_rule = bsp1_pred > trade_param_grid["bsp1_pred_long_exit"] and not last_bsp.is_buy
            # 最大止盈止损保护
            tp = long_profit > trade_param_grid["tp_long"]
            sl = long_profit < -trade_param_grid["sl_long"]
            if tp or sl or exit_rule:
                long_order = 0
                profit += round(long_profit * money, 2)
        if short_order > 0:
            close_price = round(lv_chan[-1][-1].close, 5)
            short_profit = short_order / close_price - 1
            exit_rule = bsp1_pred > trade_param_grid["bsp1_pred_short_exit"] and last_bsp.is_buy
            # 最大止盈止损保护
            tp = short_profit > trade_param_grid["tp_short"]
            sl = short_profit < -trade_param_grid["sl_short"]
            if tp or sl or exit_rule:
                short_order = 0
                profit += round(short_profit * money, 2)

        if long_order == 0 and short_order == 0:
            if bsp1_pred > trade_param_grid["bsp1_pred_long_open"] and last_bsp.is_buy:
                long_order = round(lv_chan[-1][-1].close * fee, 5)
            if bsp1_pred > trade_param_grid["bsp1_pred_short_open"] and not last_bsp.is_buy:
                short_order = round(lv_chan[-1][-1].close / fee, 5)

        capital += profit
    return capital


def trade_objective(trial, code, begin_time, end_time, dataset_param_grid, model, feature_names):
    trade_param_grid.update({
        "bsp1_pred_long_open": trial.suggest_float('bsp1_pred_long_open', 0.6, 0.75, step=0.01),
        "bsp1_pred_short_open": trial.suggest_float('bsp1_pred_short_open', 0.6, 0.75, step=0.01),
        "bsp1_pred_long_exit": trial.suggest_float('bsp1_pred_long_exit', 0.5, 0.6, step=0.01),
        "bsp1_pred_short_exit": trial.suggest_float('bsp1_pred_short_exit', 0.5, 0.6, step=0.01),
        "tp_long": trial.suggest_float('tp_long', 0.002, 0.03, step=0.001),
        "sl_long": trial.suggest_float('sl_long', 0.002, 0.005, step=0.001),
        "tp_short": trial.suggest_float('tp_short', 0.002, 0.03, step=0.001),
        "sl_short": trial.suggest_float('sl_short', 0.002, 0.005, step=0.001),
    })
    capital = trade_from(code, begin_time, end_time, dataset_param_grid, model, feature_names,
                         trade_param_grid)
    return capital


model_param_grid = {
    'seed': 42,
    'device': 'cpu',
    'objective': 'binary',
    'min_split_gain': 0,
    'min_child_weight': 1e-3,
    'boosting_type': 'gbdt',
    'verbose': -1
}


def model_from(param_grid, X_train, X_val, y_train, y_val):
    model = LGBMClassifier(**param_grid)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), eval_metric="auc")
    return model


def model_objective(trial, X_train, X_val, y_train, y_val):
    model_param_grid.update({
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.05),
        'reg_alpha': trial.suggest_int('reg_alpha', 0, 50),
        'reg_lambda': trial.suggest_int('reg_lambda', 0, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0, step=0.05),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0, step=0.05),

    })
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    model_param_grid["random_state"] = model_param_grid["seed"]
    model_param_grid["bagging_seed"] = model_param_grid["seed"]
    model_param_grid["feature_fraction_seed"] = model_param_grid["seed"]
    model_param_grid["gpu_device_id"] = 0
    model_param_grid["gpu_platform_id"] = 0
    # 计算每个类别的权重
    model_param_grid.update(
        {
            "class_weight": {
                0: class_weights[0],
                1: class_weights[1],
            }
        }
    )
    model = model_from(model_param_grid, X_train, X_val, y_train, y_val)
    y_pred = model.predict_proba(X_val)[:, 1]
    # 计算 F1 分数（适用于二分类）
    score = roc_auc_score(y_val, y_pred)
    # 返回平均 F1 分数
    return score


dataset_param_grid = {
    "trigger_step": True,  # 打开开关！
    "skip_step": 200,
    "divergence_rate": float("inf"),
    "bsp2_follow_1": False,
    "bsp3_follow_1": False,
    "min_zs_cnt": 0,
    "macd_algo": "peak",
    "bs_type": '1,1p',
    "cal_rsi": True,
    "cal_kdj": True,
    "cal_demark": False,
    "kl_data_check": False
}


def dataset_from(code, begin_time, end_time, param_grid):
    config = CChanConfig(conf=param_grid.copy())

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

        lv_chan = chan_snapshot[0]
        bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        if last_bsp.klu.idx not in bsp_dict and last_bsp.klu.klc.idx == lv_chan[-2].idx:
            bsp_dict[last_bsp.klu.idx] = {
                "feature": last_bsp.features,
            }
            factors = FeatureFactors(chan_snapshot[0],
                                     MAX_BI=param_grid["MAX_BI"],
                                     MAX_ZS=param_grid["MAX_ZS"],
                                     MAX_SEG=param_grid["MAX_SEG"],
                                     MAX_SEGSEG=param_grid["MAX_SEGSEG"],
                                     MAX_SEGZS=param_grid["MAX_SEGZS"]
                                     ).get_factors()
            for key in factors.keys():
                bsp_dict[last_bsp.klu.idx]['feature'].add_feat(key, float(factors[key]))

    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp(0)]
    rows = []
    for bsp_klu_idx, feature_info in bsp_dict.items():
        label = int(bsp_klu_idx in bsp_academy)
        feature = feature_info["feature"]
        row = {"label": label}
        for feature_name, value in feature.items():
            row.update({feature_name: value})
        rows.append(row)
    df = pd.DataFrame(rows)
    labels = df["label"].to_numpy(dtype=int)
    features = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    feature_names = df.columns[1:].tolist()

    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, shuffle=False)
    return X_train, X_val, y_train, y_val, feature_names


def dataset_objective(trial, code, begin_time, end_time):
    dataset_param_grid.update({
        "macd": {
            "fast": trial.suggest_int('fast', 3, 10),
            "slow": trial.suggest_int('slow', 11, 30),
            "signal": trial.suggest_int('signal', 5, 16),
        },
        "MAX_BI": trial.suggest_int('MAX_BI', 1, 12),
        "MAX_ZS": trial.suggest_int('MAX_ZS', 1, 3),
        "MAX_SEG": trial.suggest_int('MAX_SEG', 1, 3),
        "MAX_SEGSEG": trial.suggest_int('MAX_SEGSEG', 1, 3),
        "MAX_SEGZS": trial.suggest_int('MAX_SEGZS', 1, 3),
    })
    X_train, X_val, y_train, y_val, feature_names = dataset_from(code, begin_time, end_time, dataset_param_grid)
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
    study.optimize(lambda trial: model_objective(trial, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val),
                   n_trials=500, n_jobs=-1)
    model_param_grid.update(study.best_params)
    print(f"{code} dataset AUC:{study.best_value}")
    trial.set_user_attr("dataset_param_grid", dataset_param_grid)
    trial.set_user_attr("model_param_grid", model_param_grid)
    return study.best_value


if __name__ == "__main__":
    begin_time = "2020-01-01 00:00:00"
    end_time = "2022-01-01 00:00:00"
    val_begin_time = "2022-01-01 00:00:00"
    val_end_time = "2023-01-01 00:00:00"
    test_begin_time = "2023-01-01 00:00:00"
    test_end_time = "2024-01-01 00:00:00"
    reconnect_mt5()
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
    for i, code in enumerate(symbols):
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
        # 优化参数开始,评价指标为训练模型AUC指标最优时因子数据集为最佳
        study.optimize(lambda trial: dataset_objective(trial, code=code, begin_time=begin_time, end_time=end_time),
                       n_trials=100, n_jobs=-1,
                       timeout=60 * 60)
        # 优化参数结束
        best_auc = study.best_value
        best_dataset_param_grid = study.best_trial.user_attrs["dataset_param_grid"]
        best_model_param_grid = study.best_trial.user_attrs["model_param_grid"]
        print(f"{code} dataset最优AUC:{best_auc} model:{best_dataset_param_grid}")

        X_train, X_val, y_train, y_val, feature_names = dataset_from(code, begin_time, end_time,
                                                                     best_dataset_param_grid)
        print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
        model = model_from(best_model_param_grid, X_train, X_val, y_train, y_val)
        y_pred = model.predict_proba(X_val)[:, 1]
        # 计算 F1 分数（适用于二分类）
        score = roc_auc_score(y_val, y_pred)
        print(f"{code} 重新训练模型,AUC:{score}")
        # 优化参数开始,评价指标为交易模型赚钱最多时交易方法为最佳
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
        study.optimize(
            lambda trial: trade_objective(trial, code, val_begin_time, val_end_time, best_dataset_param_grid, model,
                                          feature_names),
            n_trials=100,
            n_jobs=-1)
        trade_param_grid.update(study.best_params)
        best_trade_param_grid = trade_param_grid.copy()
        history_profit = study.best_value
        print(f"{code} 最优交易历史盈利:{history_profit} trade:{trade_param_grid}")

        capital = trade_from(code, test_begin_time, test_end_time, best_dataset_param_grid, model,
                             feature_names, best_trade_param_grid)

        print(f"{code} 实战盈利:{capital}")
        print(f"chan:{best_dataset_param_grid},model:{best_model_param_grid},trade:{best_trade_param_grid}")
        with open("./eval.log", mode="a") as file:
            seq = [f"{code} dataset最优AUC:{best_auc}", f"{code} 最优交易历史盈利:{history_profit}",
                   f"{code} 实战盈利:{capital}",
                   f"dataset:{best_dataset_param_grid},model:{best_model_param_grid},trade:{best_trade_param_grid}"]
            file.writelines(seq)
