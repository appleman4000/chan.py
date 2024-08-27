# cython: language_level=3
# encoding:utf-8
import logging
import warnings
from typing import Dict

import joblib
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
from FeatureEngineering import FeatureFactors
from GenerateDataset import T_SAMPLE_INFO

logging.getLogger('LGBMClassifier').setLevel(logging.CRITICAL)
optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore")


def predict_bsp(model, last_bsp: CBS_Point, feature_names):
    empty = pd.DataFrame(data=[], columns=feature_names)
    features = [dict(last_bsp.features.items())]
    features = pd.DataFrame(features)
    features = pd.concat([empty, features]).to_numpy()
    return model.predict_proba(features)[0][1]


lv_list = [KL_TYPE.K_10M]
data_src = DATA_SRC.FOREX
trade_params = {
}


def run_trade(code, begin_time, end_time, dataset_params, model, feature_names, trade_params):
    config = CChanConfig(conf=dataset_params.copy())
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
        if last_bsp.klu.klc.idx == lv_chan[-1].idx and (
                BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
            factors = FeatureFactors(chan[0], MAX_BI=dataset_params["MAX_BI"],
                                     MAX_ZS=dataset_params["MAX_ZS"],
                                     MAX_SEG=dataset_params["MAX_SEG"],
                                     MAX_SEGSEG=dataset_params["MAX_SEGSEG"],
                                     MAX_SEGZS=dataset_params["MAX_SEGZS"]).get_factors()
            for key in factors.keys():
                last_bsp.features.add_feat(key, float(factors[key]))
            bsp1_pred = predict_bsp(model=model, last_bsp=last_bsp, feature_names=feature_names)
        else:
            bsp1_pred = 0.0
        if long_order > 0:
            # 止盈
            close_price = round(lv_chan[-1][-1].close, 5)
            long_profit = close_price / long_order - 1
            exit_rule = bsp1_pred > trade_params["bsp1_pred_long_exit"] and not last_bsp.is_buy
            # 最大止盈止损保护
            tp = long_profit > trade_params["tp_long"]
            sl = long_profit < -trade_params["sl_long"]
            if tp or sl or exit_rule:
                long_order = 0
                profit += round(long_profit * money, 2)
        if short_order > 0:
            close_price = round(lv_chan[-1][-1].close, 5)
            short_profit = short_order / close_price - 1
            exit_rule = bsp1_pred > trade_params["bsp1_pred_short_exit"] and last_bsp.is_buy
            # 最大止盈止损保护
            tp = short_profit > trade_params["tp_short"]
            sl = short_profit < -trade_params["sl_short"]
            if tp or sl or exit_rule:
                short_order = 0
                profit += round(short_profit * money, 2)

        if long_order == 0 and short_order == 0:
            if bsp1_pred > trade_params["bsp1_pred_long_open"] and last_bsp.is_buy:
                long_order = round(lv_chan[-1][-1].close * fee, 5)
            if bsp1_pred > trade_params["bsp1_pred_short_open"] and not last_bsp.is_buy:
                short_order = round(lv_chan[-1][-1].close / fee, 5)

        capital += profit
    return capital


def optimize_trade(trial, code, begin_time, end_time, dataset_params, model, feature_names):
    trade_params.update({
        "bsp1_pred_long_open": trial.suggest_float('bsp1_pred_long_open', 0.6, 0.75, step=0.02),
        "bsp1_pred_short_open": trial.suggest_float('bsp1_pred_short_open', 0.6, 0.75, step=0.02),
        "bsp1_pred_long_exit": trial.suggest_float('bsp1_pred_long_exit', 0.5, 0.6, step=0.02),
        "bsp1_pred_short_exit": trial.suggest_float('bsp1_pred_short_exit', 0.5, 0.6, step=0.02),
        "tp_long": trial.suggest_float('tp_long', 0.001, 0.03, step=0.001),
        "sl_long": trial.suggest_float('sl_long', 0.001, 0.005, step=0.001),
        "tp_short": trial.suggest_float('tp_short', 0.001, 0.03, step=0.001),
        "sl_short": trial.suggest_float('sl_short', 0.001, 0.005, step=0.001),
    })
    score = run_trade(code, begin_time, end_time, dataset_params, model, feature_names, trade_params)
    return score


model_params = {
    'seed': 42,
    'device': 'cpu',
    'objective': 'binary',
    'min_split_gain': 0,
    'min_child_weight': 1e-3,
    'boosting_type': 'gbdt',
    'verbose': -1,
    'num_threads': -1
}


def get_model(params, X_train, X_val, y_train, y_val):
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), eval_metric="auc")
    return model


def optimize_model(trial, X_train, X_val, y_train, y_val):
    model_params.update({
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
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
    model_params["random_state"] = model_params["seed"]
    model_params["bagging_seed"] = model_params["seed"]
    model_params["feature_fraction_seed"] = model_params["seed"]
    model_params["gpu_device_id"] = 0
    model_params["gpu_platform_id"] = 0
    # 计算每个类别的权重
    model_params.update(
        {
            "class_weight": {
                0: class_weights[0],
                1: class_weights[1],
            }
        }
    )
    model = get_model(model_params, X_train, X_val, y_train, y_val)
    y_pred = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_pred)
    return score


def get_dataset(code, begin_time, end_time, params):
    config = CChanConfig(conf=params.copy())

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

        if last_bsp.klu.idx not in bsp_dict and last_bsp.klu.klc.idx == lv_chan[-1].idx:
            bsp_dict[last_bsp.klu.idx] = {
                "feature": last_bsp.features,
            }
            factors = FeatureFactors(chan_snapshot[0],
                                     MAX_BI=params["MAX_BI"],
                                     MAX_ZS=params["MAX_ZS"],
                                     MAX_SEG=params["MAX_SEG"],
                                     MAX_SEGSEG=params["MAX_SEGSEG"],
                                     MAX_SEGZS=params["MAX_SEGZS"]
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


dataset_params = {
    "trigger_step": True,  # 打开开关！
    "skip_step": 500,
    "divergence_rate": 0.9,
    "bsp2_follow_1": False,
    "bsp3_follow_1": False,
    "min_zs_cnt": 0,
    "macd_algo": "peak",
    "bs_type": '1,1p',
    "cal_rsi": True,
    "cal_kdj": True,
    "cal_demark": False,
    "kl_data_check": False,
    "MAX_BI": 7,
    "MAX_ZS": 2,
    "MAX_SEG": 2,
    "MAX_SEGSEG": 2,
    "MAX_SEGZS": 2,
}


def optimize_dataset(trial, code, begin_time, end_time):
    dataset_params.update({
        "divergence_rate": trial.suggest_float('divergence_rate', 0.7, 1.0, step=0.05),
        "macd_algo": trial.suggest_categorical('macd_algo', ["area", "peak", "full_area", "diff", "slope", "amp"]),
    })
    X_train, X_val, y_train, y_val, feature_names = get_dataset(code, begin_time, end_time, dataset_params)
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
    study.optimize(lambda t: optimize_model(t, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val),
                   n_trials=200, n_jobs=-1)
    print(f"{code} dataset AUC:{study.best_value}")
    return study.best_value


def run_code(code):
    print(f"{code} started")
    begin_time = "2021-01-01 00:00:00"
    end_time = "2022-01-01 00:00:00"
    val_begin_time = "2022-01-01 00:00:00"
    val_end_time = "2023-01-01 00:00:00"
    test_begin_time = "2023-01-01 00:00:00"
    test_end_time = "2024-01-01 00:00:00"
    print(f"{code} 找最优的缠论参数")
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
    study.optimize(lambda trial: optimize_dataset(trial, code, begin_time, end_time), n_trials=15, n_jobs=-1)
    dataset_params.update(study.best_params)
    print(f"{code} {study.best_params}")
    X_train, X_val, y_train, y_val, feature_names = get_dataset(code, begin_time, end_time, dataset_params)
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    print(f"{code} Training data: {X_train.shape}, Validation data: {X_val.shape} class_weights:{class_weights}")
    print(f"{code} 找最优测模型参数")
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
    study.optimize(lambda trial: optimize_model(trial, X_train, X_val, y_train, y_val), n_trials=500, n_jobs=-1)
    model_params.update(study.best_params)
    print(f"{code} {study.best_params}")
    print(f"{code} 训练模型")
    model = get_model(model_params, X_train, X_val, y_train, y_val)
    y_pred = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred)
    print(f"{code} 模型 AUC:{auc_score}")
    print(f"{code} 找最优的交易参数")
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
    study.optimize(
        lambda trial: optimize_trade(trial, code, val_begin_time, val_end_time, dataset_params, model, feature_names),
        n_trials=20,
        n_jobs=-1)
    trade_params.update(study.best_params)
    print(f"{code} {study.best_params}")
    best_value = study.best_value
    print(f"{code} 优化后交易历史盈利:{best_value}")

    capital = run_trade(code, test_begin_time, test_end_time, dataset_params, model, feature_names, trade_params)

    print(f"{code} 实战盈利:{capital}")
    print(f"chan:{dataset_params},model:{model_params},trade:{trade_params}")
    with open("./report.log", mode="a") as file:
        seq = [f"{code} dataset AUC:{auc_score}", f"{code} 优化后交易历史盈利:{best_value}",
               f"{code} 实战盈利:{capital}",
               f"dataset:{dataset_params},model:{model_params},trade:{trade_params}"]
        file.writelines(seq)


def main():
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
    joblib.Parallel(8, backend="multiprocessing")(
        joblib.delayed(run_code)(symbols[i])
        for i in range(len(symbols))
    )


if __name__ == "__main__":
    main()
