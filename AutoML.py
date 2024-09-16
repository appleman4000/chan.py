# cython: language_level=3
# encoding:utf-8
import logging
import math
import os.path
import pickle
import sys
import warnings
from typing import Dict

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import early_stopping
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, KL_TYPE, AUTYPE, BSP_TYPE, FX_TYPE
from FeatureEngineering import FeatureFactors
from GenerateDataset import T_SAMPLE_INFO
from Plot.PlotDriver import CPlotDriver

logging.getLogger('LGBMClassifier').setLevel(logging.CRITICAL)
optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore")
sys.setrecursionlimit(100000)

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


def max_draw_down(return_list):
    if len(return_list) == 0:
        return 0
    '''最大回撤资金'''
    i = np.argmax((np.maximum.accumulate(return_list) - return_list))
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])  # 开始位置
    return round(return_list[j] - return_list[i])


def plot(chan, plot_marker):
    plot_config = {
        "plot_kline": False,
        "plot_kline_combine": False,
        "plot_bi": True,
        "plot_seg": False,
        "plot_zs": False,
        "plot_bsp": False,
        "plot_marker": True,
    }
    plot_para = {
        "figure": {
            "w": 400,
            "h": 10,
            "x_range": 20000,
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
    return plot_driver


def predict_bsp(model, feature: dict, feature_names):
    values = pd.DataFrame([feature], columns=feature_names)
    return model.predict_proba(values.to_numpy(np.float32))[:, 0]


def run_trade(code, lv_list, begin_time, end_time, dataset_params, model, feature_names, trade_params):
    config = CChanConfig(conf=dataset_params.copy())
    chan = CChan(
        code=code,
        data_src=DATA_SRC.FOREX,
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
    capitals = []
    trades = []
    plot_marker = {}
    bsp_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的bsp的特征
    factors0 = None
    factors1 = None
    bsp1_pred = 0
    long_klu_idx = 0
    short_klu_idx = 0
    for chan_snapshot in chan.step_load():

        lv_chan = chan_snapshot[0]
        profit = 0
        bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        if last_bsp.klu.klc.idx == lv_chan[-1].idx and (
                BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
            factors0 = FeatureFactors(chan_snapshot[0], MAX_BI=dataset_params["MAX_BI"],
                                      MAX_ZS=dataset_params["MAX_ZS"],
                                      MAX_SEG=dataset_params["MAX_SEG"],
                                      MAX_SEGSEG=dataset_params["MAX_SEGSEG"],
                                      MAX_SEGZS=dataset_params["MAX_SEGZS"]).get_factors()
            factors1 = FeatureFactors(chan_snapshot[1], MAX_BI=dataset_params["MAX_BI"],
                                      MAX_ZS=dataset_params["MAX_ZS"],
                                      MAX_SEG=dataset_params["MAX_SEG"],
                                      MAX_SEGSEG=dataset_params["MAX_SEGSEG"],
                                      MAX_SEGZS=dataset_params["MAX_SEGZS"]).get_factors()
        if last_bsp.klu.klc.idx == lv_chan[-2].idx and (
                BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type) and lv_chan[-2].fx != FX_TYPE.UNKNOWN:

            if factors0 is not None and factors1 is not None:
                features = dict()
                for key, value in factors0.items():
                    features.update({f"{key}_0": value})
                for key, value in factors1.items():
                    features.update({f"{key}_1": value})
                bsp1_pred = predict_bsp(model=model, feature=features, feature_names=feature_names)
        else:
            bsp1_pred = 0
        if long_order > 0:
            # 止盈
            close_price = round(lv_chan[-1][-1].close, 5)
            long_profit = close_price / long_order - 1
            # 最大止盈止损保护
            tp = long_profit >= trade_params["bsp1_sl_long"] * trade_params["risk_reward_ratio"]
            sl = long_profit <= -trade_params["bsp1_sl_long"]
            if tp or sl:
                long_order = 0
                bsp_dict[long_klu_idx]["close_time"] = lv_chan[-1][-1].time
                bsp_dict[long_klu_idx]["profit"] = round(long_profit * money, 2)
                profit += round(long_profit * money, 2)
        if short_order > 0:
            close_price = round(lv_chan[-1][-1].close, 5)
            short_profit = short_order / close_price - 1
            # 最大止盈止损保护
            tp = short_profit >= trade_params["bsp1_sl_short"] * trade_params["risk_reward_ratio"]
            sl = short_profit <= -trade_params["bsp1_sl_short"]
            if tp or sl:
                short_order = 0
                bsp_dict[short_klu_idx]["close_time"] = lv_chan[-1][-1].time
                bsp_dict[short_klu_idx]["profit"] = round(short_profit * money, 2)
                profit += round(short_profit * money, 2)

        if long_order == 0 and short_order == 0:
            if bsp1_pred >= trade_params["bsp1_open"] and last_bsp.is_buy and \
                    last_bsp.klu.klc.idx == lv_chan[-2].idx and \
                    (BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type) and \
                    lv_chan[-2].fx == FX_TYPE.BOTTOM:
                bsp_dict[last_bsp.klu.idx] = {
                    "is_buy": last_bsp.is_buy,
                    "open_time": lv_chan[-1][-1].time,
                    "price": lv_chan[-1][-1].close,
                    "state": 0,
                    "profit": 0
                }

                long_klu_idx = last_bsp.klu.idx
                long_order = round(lv_chan[-1][-1].close * fee, 5)
            if bsp1_pred >= trade_params["bsp1_open"] and not last_bsp.is_buy and \
                    last_bsp.klu.klc.idx == lv_chan[-2].idx and \
                    (BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type) and \
                    lv_chan[-2].fx == FX_TYPE.TOP:
                bsp_dict[last_bsp.klu.idx] = {
                    "is_buy": last_bsp.is_buy,
                    "open_time": lv_chan[-1][-1].time,
                    "price": lv_chan[-1][-1].close,
                    "state": 0,
                    "profit": 0
                }
                short_klu_idx = last_bsp.klu.idx
                short_order = round(lv_chan[-1][-1].close / fee, 5)
        capital += profit
        if profit != 0:
            trades.append(profit)
            # print(f"{profit} {capital}")
        capitals.append(capital)
    for bsp_klc_idx, feature_info in bsp_dict.items():
        bs = "BUY" if feature_info["is_buy"] else "SELL"
        error = "√" if feature_info["profit"] > 0 else "×"
        label = bs + error

        plot_marker[feature_info["open_time"].to_str()] = (
            label, "down" if feature_info["is_buy"] else "up", "red" if feature_info["is_buy"] else "green")
        if "close_time" in feature_info.keys():
            plot_marker[feature_info["close_time"].to_str()] = (
                "CLOSE", "up" if feature_info["is_buy"] else "down")
    plot_driver = plot(chan, plot_marker)
    trades = np.asarray(trades)

    if len(trades) == 0:
        score = 0
    else:
        # 胜率
        win_rate = len(trades[trades > 0]) / (len(trades) + 1e-7)
        win = np.mean(trades[trades > 0])
        loss = -np.mean(trades[trades <= 0])
        # 盈亏比
        win_loss_radio = win / (loss + 1e-7)
        score1 = (capitals[-1] - capitals[0]) / 10000
        score2 = 1 - max_draw_down(capitals) / 5000
        # 换算成等价盈亏比为1:1的胜率
        score3 = (win_rate * win_loss_radio - (1 - win_rate) + 1) / 2
        a, b, c = 2, 1, 1
        score = (a * score1 + b * score2 * c * score3) / (a + b + c)
        if math.isnan(score):
            score = 0
    return code, score, capital, plot_driver


def optimize_trade(trial, code, lv_list, begin_time, end_time, dataset_params, model, feature_names):
    trade_params = {
        "bsp1_open": trial.suggest_float('bsp1_open', 0.1, 0.5, step=0.01),
        "risk_reward_ratio": trial.suggest_float('risk_reward_ratio', 1.0, 4.0, step=0.1),
        # "bsp1_close": trial.suggest_float('bsp1_close', 0.0, bsp1_open, step=0.01),
        # "bsp1_tp_long": profit_loss,  # trial.suggest_float('bsp1_tp_long', 0.003, 0.02, step=0.001),
        "bsp1_sl_long": trial.suggest_float('bsp1_sl_long', 0.002, 0.005, step=0.001),
        # "bsp1_tp_short": profit_loss,  # trial.suggest_float('bsp1_tp_short', 0.003, 0.02, step=0.001),
        "bsp1_sl_short": trial.suggest_float('bsp1_sl_short', 0.002, 0.005, step=0.001),
    }
    code, score, capital, plot_driver = run_trade(code, lv_list, begin_time, end_time, dataset_params, model,
                                                  feature_names,
                                                  trade_params)
    print(f"{code} score: {score} {capital} {trade_params}")
    trial.set_user_attr("trade_params", trade_params)
    trial.set_user_attr("capital", capital)
    trial.set_user_attr("plot_driver", plot_driver)
    return score


def optimize_model(trial, X_train, X_test, y_train, y_test, seed):
    X_train, X_test, y_train, y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()

    X_train = X_train.to_numpy(np.float32)
    label_train = y_train["label"].to_numpy(int)
    X_test = X_test.to_numpy(np.float32)
    label_test = y_test["label"].to_numpy(int)
    negative_samples = len(label_train[label_train == 0])
    positive_samples = len(label_train[label_train == 1])
    model_params = {
        'device': 'gpu',
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'num_threads': 1,
        "scale_pos_weight": negative_samples / positive_samples,
        "random_state": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'num_leaves': trial.suggest_int('num_leaves', 7, 31),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.01, 1.0, step=0.01),
        'min_child_weight': trial.suggest_float('min_child_weight', 0, 10, step=0.01),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0, step=0.01),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0, step=0.01),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0, step=0.01),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0, step=0.01),
        'gpu_platform_id': 0,  # 可选，通常为 0
        'gpu_device_id': trial.number % 2  # 可选，通常为 0

    }
    model = lgb.LGBMClassifier(**model_params)
    X_train, label_train = shuffle(X_train, label_train, random_state=42)
    model.fit(X=X_train, y=label_train, eval_set=[(X_test, label_test)],
              callbacks=[early_stopping(stopping_rounds=100, verbose=False)])
    # 从 evals_result_ 中提取验证集的 AUC 值
    evals_result = model.evals_result_

    # 获取验证集的 AUC 列表
    auc_values = evals_result['valid_0']['auc']

    # 获取最佳迭代次数
    best_iteration = model.best_iteration_

    # 获取最佳 AUC 值
    score = auc_values[best_iteration - 1]
    trial.set_user_attr("model", model)
    trial.set_user_attr("model_params", model_params)

    return score


def get_dataset(code, lv_list, begin_time, end_time, params):
    print(f"{code} 开始产生数据集")
    config = CChanConfig(conf=params.copy())

    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=DATA_SRC.FOREX,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.NONE,
    )

    bsp_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的bsp的特征
    # 跑策略，保存买卖点的特征
    factors0 = None
    factors1 = None
    for chan_snapshot in chan.step_load():

        lv_chan = chan_snapshot[0]
        bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]
        if last_bsp.klu.klc.idx == lv_chan[-1].idx and (BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
            factors0 = FeatureFactors(chan_snapshot[0],
                                      MAX_BI=params["MAX_BI"],
                                      MAX_ZS=params["MAX_ZS"],
                                      MAX_SEG=params["MAX_SEG"],
                                      MAX_SEGSEG=params["MAX_SEGSEG"],
                                      MAX_SEGZS=params["MAX_SEGZS"]
                                      ).get_factors()
            factors1 = FeatureFactors(chan_snapshot[1],
                                      MAX_BI=params["MAX_BI"],
                                      MAX_ZS=params["MAX_ZS"],
                                      MAX_SEG=params["MAX_SEG"],
                                      MAX_SEGSEG=params["MAX_SEGSEG"],
                                      MAX_SEGZS=params["MAX_SEGZS"]
                                      ).get_factors()
        if last_bsp.klu.idx not in bsp_dict and last_bsp.klu.klc.idx == lv_chan[-2].idx and (
                BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type) and lv_chan[-2].fx != FX_TYPE.UNKNOWN:
            if factors0 is None or factors1 is None:
                continue
            features = dict()
            for key, value in factors0.items():
                features.update({f"{key}_0": value})
            for key, value in factors1.items():
                features.update({f"{key}_1": value})
            bsp_dict[last_bsp.klu.idx] = {
                "feature": features,
                "price": lv_chan[-1][-1].close,
                "is_buy": last_bsp.is_buy,
                "state": 0,
                "last_bsp": last_bsp
            }

    bsp_academy = np.array([bsp.klu.idx for bsp in chan.get_bsp(0)], dtype=int)
    rows = []
    for bsp_klu_idx, feature_info in bsp_dict.items():
        label = int(bsp_klu_idx in bsp_academy)
        row = {"label": label}
        feature = feature_info["feature"]
        row.update(feature)
        rows.append(row)
    df = pd.DataFrame(rows)
    return code, df


def run_codes(codes):
    lv_list = [KL_TYPE.K_30M, KL_TYPE.K_5M]
    begin_time = "2018-01-01 00:00:00"
    end_time = "2023-01-01 00:00:00"
    val_begin_time = "2023-01-01 00:00:00"
    val_end_time = "2024-01-01 00:00:00"
    test_begin_time = "2024-01-01 00:00:00"
    test_end_time = "2024-09-10 00:00:00"
    dataset_params = {
        "trigger_step": True,  # 打开开关！
        "bi_strict": True,
        "skip_step": 200,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": True,
        "macd_algo": "peak",
        # "bs_type": '1,2,3a,1p,2s,3b',
        "bs_type": '1,1p',
        "print_warning": True,
        "zs_algo": "normal",
        "cal_rsi": False,
        "cal_boll": False,
        "cal_kdj": False,
        "kl_data_check": False,
        "MAX_BI": 7,
        "MAX_ZS": 1,
        "MAX_SEG": 1,
        "MAX_SEGSEG": 1,
        "MAX_SEGZS": 1,
    }
    if not os.path.exists("./result/all_codes.dat"):
        print("制作训练集")
        results = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(get_dataset)(code, lv_list, begin_time, end_time, dataset_params) for code in codes)
        with open("./result/all_codes.dat", "wb") as fid:
            pickle.dump(results, fid)
    else:
        with open("./result/all_codes.dat", "rb") as fid:
            results = pickle.load(fid)

    X_train = dict()
    y_train = dict()
    X_test = dict()
    y_test = dict()
    feature_names = dict()
    all_feature_names = set()
    for code, dataset in results:
        all_feature_names.update(set(dataset.columns))
    all_feature_names.remove("label")
    all_x_train = pd.DataFrame(columns=list(all_feature_names))
    all_y_train = pd.DataFrame()
    all_x_test = pd.DataFrame(columns=list(all_feature_names))
    all_y_test = pd.DataFrame()
    for code, dataset in results:
        dataset = pd.DataFrame(dataset, columns=["label"] + list(all_feature_names))
        X_train[code], X_test[code], y_train[code], y_test[code] = train_test_split(dataset[list(all_feature_names)],
                                                                                    dataset[["label"]],
                                                                                    test_size=0.2,
                                                                                    shuffle=False)
        all_x_train = pd.concat([all_x_train, X_train[code]])
        all_y_train = pd.concat([all_y_train, y_train[code]])
        all_x_test = pd.concat([all_x_test, X_test[code]])
        all_y_test = pd.concat([all_y_test, y_test[code]])
    feature_names = list(all_feature_names)
    for code in codes:
        negative_samples = len(y_train[code][y_train[code]["label"] == 0])
        positive_samples = len(y_train[code][y_train[code]["label"] == 1])
        print(
            f"Train Dataset:{X_train[code].shape} Test Dataset:{X_test[code].shape} {negative_samples / positive_samples}")
        print(f"找最优模型参数")
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
        study.optimize(
            lambda trial: optimize_model(trial, X_train[code], X_test[code], y_train[code], y_test[code], seed=42),
            n_trials=1000,
            n_jobs=8)
        model = study.best_trial.user_attrs["model"]
        model_params = study.best_trial.user_attrs["model_params"]
        auc = study.best_trial.value
        print(f"AUC {auc} {model_params}")
        print(f"{code} 找最优交易参数")
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(),
                                    storage=storage)
        study.optimize(
            lambda trial: optimize_trade(trial, code, lv_list, val_begin_time, val_end_time, dataset_params, model,
                                         feature_names),
            n_trials=50,
            n_jobs=4)
        trade_params = study.best_trial.user_attrs["trade_params"]
        capital = study.best_trial.user_attrs["capital"]
        plot_driver = study.best_trial.user_attrs["plot_driver"]
        plot_driver.save2img(f"./result/{code}_train.png")
        plt.clf()
        print(f"{code} Best Score {study.best_trial.value} Best capital {capital} trade {trade_params}")
        print("开始测试")
        code, score, capital, plot_driver = run_trade(code, lv_list, test_begin_time, test_end_time, dataset_params,
                                                      model,
                                                      feature_names,
                                                      trade_params)
        plot_driver.save2img(f"./result/{code}_test.png")
        plt.clf()
        message = f"{code} 2024年实战盈利:{score} {capital} ,dataset:{dataset_params},trade:{trade_params}\n"
        print(message)
        with open("./result/report.log", mode="a") as file:
            file.writelines(message)


def main():
    run_codes(symbols)


if __name__ == "__main__":
    main()
