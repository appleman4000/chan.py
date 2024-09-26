# cython: language_level=3
# encoding:utf-8
import datetime
import logging
import math
import os.path
import pickle
import sys
import warnings
from typing import Dict

import lightgbm as lgb
import matplotlib
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import early_stopping
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, KL_TYPE, AUTYPE, BSP_TYPE
from FeatureLib import FeatureLib
from GenerateDataset import T_SAMPLE_INFO
from Plot.PlotDriver import CPlotDriver

logging.getLogger('LGBMClassifier').setLevel(logging.CRITICAL)
optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore")
sys.setrecursionlimit(100000)
local_time_format = '%Y-%m-%d %H:%M:%S'
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


def L0_draw_down(return_list):
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
    values = values.to_numpy(np.float32)
    return model.predict_proba(values)[0][1]


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
    long_klu_idx = 0
    short_klu_idx = 0
    treated_bsp_idx = set()
    for chan_snapshot in chan.step_load():

        lv0_chan = chan_snapshot[0]
        if len(lv0_chan.bi_list) < dataset_params["L0_BI"]:
            continue
        if len(lv0_chan.seg_list) < dataset_params["L0_SEG"]:
            continue
        if len(lv0_chan.segseg_list) < dataset_params["L0_SEGSEG"]:
            continue
        if len(lv0_chan.zs_list) < dataset_params["L0_ZS"]:
            continue
        if len(lv0_chan.segzs_list) < dataset_params["L0_SEGZS"]:
            continue

        lv1_chan = chan_snapshot[1]
        if len(lv1_chan.bi_list) < dataset_params["L1_BI"]:
            continue
        if len(lv1_chan.seg_list) < dataset_params["L1_SEG"]:
            continue
        if len(lv1_chan.segseg_list) < dataset_params["L1_SEGSEG"]:
            continue
        if len(lv1_chan.zs_list) < dataset_params["L1_ZS"]:
            continue
        if len(lv1_chan.segzs_list) < dataset_params["L1_SEGZS"]:
            continue

        # lv2_chan = chan[2]
        # if len(lv2_chan.bi_list) < dataset_params["L2_BI"]:
        #     continue
        # if len(lv2_chan.seg_list) < dataset_params["L2_SEG"]:
        #     continue
        # if len(lv2_chan.segseg_list) < dataset_params["L2_SEGSEG"]:
        #     continue
        # if len(lv2_chan.zs_list) < dataset_params["L2_ZS"]:
        #     continue
        # if len(lv2_chan.segzs_list) < dataset_params["L2_SEGZS"]:
        #     continue
        profit = 0
        bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        if long_order > 0:
            # 止盈
            close_price = round(lv0_chan[-1][-1].close, 5)
            long_profit = close_price / long_order - 1
            atr = bsp_dict[long_klu_idx]["atr"]
            bsp_price = bsp_dict[long_klu_idx]["bsp_price"]
            tp = close_price >= bsp_price + 6 * atr
            sl = close_price <= bsp_price - 2 * atr
            # 最大止盈止损保护
            # tp = long_profit >= trade_params["bsp1_sl_long"] * trade_params["risk_reward_ratio"]
            # sl = long_profit <= -trade_params["bsp1_sl_long"]
            exit_rule = last_bsp.klu.klc.idx == lv0_chan[-2].idx and (
                    BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type) and not last_bsp.is_buy
            if exit_rule or tp or sl:
                long_order = 0
                bsp_dict[long_klu_idx]["close_time"] = lv0_chan[-1][-1].time
                bsp_dict[long_klu_idx]["profit"] = round(long_profit * money, 2)
                profit += round(long_profit * money, 2)
        if short_order > 0:
            close_price = round(lv0_chan[-1][-1].close, 5)
            short_profit = short_order / close_price - 1
            atr = bsp_dict[short_klu_idx]["atr"]
            bsp_price = bsp_dict[short_klu_idx]["bsp_price"]
            tp = close_price <= bsp_price - 6 * atr
            sl = close_price >= bsp_price + 2 * atr
            # 最大止盈止损保护
            # tp = short_profit >= trade_params["bsp1_sl_short"] * trade_params["risk_reward_ratio"]
            # sl = short_profit <= -trade_params["bsp1_sl_short"]
            exit_rule = last_bsp.klu.klc.idx == lv0_chan[-2].idx and (
                    BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type) and last_bsp.is_buy
            if exit_rule or tp or sl:
                short_order = 0
                bsp_dict[short_klu_idx]["close_time"] = lv0_chan[-1][-1].time
                bsp_dict[short_klu_idx]["profit"] = round(short_profit * money, 2)
                profit += round(short_profit * money, 2)

        if long_order == 0 and short_order == 0:
            if last_bsp.klu.idx not in treated_bsp_idx and last_bsp.klu.klc.idx == lv0_chan[-2].idx and (
                    BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
                treated_bsp_idx.add(last_bsp.klu.idx)
                factors = FeatureLib(chan_snapshot,
                                     0.01 if 'JPY' in code else 0.0001,
                                     dataset_params["L0_BI"],
                                     dataset_params["L0_ZS"],
                                     dataset_params["L0_SEG"],
                                     dataset_params["L0_SEGSEG"],
                                     dataset_params["L0_SEGZS"],
                                     dataset_params["L1_BI"],
                                     dataset_params["L1_ZS"],
                                     dataset_params["L1_SEG"],
                                     dataset_params["L1_SEGSEG"],
                                     dataset_params["L1_SEGZS"],
                                     # dataset_params["L2_BI"],
                                     # dataset_params["L2_ZS"],
                                     # dataset_params["L2_SEG"],
                                     # dataset_params["L2_SEGSEG"],
                                     # dataset_params["L2_SEGZS"]
                                     ).get_factors()
                bsp_pred = predict_bsp(model=model, feature=factors, feature_names=feature_names)
                if bsp_pred >= trade_params["bsp1_open"] and last_bsp.is_buy:
                    bsp_dict[last_bsp.klu.idx] = {
                        "is_buy": last_bsp.is_buy,
                        "open_time": lv0_chan[-1][-1].time,
                        "price": lv0_chan[-1][-1].close,
                        "state": 0,
                        "profit": 0,
                        "atr": last_bsp.klu.atr,
                        "bsp_price": last_bsp.klu.close
                    }
                    long_klu_idx = last_bsp.klu.idx
                    long_order = round(lv0_chan[-1][-1].close * fee, 5)
                if bsp_pred >= trade_params["bsp1_open"] and not last_bsp.is_buy:
                    bsp_dict[last_bsp.klu.idx] = {
                        "is_buy": last_bsp.is_buy,
                        "open_time": lv0_chan[-1][-1].time,
                        "price": lv0_chan[-1][-1].close,
                        "state": 0,
                        "profit": 0,
                        "atr": last_bsp.klu.atr,
                        "bsp_price": last_bsp.klu.close
                    }
                    short_klu_idx = last_bsp.klu.idx
                    short_order = round(lv0_chan[-1][-1].close / fee, 5)
        capital += profit
        if profit != 0:
            trades.append(profit)
            print(f"{profit} {capital}")
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
        score2 = 1 - L0_draw_down(capitals) / 5000
        # 换算成等价盈亏比为1:1的胜率
        score3 = (win_rate * win_loss_radio - (1 - win_rate) + 1) / 2
        a, b, c = 2, 1, 1
        score = (a * score1 + b * score2 * c * score3) / (a + b + c)
        if math.isnan(score):
            score = 0
    return code, score, capital, plot_driver


def optimize_trade(trial, code, lv_list, begin_time, end_time, dataset_params, model, feature_names):
    trade_params = {
        "bsp1_open": trial.suggest_float('bsp1_open', 0.5, 0.8, step=0.01),
        "risk_reward_ratio": trial.suggest_float('risk_reward_ratio', 1.0, 3.0, step=0.1),
        # "bsp1_close": trial.suggest_float('bsp1_close', 0.0, bsp1_open, step=0.01),
        # "bsp1_tp_long": profit_loss,  # trial.suggest_float('bsp1_tp_long', 0.003, 0.02, step=0.001),
        "bsp1_sl_long": 0.004,  # trial.suggest_float('bsp1_sl_long', 0.002, 0.005, step=0.001),
        # "bsp1_tp_short": profit_loss,  # trial.suggest_float('bsp1_tp_short', 0.003, 0.02, step=0.001),
        "bsp1_sl_short": 0.004,  # trial.suggest_float('bsp1_sl_short', 0.002, 0.005, step=0.001),
    }
    code, score, capital, plot_driver = run_trade(code, lv_list, begin_time, end_time, dataset_params, model,
                                                  feature_names,
                                                  trade_params)
    print(f"{code} score: {score} {capital} {trade_params}")
    trial.set_user_attr("trade_params", trade_params)
    trial.set_user_attr("capital", capital)
    trial.set_user_attr("plot_driver", plot_driver)
    return score


def get_model(model_params, X_train, X_test, y_train, y_test):
    X_train, X_test, y_train, y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()

    X_train = X_train.to_numpy(np.float32)
    label_train = y_train["label"].to_numpy(int)
    X_test = X_test.to_numpy(np.float32)
    label_test = y_test["label"].to_numpy(int)
    model = lgb.LGBMClassifier(**model_params)
    X_train, label_train = shuffle(X_train, label_train, random_state=42)
    model.fit(X=X_train, y=label_train, eval_set=[(X_test, label_test)],
              callbacks=[early_stopping(stopping_rounds=10, verbose=False)])
    # 从 evals_result_ 中提取验证集的 AUC 值
    evals_result = model.evals_result_

    # 获取验证集的 AUC 列表
    auc_values = evals_result['valid_0']['auc']

    # 获取最佳迭代次数
    best_iteration = model.best_iteration_

    # 获取最佳 AUC 值
    score = auc_values[best_iteration - 1]
    return model, score


def optimize_model(trial, X_train, X_test, y_train, y_test, seed):
    X_train, X_test, y_train, y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()

    X_train = X_train.to_numpy(np.float32)
    label_train = y_train["label"].to_numpy(int)
    profit_train = y_train["profit"].to_numpy(float)
    X_test = X_test.to_numpy(np.float32)
    label_test = y_test["label"].to_numpy(int)
    profit_test = y_test["profit"].to_numpy(float)
    negative_samples = len(label_train[label_train == 0])
    positive_samples = len(label_train[label_train == 1])
    scale_pos_weight = negative_samples / positive_samples
    # sample_weight = []
    # min_profit = np.min(np.abs(profit_train[profit_train > 0]))
    # L0_profit = np.max(np.abs(profit_train[profit_train > 0]))
    # for i, label in enumerate(label_train):
    #     if label == 0:
    #         sample_weight.append(1.0)
    #     else:
    #         sample_weight.append(scale_pos_weight * ((profit_train[i] - min_profit) / (L0_profit - min_profit) + 0.5))
    # eval_sample_weight = []
    # for i, label in enumerate(label_test):
    #     if label == 0:
    #         eval_sample_weight.append(1.0)
    #     else:
    #         eval_sample_weight.append(
    #             scale_pos_weight * ((profit_test[i] - min_profit) / (L0_profit - min_profit) + 0.5))
    # class_weights = class_weight.compute_class_weight(
    #     "balanced", classes=np.unique(label_train), y=label_train
    # )
    model_params = {
        'device': 'gpu',
        'objective': 'binary',
        'metric': 'auc',  # average_precision
        'boosting_type': 'gbdt',
        'verbose': -1,
        'num_threads': 8,
        "random_state": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'num_leaves': trial.suggest_int('num_leaves', 15, 31),  # 调整范围
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, step=0.01),  # 限制学习率上限
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=10),  # 缩小范围
        'min_child_samples': trial.suggest_int('min_child_samples', 50, 100, step=10),  # 提高最小样本数
        'subsample': trial.suggest_float('subsample', 0.8, 1.0, step=0.01),  # 提高下采样比例
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0, step=0.01),  # 调整特征子样本
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),  # 缩小正则化范围
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),  # 缩小正则化范围
        'gpu_platform_id': 0,
        'gpu_device_id': trial.number % 2,
        'scale_pos_weight': scale_pos_weight,
    }
    model = lgb.LGBMClassifier(**model_params)

    # X_train, label_train = shuffle(X_train, label_train, random_state=42)
    model.fit(X=X_train, y=label_train, eval_set=[(X_test, label_test)],
              callbacks=[early_stopping(stopping_rounds=10, verbose=False)])
    # 从 evals_result_ 中提取验证集的 AUC 值
    evals_result = model.evals_result_

    # 获取验证集的 AUC 列表
    values = evals_result['valid_0']['auc']

    # 获取最佳迭代次数
    best_iteration = model.best_iteration_

    # 获取最佳 AUC 值
    score = values[best_iteration - 1]
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
    for chan_snapshot in chan.step_load():
        lv0_chan = chan_snapshot[0]
        if len(lv0_chan.bi_list) < params["L0_BI"]:
            continue
        if len(lv0_chan.seg_list) < params["L0_SEG"]:
            continue
        if len(lv0_chan.segseg_list) < params["L0_SEGSEG"]:
            continue
        if len(lv0_chan.zs_list) < params["L0_ZS"]:
            continue
        if len(lv0_chan.segzs_list) < params["L0_SEGZS"]:
            continue

        lv1_chan = chan_snapshot[1]
        if len(lv1_chan.bi_list) < params["L1_BI"]:
            continue
        if len(lv1_chan.seg_list) < params["L1_SEG"]:
            continue
        if len(lv1_chan.segseg_list) < params["L1_SEGSEG"]:
            continue
        if len(lv1_chan.zs_list) < params["L1_ZS"]:
            continue
        if len(lv1_chan.segzs_list) < params["L1_SEGZS"]:
            continue

        # lv2_chan = chan_snapshot[2]
        # if len(lv2_chan.bi_list) < params["L2_BI"]:
        #     continue
        # if len(lv2_chan.seg_list) < params["L2_SEG"]:
        #     continue
        # if len(lv2_chan.segseg_list) < params["L2_SEGSEG"]:
        #     continue
        # if len(lv2_chan.zs_list) < params["L2_ZS"]:
        #     continue
        # if len(lv2_chan.segzs_list) < params["L2_SEGZS"]:
        #     continue
        bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        if last_bsp.klu.idx not in bsp_dict and last_bsp.klu.klc.idx == lv0_chan[-2].idx and (
                BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type or
                BSP_TYPE.T2 in last_bsp.type or BSP_TYPE.T2S in last_bsp.type or
                BSP_TYPE.T3A in last_bsp.type or BSP_TYPE.T3B in last_bsp.type):
            factors = FeatureLib(chan,
                                 0.01 if 'JPY' in code else 0.0001,
                                 params["L0_BI"],
                                 params["L0_ZS"],
                                 params["L0_SEG"],
                                 params["L0_SEGSEG"],
                                 params["L0_SEGZS"],
                                 params["L1_BI"],
                                 params["L1_ZS"],
                                 params["L1_SEG"],
                                 params["L1_SEGSEG"],
                                 params["L1_SEGZS"],
                                 # params["L2_BI"],
                                 # params["L2_ZS"],
                                 # params["L2_SEG"],
                                 # params["L2_SEGSEG"],
                                 # params["L2_SEGZS"]
                                 ).get_factors()
            bsp_dict[last_bsp.klu.idx] = {
                "feature": factors,
                "price": lv0_chan[-1][-1].close,
                "is_buy": last_bsp.is_buy,
                "state": 0,
                "last_bsp": last_bsp,
                "open_time": lv0_chan[-1][-1].time
            }

    bsp_academy = np.array([bsp.klu.idx for bsp in chan.get_bsp(0)], dtype=int)
    rows = []
    for bsp_klu_idx, feature_info in bsp_dict.items():
        label = int(bsp_klu_idx in bsp_academy)
        profit = 0
        open_price = feature_info["price"]
        last_bsp = feature_info["last_bsp"]
        if label == 1:
            if last_bsp.bi.next:
                open_price = feature_info["price"]
                if last_bsp.is_buy:
                    profit = last_bsp.bi.next.get_end_val() / open_price - 1
                    assert profit > 0
                else:
                    profit = open_price / last_bsp.bi.next.get_end_val() - 1
                    assert profit > 0
        open_time = feature_info["open_time"].ts
        bsp_types = ",".join([t.value for t in last_bsp.type])
        row = {"code": code, "open_time": open_time, "label": label, "bsp_types": bsp_types, "profit": profit}
        feature = feature_info["feature"]
        row.update(feature)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def run_codes(codes):
    lv_list: list[KL_TYPE] = [KL_TYPE.K_1H, KL_TYPE.K_10M]
    begin_time = "2010-01-01 00:00:00"
    end_time = "2023-01-01 00:00:00"
    # val_begin_time = "2023-01-01 00:00:00"
    # val_end_time = "2024-01-01 00:00:00"
    test_begin_time = "2023-01-01 00:00:00"
    test_end_time = "2024-09-24 00:00:00"
    dataset_params = {
        "trigger_step": True,  # 打开开关！
        "bi_strict": True,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": True,
        "bsp3_follow_1": True,
        "min_zs_cnt": 0,
        "bs1_peak": True,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
        "cal_rsi": False,
        "cal_boll": False,
        "cal_kdj": False,
        "kl_data_check": False,
        "L0_BI": 5,
        "L0_ZS": 1,
        "L0_SEG": 2,
        "L0_SEGSEG": 2,
        "L0_SEGZS": 1,
        "L1_BI": 5,
        "L1_ZS": 1,
        "L1_SEG": 2,
        "L1_SEGSEG": 2,
        "L1_SEGZS": 1,
        # "L2_BI": 5,
        # "L2_ZS": 1,
        # "L2_SEG": 2,
        # "L2_SEGSEG": 2,
        # "L2_SEGZS": 1
    }

    if not os.path.exists("./result/all_codes.dat"):
        print("制作训练集")
        start = datetime.datetime.strptime(begin_time, local_time_format)
        dfs = []
        while start < datetime.datetime.strptime(end_time, local_time_format):
            df = Parallel(n_jobs=len(symbols), backend="multiprocessing")(
                delayed(get_dataset)(code, lv_list,
                                     start.strftime(local_time_format),
                                     min(datetime.datetime.strptime(end_time, local_time_format),
                                         (start + datetime.timedelta(days=150))).strftime(local_time_format),
                                     dataset_params) for code in codes)
            print(
                f"{min(datetime.datetime.strptime(end_time, local_time_format), (start + datetime.timedelta(days=150))).strftime(local_time_format)} completed")

            start = start + datetime.timedelta(days=120)
            dfs += df
        print("all tasks completed")
        merged_df = pd.concat(dfs)
        merged_df.drop_duplicates(subset=['code', 'open_time'], keep='last', inplace=True)
        merged_df.sort_values(by='open_time', inplace=True)

        with open("./result/all_codes.dat", "wb") as fid:
            pickle.dump(merged_df, fid)
    else:
        with open("./result/all_codes.dat", "rb") as fid:
            merged_df = pickle.load(fid)
    merged_df = merged_df[merged_df['bsp_types'].str.split(',').apply(lambda x: '1' in x or '1p' in x)]
    # # 删除重复的列
    merged_df = merged_df.T.drop_duplicates().T
    feature_names = merged_df.columns.tolist()
    feature_names.remove("label")
    feature_names.remove("profit")
    feature_names.remove("open_time")
    feature_names.remove("code")
    feature_names.remove("bsp_types")
    all_x_train, all_x_test, all_y_train, all_y_test = train_test_split(
        merged_df[feature_names],
        merged_df[["label", "profit"]],
        test_size=0.1,
        shuffle=False)
    negative_samples = len(all_y_train[all_y_train["label"] == 0])
    positive_samples = len(all_y_train[all_y_train["label"] == 1])
    print(
        f"Train Dataset:{all_x_train.shape} Test Dataset:{all_x_test.shape} {negative_samples / positive_samples}")

    print(f"找最优模型参数")
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
    study.optimize(
        lambda trial: optimize_model(trial, all_x_train, all_x_test, all_y_train, all_y_test, seed=42),
        n_trials=1000,
        n_jobs=8)
    model = study.best_trial.user_attrs["model"]
    model_params = study.best_trial.user_attrs["model_params"]
    auc = study.best_trial.value
    # model_params = {'device': 'gpu', 'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'verbose': -1, 'num_threads': 8,
    #  'random_state': 42, 'bagging_seed': 42, 'feature_fraction_seed': 42, 'max_depth': 6, 'num_leaves': 15,
    #  'learning_rate': 0.08, 'n_estimators': 710, 'min_child_samples': 50, 'subsample': 0.8700000000000001,
    #  'colsample_bytree': 0.9700000000000001, 'reg_alpha': 0.2677312287310178, 'reg_lambda': 4.02314360627926,
    #  'gpu_platform_id': 0, 'gpu_device_id': 0, 'scale_pos_weight': 4.848513011152416}
    # model,auc = get_model(model_params,all_x_train, all_x_test, all_y_train, all_y_test)
    print(f"AUC {auc} {model_params}")
    y_prob = np.array(model.predict_proba(all_x_test.to_numpy(np.float32))[:, 1])
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # 计算每个阈值下的正确率、召回率和F1分数
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        accuracy = accuracy_score(all_y_test['label'].astype(int), y_pred)
        precision = precision_score(all_y_test['label'].astype(int), y_pred)
        recall = recall_score(all_y_test['label'].astype(int), y_pred)
        f1 = f1_score(all_y_test['label'].astype(int), y_pred)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # 绘制曲线
    matplotlib.use('Agg')
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, accuracies, marker='o', label='Accuracy')
    plt.plot(thresholds, precisions, marker='o', label='Precision')
    plt.plot(thresholds, recalls, marker='x', label='Recall', linestyle='--')
    plt.plot(thresholds, f1_scores, marker='s', label='F1 Score', linestyle='-.')
    plt.ylabel('Score')
    plt.title('Threshold vs Accuracy, Precision, Recall, and F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./result/train.png")
    plt.clf()
    matplotlib.use('Agg')
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"best_threshold:{best_threshold}")
    for code in codes:
        # print(f"{code} 找最优交易参数")
        # storage = optuna.storages.InMemoryStorage()
        # study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(),
        #                             storage=storage)
        # study.optimize(
        #     lambda trial: optimize_trade(trial, code, lv_list, val_begin_time, val_end_time, dataset_params, model,
        #                                  feature_names),
        #     n_trials=50,
        #     n_jobs=8)
        # trade_params = study.best_trial.user_attrs["trade_params"]
        # capital = study.best_trial.user_attrs["capital"]
        # plot_driver = study.best_trial.user_attrs["plot_driver"]
        # plot_driver.save2img(f"./result/{code}_train.png")
        # plt.clf()
        # print(f"{code} Best Score {study.best_trial.value} Best capital {capital} trade {trade_params}")
        print("开始测试")
        trade_params = {
            "bsp1_open": 0.4,
            "risk_reward_ratio": 2.5,
            "bsp1_sl_long": 0.005,  # trial.suggest_float('bsp1_sl_long', 0.002, 0.005, step=0.001),
            "bsp1_sl_short": 0.005,  # trial.suggest_float('bsp1_sl_short', 0.002, 0.005, step=0.001),
        }
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
