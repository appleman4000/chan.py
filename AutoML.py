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
import matplotlib
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import early_stopping
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
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


def max_draw_down(return_list):
    if len(return_list) == 0:
        return 0
    '''最大回撤资金'''
    i = np.argmax((np.maximum.accumulate(return_list) - return_list))
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])  # 开始位置
    return round(return_list[j] - return_list[i])


def plot(chan, plot_marker, name):
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
    values = pd.DataFrame([feature], columns=feature_names).to_numpy(np.float32)
    return model.predict_proba(values)[0, 1]


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
    factors = None
    bsp1_pred = 0
    for chan_snapshot in chan.step_load():

        lv_chan = chan_snapshot[0]
        profit = 0
        bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        if last_bsp.klu.klc.idx == lv_chan[-1].idx and (
                BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
            factors = FeatureFactors(chan_snapshot[0], MAX_BI=dataset_params["MAX_BI"],
                                     MAX_ZS=dataset_params["MAX_ZS"],
                                     MAX_SEG=dataset_params["MAX_SEG"],
                                     MAX_SEGSEG=dataset_params["MAX_SEGSEG"],
                                     MAX_SEGZS=dataset_params["MAX_SEGZS"]).get_factors()
        if last_bsp.klu.klc.idx == lv_chan[-2].idx and (
                BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type) and lv_chan[-2].fx != FX_TYPE.UNKNOWN:
            if factors is None:
                continue
            last_bsp.add_feat(factors)
            features = dict(last_bsp.features.items())
            bsp1_pred = predict_bsp(model=model, feature=features, feature_names=feature_names)
        else:
            bsp1_pred = 0
        if long_order > 0:
            # 止盈
            close_price = round(lv_chan[-1][-1].close, 5)
            long_profit = close_price / long_order - 1
            # cross = lv_chan[-1][-1].cross[f"Cross{trade_params['cross_period']}"]
            # exit_rule = lv_chan[-1][-1].idx - long_klu_idx >= 7 and lv_chan[-1].fx == FX_TYPE.TOP
            # exit_rule = lv_chan[-1][-1].rsi >= 0.7
            # 最大止盈止损保护
            tp = long_profit >= trade_params["bsp1_tp_long"]
            sl = long_profit <= -trade_params["bsp1_tp_long"] / trade_params["long_profit_loss_radio"]
            if tp or sl:
                long_order = 0
                bsp_dict[long_klu_idx]["close_time"] = lv_chan[-1][-1].time
                bsp_dict[long_klu_idx]["profit"] = round(long_profit * money, 2)
                profit += round(long_profit * money, 2)
        if short_order > 0:
            close_price = round(lv_chan[-1][-1].close, 5)
            short_profit = short_order / close_price - 1
            # exit_rule = lv_chan[-2][-1].indicators["EMA26"] < 0 <= lv_chan[-1][-1].indicators["EMA26"]
            # exit_rule = bsp1_pred > trade_params["bsp1_close"] and last_bsp.is_buy
            # cross = lv_chan[-1][-1].cross[f"Cross{trade_params['cross_period']}"]
            # exit_rule = cross == -1
            # exit_rule = lv_chan[-1][-1].idx - short_klu_idx >= 7 and lv_chan[-1].fx == FX_TYPE.BOTTOM
            # exit_rule = lv_chan[-1][-1].rsi <= 0.3
            # 最大止盈止损保护
            tp = short_profit >= trade_params["bsp1_tp_short"]
            sl = short_profit <= -trade_params["bsp1_tp_short"] / trade_params["short_profit_loss_radio"]
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
        label = feature_info["profit"] > 0
        bs = "b1" if feature_info["is_buy"] else "s1"
        error = "√" if label else "×"
        label = bs + error

        plot_marker[feature_info["open_time"].to_str()] = (
            label, "down" if feature_info["is_buy"] else "up")
    plot_driver = plot(chan, plot_marker, name=f"{code}")
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
        a, b, c = 4, 1, 1
        score = (a * score1 + b * score2 * c * score3) / (a + b + c)
        if math.isnan(score):
            score = 0
    print(f"{code} 盈利:{score} {capital}")
    return code, score, capital, plot_driver


def optimize_trade(trial, code, lv_list, begin_time, end_time, dataset_params, model, feature_names, profit_loss):
    trade_params = {
        "bsp1_open": trial.suggest_float('bsp1_open', 0.1, 0.25, step=0.01),
        "long_profit_loss_radio": trial.suggest_float('long_profit_loss_radio', 1.0, 3.0, step=0.1),
        "short_profit_loss_radio": trial.suggest_float('short_profit_loss_radio', 1.0, 3.0, step=0.1),
        # "bsp1_close": trial.suggest_float('bsp1_close', 0.0, bsp1_open, step=0.01),
        # "cross_period": trial.suggest_int('cross_period', 8, 30),
        "bsp1_tp_long": profit_loss,  # trial.suggest_float('bsp1_tp_long', 0.003, 0.02, step=0.001),
        # "bsp1_sl_long": trial.suggest_float('bsp1_sl_long', 0.002, 0.005, step=0.001),
        "bsp1_tp_short": profit_loss,  # trial.suggest_float('bsp1_tp_short', 0.003, 0.02, step=0.001),
        # "bsp1_sl_short": trial.suggest_float('bsp1_sl_short', 0.002, 0.005, step=0.001),
    }
    code, score, capital, plot_driver = run_trade(code, lv_list, begin_time, end_time, dataset_params, model,
                                                  feature_names,
                                                  trade_params)
    print(f"{code} score: {score} {capital} {trade_params}")
    trial.set_user_attr("trade_params", trade_params)
    trial.set_user_attr("capital", capital)
    trial.set_user_attr("plot_driver", plot_driver)
    return score


def get_model(code, params, X_train, X_test, y_train, y_test):
    X_train, X_test, y_train, y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()

    X_train = X_train.to_numpy(np.float32)
    profit_loss_train = y_train["profit_loss"].to_numpy(np.float32)
    profit_loss_test = y_test["profit_loss"].to_numpy(np.float32)
    label_train = y_train["label"].to_numpy(int)
    X_test = X_test.to_numpy(np.float32)
    label_test = y_test["label"].to_numpy(int)

    size0 = len(label_train[label_train == 0])
    size1 = len(label_train[label_train == 1])
    w0 = (size0 + size1) / size0
    w1 = (size0 + size1) / size1
    sample_weight = []
    min_profit_loss = np.min(profit_loss_train)
    assert min_profit_loss > 0
    max_profit_loss = np.max(profit_loss_train)
    for idx, label in enumerate(label_train):
        if label == 0:
            w = w0 * (profit_loss_train[idx] - min_profit_loss) / (max_profit_loss - min_profit_loss)
        else:
            w = w1 * (profit_loss_train[idx] - min_profit_loss) / (max_profit_loss - min_profit_loss)
        sample_weight.append(w)
    sample_weight = np.array(sample_weight)

    model = lgb.LGBMClassifier(**params)
    X_train, label_train, sample_weight = shuffle(X_train, label_train, sample_weight, random_state=42)
    model.fit(X=X_train, y=label_train, eval_set=[(X_test, label_test)], sample_weight=sample_weight,
              callbacks=[early_stopping(stopping_rounds=100, verbose=False)])
    test_probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(label_test, test_probs)
    print(f"{code} AUC {auc}")
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    thresholds = np.linspace(0.0, 1.0, 100)
    print("计算正确率、精确率、召回率和F1分数")
    for threshold in thresholds:
        y_pred = (test_probs >= threshold).astype(int)
        accuracy = accuracy_score(label_test, y_pred)
        precision = precision_score(label_test, y_pred)
        recall = recall_score(label_test, y_pred)
        f1 = f1_score(label_test, y_pred)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    # 绘制曲线
    matplotlib.use('Agg')
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, accuracies, marker='x', label='Accuracy')
    plt.plot(thresholds, precisions, marker='x', label='Precision')
    plt.plot(thresholds, recalls, marker='x', label='Recall')
    plt.plot(thresholds, f1_scores, marker='x', label='F1 Score')

    plt.xlabel('Probability Threshold')
    plt.ylabel('Score')
    plt.title(f'{code} AUC:{auc} Threshold vs Accuracy,Precision, Recall,F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./result/{code}_metric.png")
    plt.clf()
    matplotlib.use('Agg')

    return model


def optimize_model(trial, X_train, X_test, y_train, y_test, seed):
    X_train, X_test, y_train, y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()

    X_train = X_train.to_numpy(np.float32)
    profit_loss_train = y_train["profit_loss"].to_numpy(np.float32)
    profit_loss_test = y_test["profit_loss"].to_numpy(np.float32)
    label_train = y_train["label"].to_numpy(int)
    X_test = X_test.to_numpy(np.float32)
    label_test = y_test["label"].to_numpy(int)
    model_params = {
        'device': 'gpu',
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'num_threads': 1,
        "random_state": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'num_leaves': trial.suggest_int('num_leaves', 8, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, step=0.01),
        'n_estimators': trial.suggest_int('n_estimators', 50, 5000),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.01, 1.0, step=0.01),
        'min_child_weight': trial.suggest_float('min_child_weight', 0, 10, step=0.01),
        'subsample': 0.9,  # trial.suggest_float('subsample', 0.7, 1.0, step=0.01),
        'subsample_freq': 4,  # trial.suggest_int('subsample_freq', 1, 7),
        'colsample_bytree': 0.9,  # trial.suggest_float('colsample_bytree', 0.7, 1.0, step=0.01),
        'reg_alpha': trial.suggest_int('reg_alpha', 0, 100),
        'reg_lambda': trial.suggest_int('reg_lambda', 0, 100),
        # 'feature_fraction': 0.95,  # trial.suggest_float('feature_fraction', 0.7, 1.0, step=0.01),
        # 'bagging_fraction': 0.95,  # trial.suggest_float('bagging_fraction', 0.7, 1.0, step=0.01),
        'gpu_platform_id': 0,  # 可选，通常为 0
        'gpu_device_id': trial.number % 2  # 可选，通常为 0

    }
    size0 = len(label_train[label_train == 0])
    size1 = len(label_train[label_train == 1])
    w0 = (size0 + size1) / size0
    w1 = (size0 + size1) / size1
    sample_weight = []
    min_profit_loss = np.min(profit_loss_train)
    assert min_profit_loss > 0
    max_profit_loss = np.max(profit_loss_train)
    for idx, label in enumerate(label_train):
        if label == 0:
            w = w0 * (profit_loss_train[idx] - min_profit_loss) / (max_profit_loss - min_profit_loss)
        else:
            w = w1 * (profit_loss_train[idx] - min_profit_loss) / (max_profit_loss - min_profit_loss)
        sample_weight.append(w)

    sample_weight = np.array(sample_weight)
    model = lgb.LGBMClassifier(**model_params)
    X_train, label_train, sample_weight = shuffle(X_train, label_train, sample_weight, random_state=42)
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # cv_aucs = []
    # for train_index, val_index in skf.split(X_train, y_train):
    #     X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
    #     y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
    #     sample_weight_fold_train = sample_weight[train_index]
    #
    #     model.fit(X=X_fold_train, y=y_fold_train, eval_set=[(X_fold_val, y_fold_val)],
    #               sample_weight=sample_weight_fold_train,
    #               callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
    #     y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
    #     auc = roc_auc_score(y_fold_val, y_pred_proba)
    #     cv_aucs.append(auc)
    # score = np.mean(cv_aucs)
    model.fit(X=X_train, y=label_train, eval_set=[(X_test, label_test)], sample_weight=sample_weight,
              callbacks=[early_stopping(stopping_rounds=100, verbose=False)])
    val_probs = model.predict_proba(X_test)[:, 1]

    profit_loss = np.mean(profit_loss_test[label_test == 1])
    # 计算AUC并记录
    score = roc_auc_score(label_test, val_probs)
    trial.set_user_attr("model", model)
    trial.set_user_attr("model_params", model_params)
    trial.set_user_attr("profit_loss", profit_loss)

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
    factors = None
    for chan_snapshot in chan.step_load():

        lv_chan = chan_snapshot[0]
        bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]
        if last_bsp.klu.klc.idx == lv_chan[-1].idx and (BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
            factors = FeatureFactors(chan_snapshot[0],
                                     MAX_BI=params["MAX_BI"],
                                     MAX_ZS=params["MAX_ZS"],
                                     MAX_SEG=params["MAX_SEG"],
                                     MAX_SEGSEG=params["MAX_SEGSEG"],
                                     MAX_SEGZS=params["MAX_SEGZS"]
                                     ).get_factors()
        if last_bsp.klu.idx not in bsp_dict and last_bsp.klu.klc.idx == lv_chan[-2].idx and (
                BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type) and lv_chan[-2].fx != FX_TYPE.UNKNOWN:
            if factors is None:
                continue
            last_bsp.add_feat(factors)
            features = dict(last_bsp.features.items())
            bsp_dict[last_bsp.klu.idx] = {
                "feature": features,
                "price": lv_chan[-1][-1].close,
                "is_buy": last_bsp.is_buy,
                "state": 0,
                "last_bsp": last_bsp
            }

    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp(0) if bsp.bi.next is not None]
    rows = []
    for bsp_klu_idx, feature_info in bsp_dict.items():
        label = int(bsp_klu_idx in bsp_academy)
        profit_loss = 0
        if label == 1:
            last_bsp = feature_info["last_bsp"]
            if last_bsp.is_buy:
                if last_bsp.bi.next is None:
                    continue
                profit_loss = (last_bsp.bi.next.get_end_val() / feature_info["price"]) - 1
                assert profit_loss > 0
            else:
                if last_bsp.bi.next is None:
                    continue
                profit_loss = last_bsp.bi.next.get_end_val() / feature_info["price"] - 1
                assert profit_loss < 0
        if label == 0:
            last_bsp = feature_info["last_bsp"]
            if last_bsp.is_buy:
                profit_loss = last_bsp.bi.get_end_val() / feature_info["price"] - 1
                assert profit_loss < 0
            else:
                profit_loss = last_bsp.bi.get_end_val() / feature_info["price"] - 1
                assert profit_loss > 0

        row = {"label": label, "profit_loss": abs(profit_loss)}
        feature = feature_info["feature"]
        row.update(feature)
        rows.append(row)
    df = pd.DataFrame(rows)
    return code, df


def run_codes(codes):
    lv_list = [KL_TYPE.K_30M]
    begin_time = "2020-01-01 00:00:00"
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
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
        "cal_rsi": True,
        "cal_boll": True,
        "cal_kdj": True,
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
    all_feature_names.remove("profit_loss")
    all_x_train = pd.DataFrame(columns=list(all_feature_names))
    all_y_train = pd.DataFrame()
    for code, dataset in results:
        dataset = pd.DataFrame(dataset, columns=["label", "profit_loss"] + list(all_feature_names))
        feature_names[code] = list(all_feature_names)
        X_train[code], X_test[code], y_train[code], y_test[code] = train_test_split(dataset[list(all_feature_names)],
                                                                                    dataset[["label", "profit_loss"]],
                                                                                    test_size=0.2,
                                                                                    shuffle=False)
        all_x_train = pd.concat([all_x_train, X_train[code]])
        all_y_train = pd.concat([all_y_train, y_train[code]])

    for code in codes:
        print(f"{code} Train Dataset:{X_train[code].shape} Test Dataset:{X_test[code].shape}")
        print(f"{code}找最优模型参数")
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
        study.optimize(
            lambda trial: optimize_model(trial, X_train[code], X_test[code], y_train[code], y_test[code], seed=42),
            n_trials=1000,
            n_jobs=-1)
        model = study.best_trial.user_attrs["model"]
        model_params = study.best_trial.user_attrs["model_params"]
        profit_loss = study.best_trial.user_attrs["profit_loss"]
        print(f"{code} AUC {study.best_trial.value} {profit_loss} {model_params}")
        print(f"{code}找最优交易参数")
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(),
                                    storage=storage)
        study.optimize(
            lambda trial: optimize_trade(trial, code, lv_list, val_begin_time, val_end_time, dataset_params, model,
                                         feature_names[code], profit_loss),
            n_trials=50,
            n_jobs=8)
        trade_params = study.best_trial.user_attrs["trade_params"]
        capital = study.best_trial.user_attrs["capital"]
        plot_driver = study.best_trial.user_attrs["plot_driver"]
        plot_driver.save2img(f"./result/{code}_train.png")
        plt.clf()
        print(f"{code} Best Score {study.best_trial.value} Best capital {capital} ")
        print("开始测试")
        code, score, capital, plot_driver = run_trade(code, lv_list, test_begin_time, test_end_time, dataset_params,
                                                      model,
                                                      feature_names[code],
                                                      trade_params)
        plot_driver.save2img(f"./result/{code}_test.png")
        plt.clf()
        with open("./result/report.log", mode="a") as file:
            seq = f"{code} 2024年实战盈利:{score} {capital} ,dataset:{dataset_params},trade:{trade_params}\n"
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

    run_codes(symbols)


if __name__ == "__main__":
    main()
