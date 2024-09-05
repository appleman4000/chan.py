# cython: language_level=3
# encoding:utf-8
import logging
import math
import os.path
import pickle
import warnings
from typing import Dict

import matplotlib
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import class_weight

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, KL_TYPE, AUTYPE, BSP_TYPE
from FeatureEngineering import FeatureFactors
from GenerateDataset import T_SAMPLE_INFO
from Plot.PlotDriver import CPlotDriver

logging.getLogger('LGBMClassifier').setLevel(logging.CRITICAL)
optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore")


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
        "plot_kline": True,
        "plot_bi": False,
        "plot_seg": True,
        "plot_zs": True,
        "plot_bsp": False,
        "plot_marker": True,
    }
    plot_para = {
        "figure": {
            "x_range": 8000,
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
    plot_driver.save2img(f"{name}.png")


def predict_bsp(model, feature: dict, feature_names):
    features = [feature]
    features = pd.DataFrame(features, columns=feature_names).to_numpy(dtype=np.float32)
    return model.predict_proba(features)[0][1]


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
    bsp1_pred = 0
    for chan_snapshot in chan.step_load():

        lv_chan = chan_snapshot[0]
        profit = 0
        bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        if last_bsp.klu.idx not in bsp_dict and last_bsp.klu.klc.idx == lv_chan[-1].idx and (
                BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
            factors = FeatureFactors(chan_snapshot[0], MAX_BI=dataset_params["MAX_BI"],
                                     MAX_ZS=dataset_params["MAX_ZS"],
                                     MAX_SEG=dataset_params["MAX_SEG"],
                                     MAX_SEGSEG=dataset_params["MAX_SEGSEG"],
                                     MAX_SEGZS=dataset_params["MAX_SEGZS"]).get_factors()
            last_bsp.features.add_feat(factors)
            bsp1_pred = predict_bsp(model=model, feature=dict(last_bsp.features.items()), feature_names=feature_names)
        else:
            bsp1_pred = 0
        # if last_bsp.klu.klc.idx - lv_chan[-1].idx <= -3:
        #     bsp1_pred = 0
        if long_order > 0:
            # 止盈
            close_price = round(lv_chan[-1][-1].close, 5)
            long_profit = close_price / long_order - 1
            # exit_rule = bsp1_pred > trade_params["bsp1_close"] and not last_bsp.is_buy
            # exit_rule = lv_chan[-2].fx == FX_TYPE.TOP and long_profit > 0.004
            # 最大止盈止损保护
            tp = long_profit > trade_params["bsp1_tp_long"]
            sl = long_profit < -trade_params["bsp1_sl_long"]
            if tp or sl:
                long_order = 0
                profit += round(long_profit * money, 2)
        if short_order > 0:
            close_price = round(lv_chan[-1][-1].close, 5)
            short_profit = short_order / close_price - 1
            # exit_rule = bsp1_pred > trade_params["bsp1_close"] and last_bsp.is_buy
            # exit_rule = lv_chan[-2].fx == FX_TYPE.BOTTOM and short_profit > 0.004
            # 最大止盈止损保护
            tp = short_profit > trade_params["bsp1_tp_short"]
            sl = short_profit < -trade_params["bsp1_sl_short"]
            if tp or sl:
                short_order = 0
                profit += round(short_profit * money, 2)

        if long_order == 0 and short_order == 0:
            if bsp1_pred >= trade_params["bsp1_open"] and last_bsp.is_buy:
                bsp_dict[last_bsp.klu.idx] = {
                    "is_buy": last_bsp.is_buy,
                    "open_time": lv_chan[-1][-1].time,
                }
                long_order = round(lv_chan[-1][-1].close * fee, 5)
            if bsp1_pred >= trade_params["bsp1_open"] and not last_bsp.is_buy:
                bsp_dict[last_bsp.klu.idx] = {
                    "is_buy": last_bsp.is_buy,
                    "open_time": lv_chan[-1][-1].time,
                }
                short_order = round(lv_chan[-1][-1].close / fee, 5)
        capital += profit
        if profit != 0:
            trades.append(profit)
            print(f"{profit} {capital}")
        capitals.append(capital)
    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
    for bsp_klu_idx, feature_info in bsp_dict.items():
        label = int(bsp_klu_idx in bsp_academy)  # 以买卖点识别是否准确为label
        plot_marker[feature_info["open_time"].to_str()] = (
            "√" if label else "×", "down" if feature_info["is_buy"] else "up")
    plot(chan, plot_marker,
         name=f"{begin_time.replace(':', '_').replace(' ', '_').replace('-', '_')} {end_time.replace(':', '_').replace(' ', '_').replace('-', '_')}")
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

    return score, capital


def optimize_trade(trial, code, lv_list, begin_time, end_time, dataset_params, model, feature_names):
    trade_params = {
        "bsp1_open": trial.suggest_float('bsp1_open', 0.2, 0.8, step=0.01),
        "bsp1_close": trial.suggest_float('bsp1_close', 0.0, 0.5, step=0.01),
        "bsp1_tp_long": trial.suggest_float('bsp1_tp_long', 0.003, 0.02, step=0.001),
        "bsp1_sl_long": trial.suggest_float('bsp1_sl_long', 0.001, 0.005, step=0.001),
        "bsp1_tp_short": trial.suggest_float('bsp1_tp_short', 0.003, 0.02, step=0.001),
        "bsp1_sl_short": trial.suggest_float('bsp1_sl_short', 0.001, 0.005, step=0.001),
    }
    score, capital = run_trade(code, lv_list, begin_time, end_time, dataset_params, model, feature_names, trade_params)
    print(f"{code} score: {score} {capital} {trade_params}")
    trial.set_user_attr("trade_params", trade_params)
    trial.set_user_attr("capital", capital)
    return score


def get_model(params, X_dataset, y_dataset):
    model = LGBMClassifier(**params)
    model.fit(X_dataset, y_dataset)
    return model


def optimize_model(trial, X_dataset, y_dataset):
    model_params = {
        'seed': 42,
        'device': 'cpu',
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'num_threads': 1,
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 10.0, step=0.01),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 100.0, step=0.001),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.01),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.01),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 100, step=0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 100, step=0.1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0, step=0.01),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0, step=0.01),

    }

    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_dataset), y=y_dataset
    )
    model_params.update(
        {
            "random_state": model_params["seed"],
            "bagging_seed": model_params["seed"],
            "feature_fraction_seed": model_params["seed"],
            "class_weight": {
                0: class_weights[0],
                1: class_weights[1],
            }
        }
    )
    model = get_model(model_params, X_dataset, y_dataset)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
    scores = cross_val_score(model, X_dataset, y_dataset, cv=kf,
                             scoring=auc_scorer)

    # y_pred = model.predict_proba(X_val)[:, 1]
    # score = roc_auc_score(y_val, y_pred)
    score = scores.mean()
    trial.set_user_attr("model_params", model_params)
    return score


def get_dataset(code, lv_list, begin_time, end_time, params, feature_names=None):
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
    for chan_snapshot in chan.step_load():

        lv_chan = chan_snapshot[0]
        bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        if last_bsp.klu.idx not in bsp_dict and last_bsp.klu.klc.idx == lv_chan[-1].idx:
            bsp_dict[last_bsp.klu.idx] = {
                "feature": dict(last_bsp.features.items()),
            }
            factors = FeatureFactors(chan_snapshot[0],
                                     MAX_BI=params["MAX_BI"],
                                     MAX_ZS=params["MAX_ZS"],
                                     MAX_SEG=params["MAX_SEG"],
                                     MAX_SEGSEG=params["MAX_SEGSEG"],
                                     MAX_SEGZS=params["MAX_SEGZS"]
                                     ).get_factors()
            bsp_dict[last_bsp.klu.idx]["feature"].update(factors)

    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp(0)]
    rows = []
    for bsp_klu_idx, feature_info in bsp_dict.items():
        label = int(bsp_klu_idx in bsp_academy)
        row = {"label": label}
        feature = feature_info["feature"]
        row.update(feature)
        rows.append(row)
    if feature_names is None:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(rows, columns=["label"] + feature_names)
    labels = df["label"].to_numpy(dtype=int)
    features = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    feature_names = df.columns[1:].tolist()
    return features, labels, feature_names


def optimize_dataset(trial, code, lv_list, begin_time, end_time):
    dataset_params = {
        "trigger_step": True,  # 打开开关！
        "skip_step": 500,
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs_type": '1,1p',
        "cal_rsi": True,
        "cal_kdj": True,
        "cal_demark": False,
        "kl_data_check": False,
        "MAX_BI": 7,
        "MAX_ZS": 1,
        "MAX_SEG": 1,
        "MAX_SEGSEG": 1,
        "MAX_SEGZS": 1,
        "divergence_rate": trial.suggest_float('divergence_rate', 0.6, 0.8, step=0.05),
        "macd_algo": trial.suggest_categorical('macd_algo', ["area", "peak", "full_area", "diff", "slope", "amp"]),
    }
    X_dataset, y_dataset, feature_names = get_dataset(code, lv_list, begin_time, end_time, dataset_params)
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
    study.optimize(lambda t: optimize_model(t, X_dataset, y_dataset),
                   n_trials=500, n_jobs=4)
    print(f"{code} dataset auc:{study.best_trial.value}")
    return study.best_trial.value


def run_code(code):
    lv_list = [KL_TYPE.K_30M]
    begin_time = "2010-01-01 00:00:00"
    end_time = "2022-01-01 00:00:00"
    val_begin_time = "2022-01-01 00:00:00"
    val_end_time = "2023-01-01 00:00:00"
    test_begin_time = "2023-01-01 00:00:00"
    test_end_time = "2024-01-01 00:00:00"
    dataset_params = {
        "trigger_step": True,  # 打开开关！
        "skip_step": 500,
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs_type": '1,1p',
        "cal_rsi": True,
        "cal_kdj": True,
        "cal_demark": False,
        "kl_data_check": False,
        "MAX_BI": 7,
        "MAX_ZS": 1,
        "MAX_SEG": 1,
        "MAX_SEGSEG": 1,
        "MAX_SEGZS": 1,
        "divergence_rate": float("inf"),
        "macd_algo": "area"
    }
    if not os.path.exists(f"{code}.train"):
        X_dataset, y_dataset, feature_names = get_dataset(code, lv_list, begin_time, end_time, dataset_params)
        with open(f"{code}.train", "wb") as fid:
            pickle.dump((X_dataset, y_dataset, feature_names), fid)
    else:
        with open(f"{code}.train", "rb") as fid:
            X_dataset, y_dataset, feature_names = pickle.load(fid)

    n_0 = np.sum(y_dataset == 0)
    n_1 = np.sum(y_dataset == 1)
    scale_pos_weight = n_0 / n_1
    print(f"dataset: {X_dataset.shape}, scale_pos_weight:{scale_pos_weight}")
    print(f"找最优模型参数")
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
    study.optimize(
        lambda trial: optimize_model(trial, X_dataset, y_dataset),
        n_trials=500,
        n_jobs=-1)
    model_params = study.best_trial.user_attrs["model_params"]
    print(f"{study.best_trial.value} {model_params}")
    print(f"训练模型")
    model = get_model(model_params, X_dataset, y_dataset)

    if not os.path.exists(f"{code}.val"):
        X_dataset_val, y_dataset_val, feature_names = get_dataset(code, lv_list, val_begin_time, val_end_time,
                                                                  dataset_params,
                                                                  feature_names)
        with open(f"{code}.val", "wb") as fid:
            pickle.dump((X_dataset_val, y_dataset_val, feature_names), fid)
    else:
        with open(f"{code}.val", "rb") as fid:
            X_dataset_val, y_dataset_val, feature_names = pickle.load(fid)
    y_prob = np.array(model.predict_proba(X_dataset_val)[:, 1])
    auc = roc_auc_score(y_dataset_val, y_prob)
    print(f"测试集模型 auc:{auc}")
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    recalls = []
    f1_scores = []

    # 计算每个阈值下的正确率、召回率和F1分数
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        accuracy = accuracy_score(y_dataset_val, y_pred)
        recall = recall_score(y_dataset_val, y_pred)
        f1 = f1_score(y_dataset_val, y_pred)
        accuracies.append(accuracy)
        recalls.append(recall)
        f1_scores.append(f1)

    # 绘制曲线
    matplotlib.use('Agg')
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, accuracies, marker='o', label='Accuracy')
    plt.plot(thresholds, recalls, marker='x', label='Recall', linestyle='--')
    plt.plot(thresholds, f1_scores, marker='s', label='F1 Score', linestyle='-.')

    plt.xlabel('Probability Threshold')
    plt.ylabel('Score')
    plt.title('Threshold vs Accuracy, Recall, and F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./{code}.png")
    plt.clf()
    matplotlib.use('Agg')
    # feature_importances = model.feature_importances_
    # # 特征名称
    # # 创建特征重要性数据框
    # importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    # importance_df = importance_df.sort_values(by='Importance', ascending=False)
    # pd.set_option('display.max_rows', None)
    # print(importance_df)
    trade_params = {
        "bsp1_open": 0.6,
        "bsp1_close": 0.5,
        "bsp1_tp_long": 0.003,
        "bsp1_sl_long": 0.003,
        "bsp1_tp_short": 0.003,
        "bsp1_sl_short": 0.003,
    }
    score1, capital1 = run_trade(code, lv_list, val_begin_time, val_end_time, dataset_params, model,
                                 feature_names,
                                 trade_params)

    print(f"{code} 第一年实战盈利:{score1} {capital1} {trade_params}")

    score2, capital2 = run_trade(code, lv_list, test_begin_time, test_end_time, dataset_params, model, feature_names,
                                 trade_params)

    print(f"{code} 第二年实战盈利:{score2} {capital2}")
    print(f"chan:{dataset_params},model:{model_params},trade:{trade_params}")
    with open("./report.log", mode="a") as file:
        seq = f"{code} auc:{auc} 第一年实战盈利:{capital1} 第二年实战盈利:{capital2},dataset:{dataset_params},model:{model_params},trade:{trade_params}\n"
        file.writelines(seq)


def main():
    symbols = [
        # Major
        "EURUSD",
        # "GBPUSD",
        # "AUDUSD",
        # "NZDUSD",
        # "USDJPY",
        # "USDCAD",
        # "USDCHF",
        # # # Crosses
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
        #

    ]
    Parallel(n_jobs=1, backend="multiprocessing")(
        delayed(run_code)(code) for code in symbols)


if __name__ == "__main__":
    main()
