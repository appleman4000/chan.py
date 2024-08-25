# encoding:utf-8
import logging
from typing import Dict

import lightgbm as lgb
import numpy as np
import optuna
import pandas
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

logging.getLogger('lightgbm').setLevel(logging.CRITICAL)
optuna.logging.set_verbosity(optuna.logging.ERROR)


def predict_bsp(model, last_bsp: CBS_Point, feature_names):
    missing = float('nan')
    feature_arr = [missing] * len(feature_names)
    for feat_name, feat_value in last_bsp.features.items():
        i = feature_names.index(feat_name)
        if i >= 0:
            feature_arr[i] = feat_value
    feature_arr = [feature_arr]
    return model.predict_proba(feature_arr)[0][1]
    # 使用 Optuna 定义超参数的搜索空间


best_model = {}
lv_list = [KL_TYPE.K_30M]
data_src = DATA_SRC.FOREX
begin_time = "2021-01-01 00:00:00"
end_time = "2022-01-01 00:00:00"
val_begin_time = "2022-01-01 00:00:00"
val_end_time = "2023-01-01 00:00:00"
test_begin_time = "2023-01-01 00:00:00"
test_end_time = "2024-01-01 00:00:00"

trade_param_grid = {
}


def trade_objective(trial, chan_param_grid, model, feature_names):
    config = CChanConfig(conf=chan_param_grid.copy())
    chan = CChan(
        code=code,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        begin_time=val_begin_time,
        end_time=val_end_time
    )
    trade_param_grid.update({
        "bsp1_pred_long_open": trial.suggest_float('bsp1_pred_long_open', 0.6, 0.8, step=0.01),
        "bsp1_pred_short_open": trial.suggest_float('bsp1_pred_short_open', 0.6, 0.8, step=0.01),
        "bsp1_pred_long_exit": trial.suggest_float('bsp1_pred_long_exit', 0.0, 0.7, step=0.01),
        "bsp1_pred_short_exit": trial.suggest_float('bsp1_pred_short_exit', 0.0, 0.7, step=0.01),
        "tp_long": trial.suggest_float('tp_long', 0.002, 0.03, step=0.001),
        "sl_long": trial.suggest_float('sl_long', 0.002, 0.01, step=0.001),
        "tp_short": trial.suggest_float('tp_short', 0.002, 0.03, step=0.001),
        "sl_short": trial.suggest_float('sl_short', 0.002, 0.01, step=0.001),
    })
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
        if last_bsp.klu.klc.idx == lv_chan[-1].idx and (BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
            factors = FeatureFactors(chan[0], MAX_BI=chan_param_grid["MAX_BI"],
                                     MAX_ZS=chan_param_grid["MAX_ZS"],
                                     MAX_SEG=chan_param_grid["MAX_SEG"],
                                     MAX_SEGSEG=chan_param_grid["MAX_SEGSEG"],
                                     MAX_SEGZS=chan_param_grid["MAX_SEGZS"]).get_factors()
            for key in factors.keys():
                last_bsp.features.add_feat(key, factors[key])
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


model_param_grid = {
    'seed': 42,
    'device': 'cpu',
    'objective': 'binary',
    'min_split_gain': 0,
    'min_child_weight': 1e-3,
    'boosting_type': 'gbdt',
}


def model_objective(trial, X_train, X_val, y_train, y_val):
    model_param_grid.update({
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 50.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 50.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),

    })
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    model_param_grid.update(
        {
            "class_weight": {
                0: class_weights[0],
                1: class_weights[1],
            }
        }
    )
    model_param_grid["random_state"] = model_param_grid["seed"]
    model_param_grid["bagging_seed"] = model_param_grid["seed"]
    model_param_grid["feature_fraction_seed"] = model_param_grid["seed"]
    model_param_grid["gpu_device_id"] = 0
    model_param_grid["gpu_platform_id"] = 0

    model = LGBMClassifier(verbose=-1, **model_param_grid)
    callbacks = [lgb.log_evaluation(period=1), lgb.early_stopping(stopping_rounds=20, verbose=False)]
    model.fit(X_train, y_train, eval_set=(X_val, y_val), eval_metric="auc", callbacks=callbacks)
    y_pred = model.predict_proba(X_val)[:, 1]
    # 计算 F1 分数（适用于二分类）
    score = roc_auc_score(y_val, y_pred)
    trial.set_user_attr("model", model)
    # 返回平均 F1 分数
    return score


chan_param_grid = {
    "trigger_step": True,  # 打开开关！
    "skip_step": 1000,
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


def all_objective(trial, code):
    chan_param_grid.update({
        # "zs_combine": trial.suggest_categorical('zs_combine', [True, False]),
        # "zs_combine_mode": trial.suggest_categorical('zs_combine_mode', ["zs", "peak"]),
        # "one_bi_zs": trial.suggest_categorical('one_bi_zs', [True, False]),
        # "zs_algo": trial.suggest_categorical('zs_algo', ["normal", "over_seg", "auto"]),
        # "bi_algo": trial.suggest_categorical('bi_algo', ["normal", "fx"]),
        # "bi_strict": trial.suggest_categorical('bi_strict', [True, False]),
        # "gap_as_kl": trial.suggest_categorical('gap_as_kl', [True, False]),
        # "bi_end_is_peak": trial.suggest_categorical('bi_end_is_peak', [True, False]),
        # "bi_fx_check": trial.suggest_categorical('bi_fx_check', ["strict", "totally", "loss", "half"]),
        # "bi_allow_sub_peak": trial.suggest_categorical('bi_allow_sub_peak', [True, False]),
        # "seg_algo": trial.suggest_categorical('seg_algo', ["chan", ]),
        # "left_seg_method": trial.suggest_categorical('left_seg_method', ["all", "peak"]),
        # "bs1_peak": trial.suggest_categorical('bs1_peak', [True, False]),
        "boll_n": trial.suggest_int('boll_n', 14, 30),
        "macd": trial.suggest_categorical('macd', [{"fast": 12, "slow": 26, "signal": 9}, ]),
        "rsi_cycle": trial.suggest_int('rsi_cycle', 14, 30),
        "kdj_cycle": trial.suggest_int('kdj_cycle', 9, 30),
        "MAX_BI": trial.suggest_int('MAX_BI', 1, 12),
        "MAX_ZS": trial.suggest_int('MAX_ZS', 1, 3),
        "MAX_SEG": trial.suggest_int('MAX_SEG', 1, 3),
        "MAX_SEGSEG": trial.suggest_int('MAX_SEGSEG', 1, 3),
        "MAX_SEGZS": trial.suggest_int('MAX_SEGZS', 1, 3),

    })
    # if chan_param_grid["zs_algo"] == "over_seg":
    #     chan_param_grid["one_bi_zs"] = False
    config = CChanConfig(conf=chan_param_grid.copy())

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
                                     MAX_BI=chan_param_grid["MAX_BI"],
                                     MAX_ZS=chan_param_grid["MAX_ZS"],
                                     MAX_SEG=chan_param_grid["MAX_SEG"],
                                     MAX_SEGSEG=chan_param_grid["MAX_SEGSEG"],
                                     MAX_SEGZS=chan_param_grid["MAX_SEGZS"]
                                     ).get_factors()
            for key in factors.keys():
                bsp_dict[last_bsp.klu.idx]['feature'].add_feat(key, factors[key])

    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp(0)]
    rows = []
    for bsp_klu_idx, feature_info in bsp_dict.items():
        label = int(bsp_klu_idx in bsp_academy)
        feature = feature_info["feature"]
        row = {"label": label}
        for feature_name, value in feature.items():
            row.update({feature_name: value})
        rows.append(row)
    df = pandas.DataFrame(rows)
    labels = df["label"].to_numpy(dtype=int)
    features = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    feature_names = df.columns[1:].tolist()

    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, shuffle=False,
                                                      random_state=42)

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
    study.optimize(lambda trial: model_objective(trial, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val),
                   n_trials=200, n_jobs=-1)
    model_param_grid.update(study.best_params)
    model = study.best_trial.user_attrs["model"]

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
    study.optimize(lambda trial: trade_objective(trial, chan_param_grid, model, feature_names), n_trials=200, n_jobs=-1)
    trade_param_grid.update(study.best_params)
    trial.set_user_attr("chan_param_grid", chan_param_grid)
    trial.set_user_attr("model", model)
    trial.set_user_attr("model_param_grid", model_param_grid)
    trial.set_user_attr("trade_param_grid", trade_param_grid)
    trial.set_user_attr("feature_names", feature_names)
    return study.best_value


if __name__ == "__main__":
    reconnect_mt5()
    symbols = [
        # Major
        "EURUSD",
        # "GBPUSD",
        # "AUDUSD",
        # "NZDUSD",
        # "USDJPY",
        # "USDCAD",
        # "USDCHF",
        # # Crosses
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
        # 优化参数开始
        study.optimize(lambda trial: all_objective(trial, code=code), n_trials=100, n_jobs=-1)
        # 优化参数结束
        print(f"{code}:校验集 {study.best_value}")
        chan_param_grid = study.best_trial.user_attrs["chan_param_grid"]
        model = study.best_trial.user_attrs["model"]
        model_param_grid = study.best_trial.user_attrs["model_param_grid"]
        trade_param_grid = study.best_trial.user_attrs["trade_param_grid"]
        feature_names = study.best_trial.user_attrs["feature_names"]

        config = CChanConfig(chan_param_grid.copy())
        chan = CChan(
            code=code,
            data_src=data_src,
            lv_list=lv_list,
            config=config,
            begin_time=test_begin_time,
            end_time=test_end_time
        )
        capital = 10000
        lots = 1
        money = 100000 * lots
        capitals = np.array([])
        profits = np.array([])
        fee = 1.0003
        long_order = 0
        short_order = 0
        history_long_orders = 0
        history_short_orders = 0
        for chan_snapshot in chan.step_load():

            lv_chan = chan_snapshot[0]
            profit = 0
            bsp_list = chan.get_bsp(0)  # 获取买卖点列表
            if not bsp_list:
                continue
            last_bsp = bsp_list[-1]
            if last_bsp.klu.klc.idx == lv_chan[-1].idx and (
                    BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
                factors = FeatureFactors(chan[0], MAX_BI=chan_param_grid["MAX_BI"],
                                         MAX_ZS=chan_param_grid["MAX_ZS"],
                                         MAX_SEG=chan_param_grid["MAX_SEG"],
                                         MAX_SEGSEG=chan_param_grid["MAX_SEGSEG"],
                                         MAX_SEGZS=chan_param_grid["MAX_SEGZS"]).get_factors()
                for key in factors.keys():
                    last_bsp.features.add_feat(key, factors[key])
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

        print(f"{code}:测试集 {capital}")
        print(f"chan:{chan_param_grid},model:{model_param_grid},trade:{trade_param_grid}")
