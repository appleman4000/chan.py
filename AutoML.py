# cython: language_level=3
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

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, KL_TYPE, AUTYPE, BSP_TYPE
from FeatureEngineering import FeatureFactors
from GenerateDataset import T_SAMPLE_INFO

logging.getLogger('LGBMClassifier').setLevel(logging.CRITICAL)
optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore")


def predict_bsp(model, feature: dict, feature_names):
    features = [feature]
    features = pd.DataFrame(features, columns=feature_names).to_numpy()
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
            last_bsp.features.add_feat(factors)
            bsp1_pred = predict_bsp(model=model, feature=dict(last_bsp.features.items()), feature_names=feature_names)
        else:
            bsp1_pred = 0.0
        if long_order > 0:
            # 止盈
            close_price = round(lv_chan[-1][-1].close, 5)
            long_profit = close_price / long_order - 1
            exit_rule = bsp1_pred > 0 and not last_bsp.is_buy
            # 最大止盈止损保护
            tp = long_profit > trade_params["bsp1_tp_long"]
            sl = long_profit < -trade_params["bsp1_sl_long"]
            if tp or sl or exit_rule:
                long_order = 0
                profit += round(long_profit * money, 2)
        if short_order > 0:
            close_price = round(lv_chan[-1][-1].close, 5)
            short_profit = short_order / close_price - 1
            exit_rule = bsp1_pred > 0 and last_bsp.is_buy
            # 最大止盈止损保护
            tp = short_profit > trade_params["bsp1_tp_short"]
            sl = short_profit < -trade_params["bsp1_sl_short"]
            if tp or sl or exit_rule:
                short_order = 0
                profit += round(short_profit * money, 2)

        if long_order == 0 and short_order == 0:
            if bsp1_pred > trade_params["bsp1_open"] and last_bsp.is_buy:
                long_order = round(lv_chan[-1][-1].close * fee, 5)
            if bsp1_pred > trade_params["bsp1_open"] and not last_bsp.is_buy:
                short_order = round(lv_chan[-1][-1].close / fee, 5)

        capital += profit
    return capital


def optimize_trade(trial, code, lv_list, begin_time, end_time, dataset_params, model, feature_names):
    trade_params = {
        "bsp1_open": trial.suggest_float('bsp1_open', 0.6, 0.8, step=0.01),
        "bsp1_tp_long": trial.suggest_float('bsp1_tp_long', 0.005, 0.02, step=0.001),
        "bsp1_sl_long": trial.suggest_float('bsp1_sl_long', 0.001, 0.005, step=0.001),
        "bsp1_tp_short": trial.suggest_float('bsp1_tp_short', 0.005, 0.02, step=0.001),
        "bsp1_sl_short": trial.suggest_float('bsp1_sl_short', 0.001, 0.005, step=0.001),
    }
    score = run_trade(code, lv_list, begin_time, end_time, dataset_params, model, feature_names, trade_params)
    print(f"{code} score: {score} {trade_params}")
    trial.set_user_attr("trade_params", trade_params)
    return score


def get_model(params, X_train, X_val, y_train, y_val, class_weights):
    params.update(
        {
            "class_weight": {
                0: class_weights[0],
                1: class_weights[1],
            }
        }
    )
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def optimize_model(trial, X_train, X_val, y_train, y_val, class_weights):
    model_params = {
        'seed': 42,
        'device': 'cpu',
        'objective': 'binary',
        'min_split_gain': 0,
        'min_child_weight': 1e-3,
        'boosting_type': 'gbdt',
        'verbose': -1,
        'num_threads': 2,
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.01),
        'n_estimators': trial.suggest_int('n_estimators', 25, 500),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.05),
        'reg_alpha': trial.suggest_int('reg_alpha', 0, 100),
        'reg_lambda': trial.suggest_int('reg_lambda', 0, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0, step=0.05),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0, step=0.05),
    }

    model_params["random_state"] = model_params["seed"]
    model_params["bagging_seed"] = model_params["seed"]
    model_params["feature_fraction_seed"] = model_params["seed"]
    # 计算每个类别的权重
    model = get_model(model_params, X_train, X_val, y_train, y_val, class_weights)
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    trial.set_user_attr("model_params", model_params)
    return auc


def get_dataset(code, lv_list, begin_time, end_time, params):
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
            bsp_dict[last_bsp.klu.idx]['feature'].update(factors)

    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp(0)]
    rows = []
    for bsp_klu_idx, feature_info in bsp_dict.items():
        label = int(bsp_klu_idx in bsp_academy)
        row = {"label": label}
        feature = feature_info["feature"]
        row.update(feature)
        rows.append(row)
    df = pd.DataFrame(rows)
    labels = df["label"].to_numpy(dtype=int)
    features = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    feature_names = df.columns[1:].tolist()

    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, shuffle=False)
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    return X_train, X_val, y_train, y_val, feature_names, class_weights


def get_all_dataset(symbols, lv_list, begin_time, end_time, params):
    df_X_train = pd.DataFrame()
    df_X_val = pd.DataFrame()
    df_y_train = pd.DataFrame()
    df_y_val = pd.DataFrame()
    for code in symbols:
        print(f"{code} 产生训练数据")
        X_train, X_val, y_train, y_val, feature_names, class_weights = get_dataset(code, lv_list, begin_time,
                                                                                   end_time, params)
        new_df_X_train = pd.DataFrame(X_train, columns=feature_names)
        new_df_y_train = pd.DataFrame(y_train)

        new_df_X_val = pd.DataFrame(X_val, columns=feature_names)
        new_df_y_val = pd.DataFrame(y_val)

        df_X_train = pd.concat([df_X_train, new_df_X_train])
        df_y_train = pd.concat([df_y_train, new_df_y_train])

        df_X_val = pd.concat([df_X_val, new_df_X_val])
        df_y_val = pd.concat([df_y_val, new_df_y_val])

    feature_names = df_X_train.columns.tolist()
    y_train = df_y_train.to_numpy(int)[:, 0]
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    return df_X_train.to_numpy(np.float32), df_X_val.to_numpy(np.float32), df_y_train.to_numpy(
        np.float32)[:, 0], df_y_val.to_numpy(np.float32)[:, 0], feature_names, class_weights


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
        "MAX_BI": 0,
        "MAX_ZS": 0,
        "MAX_SEG": 0,
        "MAX_SEGSEG": 0,
        "MAX_SEGZS": 0,
        "divergence_rate": trial.suggest_float('divergence_rate', 0.6, 0.8, step=0.05),
        "macd_algo": trial.suggest_categorical('macd_algo', ["area", "peak", "full_area", "diff", "slope", "amp"]),
    }
    X_train, X_val, y_train, y_val, feature_names, class_weights = get_dataset(code, lv_list, begin_time, end_time,
                                                                               dataset_params)
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
    study.optimize(lambda t: optimize_model(t, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
                                            class_weights=class_weights),
                   n_trials=500, n_jobs=4)
    print(f"{code} dataset auc:{study.best_trial.value}")
    return study.best_trial.value


def run_codes(symbols):
    lv_list = [KL_TYPE.K_10M]
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
        "MAX_BI": 0,
        "MAX_ZS": 0,
        "MAX_SEG": 0,
        "MAX_SEGSEG": 0,
        "MAX_SEGZS": 0,
        "divergence_rate": float("inf"),
        "macd_algo": "peak"
    }
    X_train, X_val, y_train, y_val, feature_names, class_weights = get_all_dataset(symbols, lv_list, begin_time,
                                                                                   end_time, dataset_params)
    print(f"Training data: {X_train.shape}, Validation data: {X_val.shape} class_weights:{class_weights}")
    print(f"找最优模型参数")
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
    study.optimize(lambda trial: optimize_model(trial, X_train, X_val, y_train, y_val, class_weights), n_trials=500,
                   n_jobs=-1)
    model_params = study.best_trial.user_attrs["model_params"]
    print(f"{study.best_trial.value} {model_params}")
    print(f"训练模型")
    model = get_model(model_params, X_train, X_val, y_train, y_val, class_weights)
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"模型 auc:{auc}")
    feature_importances = model.feature_importances_
    # 特征名称
    # 创建特征重要性数据框
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    pd.set_option('display.max_rows', None)
    print(importance_df)
    for code in symbols:
        print(f"{code} 找最优交易参数")
        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), storage=storage)
        study.optimize(
            lambda trial: optimize_trade(trial, code, lv_list, val_begin_time, val_end_time, dataset_params, model,
                                         feature_names), n_trials=100, n_jobs=-1)
        trade_params = study.best_trial.user_attrs["trade_params"]
        best_value = study.best_trial.value
        print(f"{code} 优化后交易历史盈利:{best_value} {trade_params}")

        capital = run_trade(code, lv_list, test_begin_time, test_end_time, dataset_params, model, feature_names,
                            trade_params)

        print(f"{code} 实战盈利:{capital}")
        print(f"chan:{dataset_params},model:{model_params},trade:{trade_params}")
        with open("./report.log", mode="a") as file:
            seq = f"{code} auc:{auc} 优化后交易历史盈利:{best_value} 实战盈利:{capital},dataset:{dataset_params},model:{model_params},trade:{trade_params}\n"
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
    # joblib.Parallel(6, backend="multiprocessing")(
    #     joblib.delayed(run_code)(code)
    #     for code in symbols
    # )
    run_codes(symbols)


if __name__ == "__main__":
    main()
