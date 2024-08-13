# cython: language_level=3
import importlib.util
import inspect
import os
import pickle
from typing import Dict, TypedDict

import numpy as np

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, BSP_TYPE
from Common.CTime import CTime
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
    plot_driver.save2img("label.png")


def get_functions_from_module(module_path):
    spec = importlib.util.spec_from_file_location("module_name", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    functions = {name: obj for name, obj in inspect.getmembers(module, inspect.isfunction)}
    return functions


def calculate_functions(functions, *args, **kwargs):
    results = {}
    for name, func in functions.items():
        results.update(func(*args, **kwargs))
    return results


def get_lightgbm_model(train_data, train_label, param_grid):
    from lightgbm import LGBMClassifier
    param_grid["random_state"] = param_grid["seed"]
    param_grid["bagging_seed"] = param_grid["seed"]
    param_grid["feature_fraction_seed"] = param_grid["seed"]
    param_grid["gpu_device_id"] = 0
    param_grid["gpu_platform_id"] = 0

    print("train lightgbm model")
    # 平衡数据集权重（注意：LGBMClassifier不支持 class_weight："balanced",需要自己计算权重）
    from sklearn.utils import class_weight

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
    model = LGBMClassifier(**param_grid)
    model.fit(
        train_data,
        train_label,
        eval_set=[(train_data, train_label)],
    )
    print("train lightgbm model completely!")
    with open("model.hdf5", "wb") as f:
        pickle.dump(model, f)


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

    ]
    if os.path.exists("feature.libsvm"):
        os.remove("feature.libsvm")
    if os.path.exists("feature.meta"):
        os.remove("feature.meta")
    if os.path.exists("model.json"):
        os.remove("model.json")
    train_data = []
    train_label = []
    for code in symbols:
        begin_time = "2010-01-01 00:00:00"
        end_time = "2021-07-10 00:00:00"
        data_src = DATA_SRC.FOREX
        lv_list = [KL_TYPE.K_1H]

        config = CChanConfig({
            "trigger_step": True,  # 打开开关！
            "bi_strict": True,
            "skip_step": 500,
            "divergence_rate": float("inf"),
            "bsp2_follow_1": False,
            "bsp3_follow_1": False,
            "min_zs_cnt": 0,
            "bs1_peak": False,
            "macd_algo": "peak",
            "bs_type": '1,2,3a,1p,2s,3b',
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
            if last_bsp.klu.idx not in bsp_dict and cur_lv_chan[-1].idx == last_bsp.klu.klc.idx and \
                    (BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
                bsp_dict[last_bsp.klu.idx] = {
                    "feature": last_bsp.features,
                    "is_buy": last_bsp.is_buy,
                    "open_time": last_klu.time,
                }
                module_path = './FeatureEngineering.py'
                functions = get_functions_from_module(module_path)
                results = calculate_functions(functions, chan)
                for key in results.keys():
                    bsp_dict[last_bsp.klu.idx]['feature'].add_feat({key: results[key]})
                print(last_bsp.klu.time, last_bsp.is_buy)

        # 生成libsvm样本特征
        bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
        for bsp_klu_idx, feature_info in bsp_dict.items():
            label = int(bsp_klu_idx in bsp_academy)  # 以买卖点识别是否准确为label
            features = []  # List[(idx, value)]
            for feature_name, value in feature_info['feature'].items():
                features.append(value)
            train_data.append(features)
            train_label.append(label)
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    print(train_data.shape)
    print(train_label.shape)
    param_grid = {
        'seed': 42,
        # 'num_class': 1,
        'device': 'cpu',
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'min_split_gain': 0,
        'min_child_weight': 1e-3,
        'subsample': 0.9,
        'subsample_freq': 1,
        'colsample_bytree': 0.9,
        'reg_alpha': 0,
        'reg_lambda': 150,
        'verbose': -1,
        'num_threads': 1,
    }
    get_lightgbm_model(train_data=train_data, train_label=train_label, param_grid=param_grid)
