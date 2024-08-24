# cython: language_level=3
# encoding:utf-8
import json
import os
import pickle
from typing import Dict

from BuySellPoint.BS_Point import CBS_Point
from ChanConfig import CChanConfig
from FeatureEngineering import FeatureFactors
from Messenger import send_message

os.environ['KERAS_BACKEND'] = 'torch'

# import keras
import datetime
import sys
from multiprocessing import Value, Process

import numpy as np

from Chan import CChan
from Common.CEnum import DATA_SRC, KL_TYPE, BSP_TYPE
from CommonTools import shanghai_to_zurich_datetime, open_order, close_order

sys.setrecursionlimit(1000000)
app_id = 'cli_a63ae160c79d500b'
app_secret = 'BvtLvfCEPEePrqdw4vddScwhKVWSCtAx'
webhook_url = 'https://open.feishu.cn/open-apis/bot/v2/hook/d8ea8601-259b-4f48-a310-1e715909232e'
keras_model = {}
lgb_model = {}
meta = {}

config = CChanConfig({
    "trigger_step": True,  # 打开开关！
    "bi_strict": True,
    "skip_step": 0,
    "divergence_rate": float("inf"),
    "bsp2_follow_1": True,
    "bsp3_follow_1": True,
    "min_zs_cnt": 0,
    "bs1_peak": True,
    "macd_algo": "diff",
    "bs_type": '1,1p,2,2s,3a,3b',
    "print_warning": True,
    "zs_algo": "normal",
    "cal_rsi": True,
    "cal_kdj": True,
    "cal_demark": False,
    "kl_data_check": False
})


def predict_bsp(lgb_model, last_bsp: CBS_Point, meta: Dict[str, int]):
    missing = -9999999
    feature_arr = [missing] * len(meta)
    for feat_name, feat_value in last_bsp.features.items():
        if feat_name in meta:
            feature_arr[meta[feat_name]] = feat_value
    feature_arr = [feature_arr]

    return lgb_model.predict_proba(feature_arr)[0][1]


# def load_keras_model(symbols, is_all_in_one):
#     if is_all_in_one:
#         code = "all_in_one"
#         for bsp_type in ["2_2s"]:  # "1_1p",
#             keras_model[f"{code}_{bsp_type}"] = keras.saving.load_model(
#                 f"./TMP/{code}_{bsp_type}_model.keras")
#             meta[code] = json.load(open(f"./TMP/EURUSD_feature.meta", "r"))
#     else:
#         for code in symbols:
#             for bsp_type in ["1_1p", "2_2s"]:
#                 keras_model[f"{code}_{bsp_type}"] = keras.saving.load_model(
#                     f"./TMP/{code}_{bsp_type}_model.keras")
#             meta[code] = json.load(open(f"./TMP/{code}_feature.meta", "r"))


# def get_predict_value(code, chan: CChan, last_bsp: CBS_Point, plot_config, plot_para):
#     # if "all_in_one" in keras_model.keys():
#     code = "all_in_one"
#     # if BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type:
#     #     bsp_type = "1_1p"
#     #     model = keras_model[f"{code}_{bsp_type}"]
#     # elif BSP_TYPE.T2 in last_bsp.type or BSP_TYPE.T2S in last_bsp.type:
#     bsp_type = "2_2s"
#     model = keras_model[f"{code}_{bsp_type}"]
#     img_array = chan_to_png(chan, plot_config, plot_para, file_path="").astype(float)
#     # img_array /= 255.0
#     missing = 0
#     feature_arr = [missing] * len(meta[code])
#     for feat_name, feat_value in last_bsp.features.items():
#         if feat_name in meta:
#             feature_arr[meta[feat_name]] = feat_value
#     img_array = np.expand_dims(img_array, axis=0)
#     feature_arr = np.expand_dims(feature_arr, axis=0)
#     value = model.predict([img_array, feature_arr], verbose=False)[0][0]
#     print(value)
#     return value


def strategy(code, lv_list, begin_date, total_profit):
    with open("./model.hdf5", 'rb') as file:
        # 使用 pickle.load 加载对象
        model = pickle.load(file)
    meta = json.load(open("./TMP/EURUSD_feature.meta", "r"))
    data_src_type = DATA_SRC.FOREX_ONLINE

    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d %H:%M:%S")
    end_date = None
    # 快照
    chan = CChan(
        code=code,
        data_src=data_src_type,
        lv_list=lv_list,
        config=config,
        begin_time=begin_date,
        end_time=end_date
    )

    capital = 10000
    lots = 0.1
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
        # print(datetime.datetime.now())
        # print(f"北京时间:{lv_chan[-1][-1].time}  on_bar {code}")
        #        assert low_chan[-1][-1].close == chan[-1][-1].close
        # print(
        #     f"北京时间:{lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(lv_chan[-1][-1].time.ts)}  on_bar {code}")
        """
        策略开始：
        这里基于chan实现你的策略
        """
        profit = 0
        bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        def sendmessage(subject, message):
            bar_time = datetime.datetime.fromtimestamp(lv_chan[-1][-1].time.ts)
            if datetime.datetime.now() - bar_time <= datetime.timedelta(minutes=5):
                send_message(app_id, app_secret, webhook_url, subject, message, [chan_snapshot])

        if long_order > 0:
            # 止盈
            close_price = round(lv_chan[-1][-1].close, 5)
            long_profit = close_price / long_order - 1
            exit_rule = last_bsp.klu.klc.idx == lv_chan[-2].idx and not last_bsp.is_buy and \
                        (BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type)
            # 最大止损保护
            tp = long_profit > 0.01
            sl = long_profit < -0.003
            if tp or sl or exit_rule:
                long_order = 0
                profit += round(long_profit * money, 2)
                subject = f'{code} {lv_chan[-1][-1].time}:sell price = {close_price}, profit = {long_profit * money:.2f}'
                print(subject)
                history_long_orders += 1
                bar_time = datetime.datetime.fromtimestamp(lv_chan[-1][-1].time.ts)
                if datetime.datetime.now() - bar_time <= datetime.timedelta(minutes=5):
                    close_order(symbol=code, comment=f"tp:{tp},sl:{sl},exit_rule:{exit_rule}")
                    message = f"北京时间:{lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(lv_chan[-1][-1].time.ts)}"
                    sendmessage(subject, message)
        if short_order > 0:
            close_price = round(lv_chan[-1][-1].close, 5)
            short_profit = short_order / close_price - 1
            exit_rule = last_bsp.klu.klc.idx == lv_chan[-2].idx and last_bsp.is_buy and \
                        (BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type)
            # 最大止损保护
            tp = short_profit > 0.1
            sl = short_profit < -0.003
            if tp or sl or exit_rule:
                short_order = 0
                profit += round(short_profit * money, 2)
                subject = f'{code} {lv_chan[-1][-1].time}:sell price = {close_price}, profit = {short_profit * money:.2f}'
                print(subject)
                history_short_orders += 1
                bar_time = datetime.datetime.fromtimestamp(lv_chan[-1][-1].time.ts)
                if datetime.datetime.now() - bar_time <= datetime.timedelta(minutes=5):
                    close_order(symbol=code, comment=f"tp:{tp},sl:{sl},exit_rule:{exit_rule}")
                    message = f"北京时间:{lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(lv_chan[-1][-1].time.ts)}"
                    sendmessage(subject, message)

        if long_order == 0 and short_order == 0:
            # 共振且同向
            if last_bsp.klu.klc.idx == lv_chan[-1].idx and last_bsp.is_buy and \
                    (BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
                factors = FeatureFactors(chan[0]).get_factors()
                for key in factors.keys():
                    last_bsp.features.add_feat(key, factors[key])
                value = predict_bsp(model, last_bsp=last_bsp, meta=meta)
                if value > 0.7:
                    long_order = round(lv_chan[-1][-1].close * fee, 5)
                    subject = f'{code} {lv_chan[-1][-1].time}:buy long price = {long_order}'
                    print(subject)
                    bar_time = datetime.datetime.fromtimestamp(lv_chan[-1][-1].time.ts)
                    if datetime.datetime.now() - bar_time <= datetime.timedelta(minutes=5):
                        open_order(code, lot=0.01, is_buy=True, comment=",".join([t.value for t in last_bsp.type]))
                        message = f"北京时间:{lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(lv_chan[-1][-1].time.ts)}"
                        sendmessage(subject, message)

        if short_order == 0 and long_order == 0:
            # 共振且同向
            if last_bsp.klu.klc.idx == lv_chan[-1].idx and not last_bsp.is_buy and \
                    (BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
                factors = FeatureFactors(chan[0]).get_factors()
                for key in factors.keys():
                    last_bsp.features.add_feat(key, factors[key])
                value = predict_bsp(model, last_bsp=last_bsp, meta=meta)
                if value > 0.7:
                    short_order = round(lv_chan[-1][-1].close / fee, 5)
                    subject = f'{code} {lv_chan[-1][-1].time}:buy short price = {short_order}'
                    print(subject)
                    bar_time = datetime.datetime.fromtimestamp(lv_chan[-1][-1].time.ts)
                    if datetime.datetime.now() - bar_time <= datetime.timedelta(minutes=5):
                        open_order(code, lot=0.01, is_buy=False, comment=",".join([t.value for t in last_bsp.type]))
                        message = f"北京时间:{lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(lv_chan[-1][-1].time.ts)}"
                        bar_time = datetime.datetime.fromtimestamp(lv_chan[-1][-1].time.ts)
                        if datetime.datetime.now() - bar_time <= datetime.timedelta(minutes=5):
                            sendmessage(subject, message)

        capital += profit
        if profit != 0:
            profits = np.append(profits, profit)
            with total_profit.get_lock():  # 使用锁来确保操作的原子性
                total_profit.value += profit

            capitals = np.append(capitals, capital)
            print(f"{code} capital:{capital}")
            win_rate = len(profits[profits > 0]) / len(profits) * 100
            print(f"{code} 胜率: {win_rate}")
            print(f"盈利总计:{total_profit.value}")


if __name__ == "__main__":
    symbols = [
        # Major
        "EURUSD",
        "GBPUSD",
        "AUDUSD",
        "NZDUSD",
        "USDJPY",
        "USDCAD",
        "USDCHF",
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
    ]
    # load_keras_model(symbols, is_all_in_one=True)
    lv_list = [KL_TYPE.K_30M]
    begin_date = "2021-01-01 00:00:00"
    total_profit = Value('f', 0)
    model = None
    tasks = []
    for symbol in symbols:
        task = Process(target=strategy, args=(symbol, lv_list, begin_date, total_profit))
        tasks.append(task)
        task.start()
    for task in tasks:
        task.join()
