# cython: language_level=3
# encoding:utf-8
import json
import os
import pickle

from xgboost import DMatrix

from BuySellPoint.BS_Point import CBS_Point

os.environ['KERAS_BACKEND'] = 'torch'
from threading import Thread

import keras

from GenerateDataset import plot_config, plot_para

import datetime
import sys
from multiprocessing import Value

import numpy as np

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, KL_TYPE, BSP_TYPE
from CommonTools import period_seconds, shanghai_to_zurich_datetime, chan_to_png
from Messenger import send_message

sys.setrecursionlimit(1000000)
app_id = 'cli_a63ae160c79d500b'
app_secret = 'BvtLvfCEPEePrqdw4vddScwhKVWSCtAx'
webhook_url = 'https://open.feishu.cn/open-apis/bot/v2/hook/d8ea8601-259b-4f48-a310-1e715909232e'
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
    # "XAUUSD",
    # "XAGUSD",
]

keras_model = {}
lgb_model = {}


def get_predict_value(code, chan: CChan, last_bsp: CBS_Point, plot_config, plot_para):
    if keras_model[code] is None:
        keras_model[code] = keras.saving.load_model(f"./TMP/{code}_model.keras")
    if lgb_model[code] is None:
        with open(f"./TMP/{code}_model.lgb", 'rb') as file:
            # 使用 pickle.load 加载对象
            lgb_model[code] = pickle.load(file)
            meta = json.load(open(f"./TMP/{code}_feature.meta", "r"))

    img_array = chan_to_png(chan, plot_config, plot_para, file_path="")
    intermediate_layer = keras_model[code].layers[-2]
    intermediate_model = keras.Model(inputs=keras_model[code].input,
                                     outputs=intermediate_layer.output)
    keras_features = intermediate_model.predict([img_array], verbose=False)[0]
    missing = -9999999
    feature_arr = [missing] * len(meta)
    for feat_name, feat_value in last_bsp.features.items():
        if feat_name in meta:
            feature_arr[meta[feat_name]] = feat_value
    feature_arr = [feature_arr]
    dtest = DMatrix(feature_arr, missing=missing)
    return lgb_model.predict_proba([dtest])[0][1]


def strategy(code, lv_list, begin_date, total_profit):
    data_src_type = DATA_SRC.FOREX_ONLINE
    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "skip_step": 500,
        "divergence_rate": 0.9,
        "min_zs_cnt": 1,
        "macd_algo": "area",
        "kl_data_check": False,
        "bs_type": "1,1p,2,2s,3a,3b",
    })

    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.datetime.now()
    end_date = end_date.timestamp()
    end_date -= end_date % period_seconds[lv_list[0]]
    end_date -= period_seconds[lv_list[0]]
    end_date = datetime.datetime.fromtimestamp(end_date)
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
    fee = 1.0002
    long_order = 0
    short_order = 0
    history_long_orders = 0
    history_short_orders = 0
    for chan_snapshot in chan.step_load():

        lv_chan = chan_snapshot[0]
        # print(datetime.datetime.now())
        # print(
        #     f"北京时间:{top_lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(top_lv_chan[-1][-1].time.ts)}  on_bar {code}")
        # print(
        #     f"北京时间:{middle_lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(middle_lv_chan[-1][-1].time.ts)}  on_bar {code}")
        # print(
        #     f"北京时间:{lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(lv_chan[-1][-1].time.ts)}  on_bar {code}")
        """
        策略开始：
        这里基于chan实现你的策略
        """
        profit = 0
        top_bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not top_bsp_list:
            continue
        last_bsp = top_bsp_list[-1]
        entry_rule = lv_chan[-1].idx == last_bsp.klu.klc.idx
        if long_order > 0:
            # 止盈
            close_price = round(lv_chan[-1][-1].close / fee, 5)
            long_profit = close_price / long_order - 1
            tp = long_profit >= 0.003
            sl = long_profit <= -0.005
            if tp or sl:
                long_order = 0
                profit += round(long_profit * money, 2)
                print(
                    f'{code} {lv_chan[-1][-1].time}:sell price = {close_price}, profit = {long_profit * money:.2f}')
                history_long_orders += 1

        if short_order > 0:
            close_price = round(lv_chan[-1][-1].close * fee, 5)
            short_profit = short_order / close_price - 1
            tp = short_profit >= 0.003
            sl = short_profit <= -0.005
            if tp or sl:
                short_profit = short_order / close_price - 1
                short_order = 0
                profit += round(short_profit * money, 2)
                print(
                    f'{code} {lv_chan[-1][-1].time}:sell price = {close_price}, profit = {short_profit * money:.2f}')
                history_short_orders += 1

        if long_order == 0 and short_order == 0:
            if entry_rule and last_bsp.is_buy and (BSP_TYPE.T2 in last_bsp.type or BSP_TYPE.T2S in last_bsp.type):
                value = get_predict_value(code, model, chan, plot_config, plot_para)
                if value > 0.6:
                    long_order = round(lv_chan[-1][-1].close * fee, 5)
                    print(f'{code} {lv_chan[-1][-1].time}:buy long price = {long_order}')
        if short_order == 0 and long_order == 0:
            if entry_rule and not last_bsp.is_buy and (BSP_TYPE.T2 in last_bsp.type or BSP_TYPE.T2S in last_bsp.type):
                value = get_predict_value(code, model, chan, plot_config, plot_para)
                if value > 0.6:
                    short_order = round(lv_chan[-1][-1].close / fee, 5)
                    print(f'{code} {lv_chan[-1][-1].time}:buy short price = {short_order}')
        # 发送买卖点信号
        if entry_rule and (BSP_TYPE.T2 in last_bsp.type or BSP_TYPE.T2S in last_bsp.type):
            price = f"{lv_chan[-1][-1].close:.5f}".rstrip('0').rstrip('.')
            subject = f"{code} {'买点' if last_bsp.is_buy else '卖点'}:{price}"
            message = f"北京时间:{lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(lv_chan[-1][-1].time.ts)}"
            bar_time = datetime.datetime.fromtimestamp(lv_chan[-1][-1].time.ts)
            if datetime.datetime.now() - bar_time < datetime.timedelta(hours=1):
                send_message(app_id, app_secret, webhook_url, subject, message, [chan])

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
    lv_list = [KL_TYPE.K_30M, KL_TYPE.K_5M]
    begin_date = "2021-04-01 00:00:00"
    total_profit = Value('f', 0)
    model = None
    threads = []
    for symbol in symbols:
        thread = Thread(target=strategy, args=(symbol, lv_list, begin_date, total_profit))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
