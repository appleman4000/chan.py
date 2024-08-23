# cython: language_level=3
# encoding:utf-8
import os

from ChanConfig import CChanConfig
from Messenger import send_message

os.environ['KERAS_BACKEND'] = 'torch'
from threading import Thread

# import keras
import datetime
import sys
from multiprocessing import Value

import numpy as np

from Chan import CChan
from Common.CEnum import DATA_SRC, KL_TYPE, BSP_TYPE
from CommonTools import shanghai_to_zurich_datetime, robot_trade

sys.setrecursionlimit(1000000)
app_id = 'cli_a63ae160c79d500b'
app_secret = 'BvtLvfCEPEePrqdw4vddScwhKVWSCtAx'
webhook_url = 'https://open.feishu.cn/open-apis/bot/v2/hook/d8ea8601-259b-4f48-a310-1e715909232e'
keras_model = {}
lgb_model = {}
meta = {}

config = CChanConfig({
    "trigger_step": True,  # 打开开关！
    "bi_strict": False,
    "skip_step": 0,
    "divergence_rate": 1.0,
    "bsp2_follow_1": True,
    "bsp3_follow_1": True,
    "min_zs_cnt": 1,
    "bs1_peak": True,
    "macd_algo": "slope",
    "bs_type": '1,2,3a,1p,2s,3b',
    "print_warning": True,
    "zs_algo": "normal",
    "cal_rsi": False,
    "cal_kdj": False,
    "cal_demark": False,
    "kl_data_check": False
})


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
    fee = 1.0002
    long_order = 0
    short_order = 0
    history_long_orders = 0
    history_short_orders = 0
    for chan_snapshot in chan.step_load():

        high_chan = chan_snapshot[0]
        lv_chan = chan_snapshot[1]
        # print(datetime.datetime.now())
        # print(f"北京时间:{lv_chan[-1][-1].time}  on_bar {code}")
        assert lv_chan[-1][-1].close == high_chan[-1][-1].close
        # print(
        #     f"北京时间:{lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(lv_chan[-1][-1].time.ts)}  on_bar {code}")
        """
        策略开始：
        这里基于chan实现你的策略
        """
        profit = 0
        bsp_list = chan.get_bsp(1)  # 获取买卖点列表
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        high_bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not high_bsp_list:
            continue
        high_last_bsp = high_bsp_list[-1]

        cycler_esonance = lv_chan[-1].idx - last_bsp.klu.klc.idx == 0
        cycler_esonance = cycler_esonance and high_chan[-1].idx - high_last_bsp.klu.klc.idx == 0

        if long_order > 0:
            # 止盈
            close_price = round(lv_chan[-1][-1].close / fee, 5)
            long_profit = close_price / long_order - 1
            tp = long_profit >= 0.002
            sl = long_profit <= -0.002
            if tp or sl:
                long_order = 0
                profit += round(long_profit * money, 2)
                print(
                    f'{code} {lv_chan[-1][-1].time}:sell price = {close_price}, profit = {long_profit * money:.2f}')
                history_long_orders += 1
        if short_order > 0:
            close_price = round(lv_chan[-1][-1].close * fee, 5)
            short_profit = short_order / close_price - 1
            tp = short_profit >= 0.002
            sl = short_profit <= -0.002
            if tp or sl:
                short_profit = short_order / close_price - 1
                short_order = 0
                profit += round(short_profit * money, 2)
                print(
                    f'{code} {lv_chan[-1][-1].time}:sell price = {close_price}, profit = {short_profit * money:.2f}')
                history_short_orders += 1

        if long_order == 0 and short_order == 0:
            if cycler_esonance and last_bsp.is_buy and (
                    BSP_TYPE.T2 in last_bsp.type or BSP_TYPE.T2S in last_bsp.type) and \
                    high_last_bsp.is_buy and (
                    BSP_TYPE.T2 in high_last_bsp.type or BSP_TYPE.T2S in high_last_bsp.type):
                long_order = round(lv_chan[-1][-1].close * fee, 5)
                print(f'{code} {lv_chan[-1][-1].time}:buy long price = {long_order}')

        if short_order == 0 and long_order == 0:
            if cycler_esonance and not last_bsp.is_buy and (
                    BSP_TYPE.T2 in last_bsp.type or BSP_TYPE.T2S in last_bsp.type) and \
                    not high_last_bsp.is_buy and (
                    BSP_TYPE.T2 in high_last_bsp.type or BSP_TYPE.T2S in high_last_bsp.type):
                short_order = round(lv_chan[-1][-1].close / fee, 5)
                print(f'{code} {lv_chan[-1][-1].time}:buy short price = {short_order}')

        # 发送买卖点信号
        if cycler_esonance and (
                last_bsp.is_buy and high_last_bsp.is_buy or not last_bsp.is_buy and not high_last_bsp.is_buy) and \
                (BSP_TYPE.T2 in last_bsp.type or BSP_TYPE.T2S in last_bsp.type):
            bar_time = datetime.datetime.fromtimestamp(lv_chan[-1][-1].time.ts)
            if datetime.datetime.now() - bar_time <= datetime.timedelta(minutes=5):
                price = f"{lv_chan[-1][-1].close:.5f}".rstrip('0').rstrip('.')
                subject = f"{code} {'买点' if last_bsp.is_buy else '卖点'}:{price}"
                message = f"北京时间:{lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(lv_chan[-1][-1].time.ts)}"
                send_message(app_id, app_secret, webhook_url, subject, message, [chan])
                comment = f"{','.join([t.name for t in last_bsp.type])}"
                robot_trade(symbol, 0.01, last_bsp.is_buy, comment)

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
        # "USDCNH",
        # "XAUUSD",
        # "XAGUSD",
    ]
    # load_keras_model(symbols, is_all_in_one=True)
    lv_list = [KL_TYPE.K_5M, KL_TYPE.K_1M]
    begin_date = "2021-08-22 00:00:00"
    total_profit = Value('f', 0)
    model = None
    threads = []
    for symbol in symbols:
        thread = Thread(target=strategy, args=(symbol, lv_list, begin_date, total_profit))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
