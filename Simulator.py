# cython: language_level=3
# encoding:utf-8
import os
import pickle

from BuySellPoint.BS_Point import CBS_Point
from ChanConfig import CChanConfig
from FeatureEngineering import FeatureFactors
from Messenger import _send_message

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
app_id = 'cli_a63ae160.759d500b'
app_secret = 'BvtLvfCEPEePrqdw4vddScwhKVWSCtAx'
webhook_url = 'https://open.feishu.cn/open-apis/bot/v2/hook/d8ea8601-259b-4f48-a310-1e715909232e'

config = CChanConfig({
    "trigger_step": True,  # 打开开关！
    "bi_strict": True,
    "skip_step": 500,
    "divergence_rate": float("inf"),
    "bsp2_follow_1": False,
    "bsp3_follow_1": False,
    "min_zs_cnt": 0,
    "bs1_peak": True,
    "macd_algo": "peak",
    "bs_type": '1,1p',
    "print_warning": True,
    "zs_algo": "normal",
    "cal_rsi": True,
    "cal_kdj": True,
    "cal_demark": False,
    "kl_data_check": False
})


def strategy(code, lv_list, begin_date, total_profit):
    with open(f"./TMP/all_in_one_1_1p_1-1p_model.hdf5", 'rb') as file:
        # 使用 pickle.load 加载对象
        lightgbm_model = pickle.load(file)
    with open(f"./TMP/all_in_one_1_1p_1-1p_model.meta", "rb") as fid:
        feature_names = pickle.load(fid)

    def predict_bsp(last_bsp: CBS_Point):
        missing = float('nan')
        feature_arr = [missing] * len(feature_names)
        for feat_name, feat_value in last_bsp.features.items():
            i = feature_names.index(feat_name)
            if i >= 0:
                feature_arr[i] = feat_value
        feature_arr = [feature_arr]
        return lightgbm_model.predict_proba(feature_arr)[0][1]

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
    bi_idx = 0
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

        def send_message(subject, message):
            bar_time = datetime.datetime.fromtimestamp(lv_chan[-1][-1].time.ts)
            if datetime.datetime.now() - bar_time <= datetime.timedelta(minutes=5):
                _send_message(app_id, app_secret, webhook_url, subject, message, [chan_snapshot])

        if last_bsp.klu.klc.idx == lv_chan[-1].idx and (BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
            factors = FeatureFactors(chan[0]).get_factors()
            for key in factors.keys():
                last_bsp.features.add_feat(key, factors[key])
            bsp1_pred = predict_bsp(last_bsp=last_bsp)
        else:
            bsp1_pred = 0.0
        if long_order > 0:
            # 止盈
            close_price = round(lv_chan[-1][-1].close, 5)
            long_profit = close_price / long_order - 1
            exit_rule = bsp1_pred > 0.75 and not last_bsp.is_buy
            # 最大止盈止损保护
            tp = long_profit > 0.03
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
                    send_message(subject, message)
        if short_order > 0:
            close_price = round(lv_chan[-1][-1].close, 5)
            short_profit = short_order / close_price - 1
            exit_rule = bsp1_pred > 0.75 and last_bsp.is_buy
            # 最大止盈止损保护
            tp = short_profit > 0.03
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
                    send_message(subject, message)

        if long_order == 0 and short_order == 0:
            if bsp1_pred > 0.75 and last_bsp.is_buy:
                long_order = round(lv_chan[-1][-1].close * fee, 5)
                subject = f'{code} {lv_chan[-1][-1].time}:buy long price = {long_order}'
                bi_idx = lv_chan.bi_list[-1].idx
                print(subject)
                bar_time = datetime.datetime.fromtimestamp(lv_chan[-1][-1].time.ts)
                if datetime.datetime.now() - bar_time <= datetime.timedelta(minutes=5):
                    open_order(code, lot=0.01, is_buy=True, comment=",".join([t.value for t in last_bsp.type]))
                    message = f"北京时间:{lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(lv_chan[-1][-1].time.ts)}"
                    send_message(subject, message)
            if bsp1_pred > 0.75 and not last_bsp.is_buy:
                short_order = round(lv_chan[-1][-1].close / fee, 5)
                subject = f'{code} {lv_chan[-1][-1].time}:buy short price = {short_order}'
                bi_idx = lv_chan.bi_list[-1].idx
                print(subject)
                bar_time = datetime.datetime.fromtimestamp(lv_chan[-1][-1].time.ts)
                if datetime.datetime.now() - bar_time <= datetime.timedelta(minutes=5):
                    open_order(code, lot=0.01, is_buy=False, comment=",".join([t.value for t in last_bsp.type]))
                    message = f"北京时间:{lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(lv_chan[-1][-1].time.ts)}"
                    bar_time = datetime.datetime.fromtimestamp(lv_chan[-1][-1].time.ts)
                    if datetime.datetime.now() - bar_time <= datetime.timedelta(minutes=5):
                        send_message(subject, message)

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
