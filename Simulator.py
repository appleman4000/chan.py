# cython: language_level=3
# encoding:utf-8
import datetime
import sys
from multiprocessing import Process
from multiprocessing import Value

import numpy as np

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, KL_TYPE
from CommonTools import period_seconds, shanghai_to_zurich_datetime
from Messenger import send_message

sys.setrecursionlimit(1000000)
app_id = 'cli_a63ae160c79d500b'
app_secret = 'BvtLvfCEPEePrqdw4vddScwhKVWSCtAx'
webhook_url = 'https://open.feishu.cn/open-apis/bot/v2/hook/d8ea8601-259b-4f48-a310-1e715909232e'
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
    # "USDCNH",
    # "XAUUSD",
    # "XAGUSD",
]


def strategy(code, global_profit):
    data_src_type = DATA_SRC.FOREX_ONLINE
    bottom_kl_type = KL_TYPE.K_3M
    top_kl_type = KL_TYPE.K_30M
    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "skip_step": 500,
        "divergence_rate": float("inf"),
        "min_zs_cnt": 1,
        "macd_algo": "slope",
        "kl_data_check": False,
        "bi_end_is_peak": True,
        "bsp1_only_multibi_zs": True,
        "max_bs2_rate": 0.999,
        "bs1_peak": True,
        "bs_type": "1,1p,2,2s,3a,3b",
        "bsp2_follow_1": True,
        "bsp3_follow_1": True,
        "bsp3_peak": True,
        "bsp2s_follow_2": True,
        "max_bsp2s_lv": None,
        "strict_bsp3": True,
    })
    begin_date = datetime.datetime(year=2021, month=1, day=1, hour=1, minute=0, second=0)
    end_date = datetime.datetime.now()
    end_date = end_date.timestamp()
    end_date -= end_date % period_seconds[top_kl_type]
    end_date -= period_seconds[top_kl_type]
    end_date = datetime.datetime.fromtimestamp(end_date)
    # 快照
    chan = CChan(
        code=code,
        data_src=data_src_type,
        lv_list=[top_kl_type, bottom_kl_type],
        config=config,
        begin_time=begin_date,
        end_time=end_date
    )

    capital = 10000
    lots = 0.1
    money = 100000 * lots
    capitals = np.array([])
    profits = np.array([])
    fee = 1.0004
    long_order = 0
    short_order = 0
    history_long_orders = 0
    history_short_orders = 0
    for chan_snapshot in chan.step_load():

        top_lv_chan = chan_snapshot[0]
        bottom_lv_chan = chan_snapshot[1]
        # print(datetime.datetime.now())
        # print(
        #     f"北京时间:{top_lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(top_lv_chan[-1][-1].time.ts)}  on_bar {code}")
        # print(
        #     f"北京时间:{middle_lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(middle_lv_chan[-1][-1].time.ts)}  on_bar {code}")
        # print(
        #     f"北京时间:{bottom_lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(bottom_lv_chan[-1][-1].time.ts)}  on_bar {code}")
        """
        策略开始：
        这里基于chan实现你的策略
        """
        profit = 0
        top_bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not top_bsp_list:
            continue
        top_last_bsp = top_bsp_list[-1]
        bottom_bsp_list = chan.get_bsp(1)  # 获取买卖点列表
        if not bottom_bsp_list:
            continue
        bottom_last_bsp = bottom_bsp_list[-1]
        top_entry_rule = top_lv_chan[-1].idx == top_last_bsp.klu.klc.idx
        botton_entry_rule = bottom_lv_chan[-1].idx == bottom_last_bsp.klu.klc.idx

        if long_order > 0:
            # 止盈
            close_price = round(bottom_lv_chan[-1][-1].close / fee, 5)
            long_profit = close_price / long_order - 1
            tp = long_profit >= 0.02
            sl = long_profit <= -0.01
            if tp or sl or top_entry_rule and not top_last_bsp.is_buy and botton_entry_rule and not bottom_last_bsp.is_buy:
                long_order = 0
                profit += round(long_profit * money, 2)
                print(
                    f'{code} {bottom_lv_chan[-1][-1].time}:sell price = {close_price}, profit = {long_profit * money:.2f}')
                history_long_orders += 1

        if short_order > 0:
            close_price = round(bottom_lv_chan[-1][-1].close * fee, 5)
            short_profit = short_order / close_price - 1
            tp = short_profit >= 0.02
            sl = short_profit <= -0.01
            if tp or sl or top_entry_rule and top_last_bsp.is_buy and botton_entry_rule and bottom_last_bsp.is_buy:
                short_profit = short_order / close_price - 1
                short_order = 0
                profit += round(short_profit * money, 2)
                print(
                    f'{code} {bottom_lv_chan[-1][-1].time}:sell price = {close_price}, profit = {short_profit * money:.2f}')
                history_short_orders += 1

        if long_order == 0 and short_order == 0:
            if top_entry_rule and botton_entry_rule and top_last_bsp.is_buy and bottom_last_bsp.is_buy:
                long_order = round(bottom_lv_chan[-1][-1].close * fee, 5)
                print(f'{code} {bottom_lv_chan[-1][-1].time}:buy long price = {long_order}')
        if short_order == 0 and long_order == 0:
            if top_entry_rule and botton_entry_rule and not top_last_bsp.is_buy and not bottom_last_bsp.is_buy:
                short_order = round(bottom_lv_chan[-1][-1].close / fee, 5)
                print(f'{code} {bottom_lv_chan[-1][-1].time}:buy short price = {short_order}')
        # 发送买卖点信号
        if top_entry_rule and botton_entry_rule and (top_last_bsp.is_buy and bottom_last_bsp.is_buy or (
                not top_last_bsp.is_buy and not bottom_last_bsp.is_buy)):
            price = f"{bottom_lv_chan[-1][-1].close:.5f}".rstrip('0').rstrip('.')
            subject = f"{code} {'买点' if top_last_bsp.is_buy else '卖点'}:{price}"
            message = f"北京时间:{bottom_lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(bottom_lv_chan[-1][-1].time.ts)}"
            bar_time = datetime.datetime.fromtimestamp(bottom_lv_chan[-1][-1].time.ts)
            if datetime.datetime.now() - bar_time < datetime.timedelta(hours=1):
                send_message(app_id, app_secret, webhook_url, subject, message, [chan])

        capital += profit
        if profit != 0:
            profits = np.append(profits, profit)
            with global_profit.get_lock():  # 使用锁来确保操作的原子性
                global_profit.value += profit

            capitals = np.append(capitals, capital)
            print(f"{code} capital:{capital}")
            win_rate = len(profits[profits > 0]) / len(profits) * 100
            print(f"{code} 胜率: {win_rate}")
            print(f"盈利总计:{global_profit.value}")


if __name__ == "__main__":
    global_profit = Value('f', 0)
    processes = []
    for symbol in symbols:
        process = Process(target=strategy, args=(symbol, global_profit))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
