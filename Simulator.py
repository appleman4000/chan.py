# cython: language_level=3
# encoding:utf-8
import os
from threading import Thread

from PIL import Image

os.environ['KERAS_BACKEND'] = 'torch'
import keras

import datetime
import io
import sys
from multiprocessing import Value

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, KL_TYPE, BSP_TYPE
from CommonTools import period_seconds, shanghai_to_zurich_datetime
from Messenger import send_message
from Plot.PlotDriver import CPlotDriver

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


def get_predict_value(model, chan, plot_config, plot_para):
    matplotlib.use('Agg')
    g = CPlotDriver(chan, plot_config, plot_para)
    # 移除标题
    for ax in g.figure.axes:
        ax.set_title("", loc="left")
        # 移除 x 轴和 y 轴标签
        ax.set_xlabel('')
        ax.set_ylabel('')

        # 移除 x 轴和 y 轴的刻度标签
        ax.set_xticks([])
        ax.set_yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    g.figure.tight_layout()
    buf = io.BytesIO()
    g.figure.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    plt.close(g.figure)
    img = Image.open(buf).resize((224, 224))
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    # 将图片转换为 NumPy 数组
    img_array = np.array(img)
    outputs = model.predict(np.expand_dims(img_array, axis=0), verbose=False)[0]
    return outputs


def strategy(code, global_profit):
    data_src_type = DATA_SRC.FOREX_ONLINE
    kl_type = KL_TYPE.K_30M
    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "skip_step": 500,
        "divergence_rate": float("inf"),
        "min_zs_cnt": 1,
        "macd_algo": "slope",
        "kl_data_check": False,
        "bs_type": "1,1p,2,2s,3a,3b",
    })
    plot_config = {
        "plot_kline": False,
        "plot_kline_combine": True,
        "plot_bi": True,
        "plot_seg": False,
        "plot_eigen": False,
        "plot_zs": True,
        "plot_macd": False,
        "plot_mean": False,
        "plot_channel": False,
        "plot_bsp": False,
        "plot_extrainfo": False,
        "plot_demark": False,
        "plot_marker": False,
        "plot_rsi": False,
        "plot_kdj": False,
    }

    plot_para = {
        "figure": {
            "w": 224 / 50,
            "h": 224 / 50,
            "x_range": 90,
        },
        "seg": {
            # "plot_trendline": True,
            "disp_end": False,
            "end_fontsize": 15,
            "width": 0.5
        },
        "bi": {
            "show_num": False,
            "disp_end": False,
            "end_fontsize": 15,
        },
        "zs": {
            "fontsize": 15,
        },
        "bsp": {
            "fontsize": 20
        },
        "segseg": {
            "end_fontsize": 15,
            "width": 0.5
        },
        "seg_bsp": {
            "fontsize": 20
        },
        "marker": {
            # "markers": {  # text, position, color
            #     '2023/06/01': ('marker here', 'up', 'red'),
            #     '2023/06/08': ('marker here', 'down')
            # },
        }
    }
    begin_date = datetime.datetime(year=2021, month=1, day=1, hour=1, minute=0, second=0)
    end_date = datetime.datetime.now()
    end_date = end_date.timestamp()
    end_date -= end_date % period_seconds[kl_type]
    end_date -= period_seconds[kl_type]
    end_date = datetime.datetime.fromtimestamp(end_date)
    # 快照
    chan = CChan(
        code=code,
        data_src=data_src_type,
        lv_list=[kl_type],
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
    model = keras.saving.load_model("./Debug/model.keras")
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
            tp = long_profit >= 0.004
            sl = long_profit <= -0.004
            if tp or sl or entry_rule and not last_bsp.is_buy:
                long_order = 0
                profit += round(long_profit * money, 2)
                print(
                    f'{code} {lv_chan[-1][-1].time}:sell price = {close_price}, profit = {long_profit * money:.2f}')
                history_long_orders += 1

        if short_order > 0:
            close_price = round(lv_chan[-1][-1].close * fee, 5)
            short_profit = short_order / close_price - 1
            tp = short_profit >= 0.004
            sl = short_profit <= -0.004
            if tp or sl or entry_rule and last_bsp.is_buy:
                short_profit = short_order / close_price - 1
                short_order = 0
                profit += round(short_profit * money, 2)
                print(
                    f'{code} {lv_chan[-1][-1].time}:sell price = {close_price}, profit = {short_profit * money:.2f}')
                history_short_orders += 1

        if long_order == 0 and short_order == 0:
            if entry_rule and last_bsp.is_buy and (BSP_TYPE.T2 in last_bsp.type or BSP_TYPE.T2S in last_bsp.type):
                value = get_predict_value(model, chan, plot_config, plot_para)
                if value > 0.55:
                    long_order = round(lv_chan[-1][-1].close * fee, 5)
                    print(f'{code} {lv_chan[-1][-1].time}:buy long price = {long_order}')
        if short_order == 0 and long_order == 0:
            if entry_rule and not last_bsp.is_buy and (BSP_TYPE.T2 in last_bsp.type or BSP_TYPE.T2S in last_bsp.type):
                value = get_predict_value(model, chan, plot_config, plot_para)
                if value < 0.45:
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
            with global_profit.get_lock():  # 使用锁来确保操作的原子性
                global_profit.value += profit

            capitals = np.append(capitals, capital)
            print(f"{code} capital:{capital}")
            win_rate = len(profits[profits > 0]) / len(profits) * 100
            print(f"{code} 胜率: {win_rate}")
            print(f"盈利总计:{global_profit.value}")


if __name__ == "__main__":
    global_profit = Value('f', 0)
    threads = []
    for symbol in symbols:
        thread = Thread(target=strategy, args=(symbol, global_profit))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
