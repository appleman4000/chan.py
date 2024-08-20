# cython: language_level=3
# encoding:utf-8
import builtins
import datetime
import sys
import time

import MetaTrader5 as mt5
import pandas as pd
import pytz

from Chan import CChan
from Common.CEnum import DATA_SRC, AUTYPE, BSP_TYPE, FX_TYPE, KL_TYPE
from CommonTools import period_name, period_seconds, server_timezone, local_timezone, \
    shanghai_to_zurich_datetime, period_mt5_map
from CommonTools import robot_trade, reconnect_mt5, get_latest_bar
from DataAPI.MT5ForexAPI import GetColumnNameFromFieldList, create_item_dict
from GenerateDataset import config
from KLine.KLine_Unit import CKLine_Unit
from Messenger import send_message

sys.setrecursionlimit(10000)

# 设置交易对
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
    "USDCNH",
    # "XAUUSD",
    # "XAGUSD",
]
periods = [KL_TYPE.K_DAY, KL_TYPE.K_30M]
# to_emails = ['appleman4000@qq.com', 'xubin.njupt@foxmail.com', '375961433@qq.com', 'idbeny@163.com', 'jflzhao@163.com',
#              '837801694@qq.com', '1169006942@qq.com', 'vincent1122@126.com']
to_emails = ['appleman4000@qq.com']


def on_bar(symbol, period, bar, enable_send_message=False):
    print(
        f"北京时间:{bar.time} 瑞士时间:{shanghai_to_zurich_datetime(bar.time.ts)}  on_bar {symbol} {period_name[period]}")
    chan = chans[symbol + str(period)]
    chan.trigger_load({period: [bar]})

    if enable_send_message:
        # 确保分型已确认
        # 30M买卖点，分型待确认
        chan_m30 = chans[symbol + str(KL_TYPE.K_30M)]
        bsp_list = chan_m30.get_bsp(0)
        if not bsp_list:
            return
        last_bsp_h1 = bsp_list[-1]
        if BSP_TYPE.T1 not in last_bsp_h1.type and BSP_TYPE.T1P not in last_bsp_h1.type and \
           BSP_TYPE.T2 not in last_bsp_h1.type and BSP_TYPE.T2S not in last_bsp_h1.type:
            return
        if chan_m30[0][-1].idx - last_bsp_h1.klu.klc.idx != 0:
            return

        price = f"{bar.close:.5f}".rstrip('0').rstrip('.')
        subject = f"外汇- {symbol} {period_name[KL_TYPE.K_30M]} {','.join([t.name for t in last_bsp_h1.type])} {'买点' if last_bsp_h1.is_buy else '卖点'} {price}"
        message = f"北京时间:{datetime.datetime.fromtimestamp(bar.time.ts + period_seconds[period]).strftime('%Y-%m-%d %H:%M')} 瑞士时间:{shanghai_to_zurich_datetime(bar.time.ts + period_seconds[period])}"
        app_id = 'cli_a63ae160c79d500b'
        app_secret = 'BvtLvfCEPEePrqdw4vddScwhKVWSCtAx'
        webhook_url = 'https://open.feishu.cn/open-apis/bot/v2/hook/b5d0499b-4082-4dd3-82a5-70528e548695'

        send_message(app_id, app_secret, webhook_url, subject, message, [chans[symbol + str(p)] for p in periods])
        comment = f"{','.join([t.name for t in last_bsp_h1.type])}"
        robot_trade(symbol, 0.01, last_bsp_h1.is_buy, comment)


def init_chan():
    for symbol in symbols:
        data_src = DATA_SRC.FOREX
        for period in periods:
            lv_list = [period]
            chan = CChan(
                code=symbol,
                begin_time=None,
                end_time=None,
                data_src=data_src,
                lv_list=lv_list,
                config=config,
                autype=AUTYPE.NONE,
            )
            chans[symbol + str(period)] = chan

    for symbol in symbols:
        for period in periods:
            bars = mt5.copy_rates_from_pos(symbol, period_mt5_map[period], 1, 1000)
            bars = pd.DataFrame(bars)
            last_bar_time = bars.iloc[-1].time
            bars.dropna(inplace=True)
            bars['time'] = pd.to_datetime(bars['time'], unit='s')
            bars['time'] = bars['time'].dt.tz_localize(server_timezone)
            bars['time'] = bars['time'].dt.tz_convert(local_timezone)
            bars['time'] = bars['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            fields = "time,open,high,low,close,volume"

            for index, bar in bars.iterrows():
                data = [
                    bar["time"],
                    bar["open"],
                    bar["high"],
                    bar["low"],
                    bar["close"],
                    bar["tick_volume"],
                ]
                bar = CKLine_Unit(create_item_dict(data, GetColumnNameFromFieldList(fields)))
                on_bar(symbol, period, bar, enable_send_message=False)
            next_bar_open[symbol + str(period)] = last_bar_time
            next_bar_open[symbol + str(period)] -= next_bar_open[symbol + str(period)] % period_seconds[period]
            next_bar_open[symbol + str(period)] += period_seconds[period]


if __name__ == "__main__":
    # 自定义 print 函数
    default_print = builtins.print


    def custom_print(*args, **kwargs):
        # 输出到控制台
        default_print(*args, **kwargs)
        # 追加日志到文件
        with open('logfile.log', 'a') as log_file:
            for arg in args:
                log_file.write(str(arg) + ' ')
            log_file.write('\n')


    # 替换内置 print 函数
    builtins.print = custom_print
    # 连接到MetaTrader 5
    if not reconnect_mt5():
        exit(0)
    chans = {}
    next_bar_open = {}
    init_chan()
    local_tz = pytz.timezone(local_timezone)
    try:
        while True:
            for symbol in symbols:
                for period in periods:
                    bar = get_latest_bar(symbol, period)
                    if bar is None:
                        continue
                    last_bar_time = bar[0]
                    if last_bar_time >= next_bar_open[symbol + str(period)]:
                        bars = mt5.copy_rates_range(symbol, period_mt5_map[period], datetime.datetime.fromtimestamp(
                            next_bar_open[symbol + str(period)]), datetime.datetime.fromtimestamp(last_bar_time))
                        next_bar_open[symbol + str(period)] = last_bar_time
                        next_bar_open[symbol + str(period)] -= next_bar_open[symbol + str(period)] % period_seconds[
                            period]
                        next_bar_open[symbol + str(period)] += period_seconds[period]
                        bars = pd.DataFrame(bars)
                        bars.dropna(inplace=True)
                        bars['time'] = pd.to_datetime(bars['time'], unit='s')
                        bars['time'] = bars['time'].dt.tz_localize(server_timezone)
                        bars['time'] = bars['time'].dt.tz_convert(local_timezone)
                        bars['time'] = bars['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        fields = "time,open,high,low,close,volume"
                        for index, bar in bars.iterrows():
                            data = [
                                bar["time"],
                                bar["open"],
                                bar["high"],
                                bar["low"],
                                bar["close"],
                                bar["tick_volume"],
                            ]
                            bar = CKLine_Unit(create_item_dict(data, GetColumnNameFromFieldList(fields)))
                            on_bar(symbol, period, bar, enable_send_message=True)
            time.sleep(1)
            # 检查连接状态，如果连接丢失则尝试重新连接
            if not mt5.terminal_info():
                print("连接丢失，尝试重新连接...")
                mt5.shutdown()
                if not reconnect_mt5():
                    break
    except KeyboardInterrupt:
        print("用户中断程序")
    finally:
        mt5.shutdown()
