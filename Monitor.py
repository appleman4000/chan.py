# cython: language_level=3
# encoding:utf-8
import builtins
import datetime
import time

import MetaTrader5 as mt5
import pandas as pd
import pytz

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, AUTYPE, KL_TYPE, BSP_TYPE
from DataAPI.MT5ForexAPI import GetColumnNameFromFieldList, create_item_dict
from KLine.KLine_Unit import CKLine_Unit
from Mail_Tools import send_email

timeframe_seconds = {
    mt5.TIMEFRAME_M1: 60,
    mt5.TIMEFRAME_M2: 120,
    mt5.TIMEFRAME_M3: 180,
    mt5.TIMEFRAME_M4: 240,
    mt5.TIMEFRAME_M5: 300,
    mt5.TIMEFRAME_M6: 360,
    mt5.TIMEFRAME_M10: 600,
    mt5.TIMEFRAME_M12: 720,
    mt5.TIMEFRAME_M15: 900,
    mt5.TIMEFRAME_M20: 1200,
    mt5.TIMEFRAME_M30: 1800,
    mt5.TIMEFRAME_H1: 3600,
    mt5.TIMEFRAME_H2: 7200,
    mt5.TIMEFRAME_H3: 10800,
    mt5.TIMEFRAME_H4: 14400,
    mt5.TIMEFRAME_H6: 21600,
    mt5.TIMEFRAME_H8: 28800,
    mt5.TIMEFRAME_H12: 43200,
    mt5.TIMEFRAME_D1: 86400,
    mt5.TIMEFRAME_W1: 604800,
    mt5.TIMEFRAME_MN1: 2592000  # 大约的月时间秒数，具体月的秒数会有所不同
}
period_map = {
    mt5.TIMEFRAME_M1: KL_TYPE.K_1M,
    mt5.TIMEFRAME_M3: KL_TYPE.K_3M,
    mt5.TIMEFRAME_M5: KL_TYPE.K_15M,
    mt5.TIMEFRAME_M15: KL_TYPE.K_15M,
    mt5.TIMEFRAME_M30: KL_TYPE.K_30M,
    mt5.TIMEFRAME_H1: KL_TYPE.K_60M,
    mt5.TIMEFRAME_H4: KL_TYPE.K_240M,
    mt5.TIMEFRAME_D1: KL_TYPE.K_DAY,

}
period_name = {
    mt5.TIMEFRAME_M1: "1分钟",
    mt5.TIMEFRAME_M3: "3分钟",
    mt5.TIMEFRAME_M5: "5分钟",
    mt5.TIMEFRAME_M15: "15分钟",
    mt5.TIMEFRAME_M30: "30分钟",
    mt5.TIMEFRAME_H1: "1小时",
    mt5.TIMEFRAME_H4: "4小时",
    mt5.TIMEFRAME_D1: "1天",

}
# 设置交易对
symbols = ["EURUSD", "USDJPY", "USDCNH", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP"]
periods = [mt5.TIMEFRAME_H4, mt5.TIMEFRAME_H1]
to_emails = ['appleman4000@qq.com', 'xubin.njupt@foxmail.com', '375961433@qq.com', 'idbeny@163.com', 'jflzhao@163.com',
             '837801694@qq.com', '1169006942@qq.com', 'vincent1122@126.com']
# to_emails = ['appleman4000@qq.com']


def shanghai_to_zurich_datetime(timestamp):
    zurich_tz = pytz.timezone('Europe/Zurich')
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    timestamp = shanghai_tz.localize(datetime.datetime.fromtimestamp(timestamp))

    # 将时间戳转换到 'Asia/Shanghai' 时区
    zurich_time = timestamp.astimezone(zurich_tz)

    # 格式化时间
    formatted_time = zurich_time.strftime('%Y-%m-%d %H:%M')

    return formatted_time


def period_seconds(period):
    return timeframe_seconds[period]


def on_bar(symbol, period, bar, enable_send_mail=False):
    print(
        f"北京时间:{bar.time} 瑞士时间:{shanghai_to_zurich_datetime(bar.time.ts)}  on_bar {symbol} {period_name[period]}")
    chan = chans[symbol + str(period)]
    chan.trigger_load({period_map[period]: [bar]})
    bsp_list = chan.get_bsp(0)
    if not bsp_list:
        return
    last_bsp = bsp_list[-1]
    if BSP_TYPE.T1 not in last_bsp.type and BSP_TYPE.T1P not in last_bsp.type and BSP_TYPE.T2 not in last_bsp.type and \
            BSP_TYPE.T2S not in last_bsp.type and BSP_TYPE.T3A not in last_bsp.type and BSP_TYPE.T3B not in last_bsp.type:
        return
    if last_bsp.klu.time != bar.time:
        return

    if enable_send_mail:
        subject = f"外汇- {symbol} {period_name[period]} {' '.join([t.name for t in last_bsp.type])} {'买点' if last_bsp.is_buy else '卖点'} {bar.close}"
        message = f"北京时间:{bar.time} 瑞士时间:{shanghai_to_zurich_datetime(bar.time.ts)}"
        send_email(to_emails, subject, message, chan)


def init_chan():
    local_tz = pytz.timezone('Asia/Shanghai')
    for symbol in symbols:
        data_src = DATA_SRC.FOREX
        for period in periods:
            lv_list = [period_map[period]]
            config = CChanConfig({
                "trigger_step": True,  # 打开开关！
                "min_zs_cnt": 1,
                "kl_data_check": False,
            })
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
            bars = mt5.copy_rates_from_pos(symbol, period, 1, 200)
            bars = pd.DataFrame(bars)
            last_bar_time = bars.iloc[-1].time
            bars.dropna(inplace=True)
            bars['time'] = pd.to_datetime(bars['time'], unit='s')
            bars['time'] = bars['time'].dt.tz_localize('Europe/Zurich')
            bars['time'] = bars['time'].dt.tz_convert(local_tz)
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
                on_bar(symbol, period, bar, enable_send_mail=False)
            next_bar_open[symbol + str(period)] = last_bar_time
            next_bar_open[symbol + str(period)] -= next_bar_open[symbol + str(period)] % period_seconds(period)
            next_bar_open[symbol + str(period)] += period_seconds(period)


def initialize_mt5():
    if not mt5.initialize():
        print("初始化失败")
        return False
    return True


def reconnect_mt5(retry_interval=5, max_retries=5):
    retries = 0
    while retries < max_retries:
        print(f"尝试重新连接 MT5 (第 {retries + 1} 次尝试)...")
        if initialize_mt5():
            print("重新连接 MT5 成功")
            return True
        retries += 1
        time.sleep(retry_interval)
    print("重新连接 MT5 失败")
    return False


def get_latest_bar(symbol):
    try:
        bar = mt5.copy_rates_from_pos(symbol, period, 1, 1)[0]
        if bar is None:
            raise ValueError(f"无法获取 {symbol} 的价格信息")
        return bar
    except Exception as e:
        print(f"获取 {symbol} 的价格信息时发生异常: {e}")
        return None


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
    local_tz = pytz.timezone('Asia/Shanghai')
    try:
        while True:
            for symbol in symbols:
                for period in periods:
                    bar = get_latest_bar(symbol)
                    if bar is None:
                        continue
                    last_bar_time = bar[0]
                    if last_bar_time >= next_bar_open[symbol + str(period)]:
                        bars = mt5.copy_rates_range(symbol, period, datetime.datetime.fromtimestamp(
                            next_bar_open[symbol + str(period)]), datetime.datetime.fromtimestamp(last_bar_time))
                        next_bar_open[symbol + str(period)] = last_bar_time
                        next_bar_open[symbol + str(period)] -= next_bar_open[symbol + str(period)] % period_seconds(
                            period)
                        next_bar_open[symbol + str(period)] += period_seconds(period)
                        bars = pd.DataFrame(bars)
                        bars.dropna(inplace=True)
                        bars['time'] = pd.to_datetime(bars['time'], unit='s')
                        bars['time'] = bars['time'].dt.tz_localize('Europe/Zurich')
                        bars['time'] = bars['time'].dt.tz_convert(local_tz)
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
                            on_bar(symbol, period, bar, enable_send_mail=True)
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
