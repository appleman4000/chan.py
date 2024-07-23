import datetime
import time

import MetaTrader5 as MT5
import pandas as pd
import pytz

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, AUTYPE, KL_TYPE, BSP_TYPE
from DataAPI.MT5ForexAPI import GetColumnNameFromFieldList, create_item_dict
from KLine.KLine_Unit import CKLine_Unit
from Mail_Tools import send_email

timeframe_seconds = {
    MT5.TIMEFRAME_M1: 60,
    MT5.TIMEFRAME_M2: 120,
    MT5.TIMEFRAME_M3: 180,
    MT5.TIMEFRAME_M4: 240,
    MT5.TIMEFRAME_M5: 300,
    MT5.TIMEFRAME_M6: 360,
    MT5.TIMEFRAME_M10: 600,
    MT5.TIMEFRAME_M12: 720,
    MT5.TIMEFRAME_M15: 900,
    MT5.TIMEFRAME_M20: 1200,
    MT5.TIMEFRAME_M30: 1800,
    MT5.TIMEFRAME_H1: 3600,
    MT5.TIMEFRAME_H2: 7200,
    MT5.TIMEFRAME_H3: 10800,
    MT5.TIMEFRAME_H4: 14400,
    MT5.TIMEFRAME_H6: 21600,
    MT5.TIMEFRAME_H8: 28800,
    MT5.TIMEFRAME_H12: 43200,
    MT5.TIMEFRAME_D1: 86400,
    MT5.TIMEFRAME_W1: 604800,
    MT5.TIMEFRAME_MN1: 2592000  # 大约的月时间秒数，具体月的秒数会有所不同
}
period_map = {
    MT5.TIMEFRAME_M1: KL_TYPE.K_1M,
    MT5.TIMEFRAME_M3: KL_TYPE.K_3M,
    MT5.TIMEFRAME_M5: KL_TYPE.K_15M,
    MT5.TIMEFRAME_M15: KL_TYPE.K_15M,
    MT5.TIMEFRAME_M30: KL_TYPE.K_30M,
    MT5.TIMEFRAME_H1: KL_TYPE.K_60M,
    MT5.TIMEFRAME_D1: KL_TYPE.K_DAY,

}
period_name = {
    MT5.TIMEFRAME_M1: "1分钟",
    MT5.TIMEFRAME_M3: "3分钟",
    MT5.TIMEFRAME_M5: "5分钟",
    MT5.TIMEFRAME_M15: "15分钟",
    MT5.TIMEFRAME_M30: "30分钟",
    MT5.TIMEFRAME_H1: "60分钟",
    MT5.TIMEFRAME_D1: "1天",

}
# 设置交易对
symbols = ["EURUSD", "USDJPY", "USDCNH", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP"]
periods = [MT5.TIMEFRAME_D1, MT5.TIMEFRAME_H1]
enable_send_mail = False
to_emails = ['appleman4000@qq.com', 'xubin.njupt@foxmail.com', '375961433@qq.com', 'idbeny@163.com', 'jflzhao@163.com',
             '837801694@qq.com', '1169006942@qq.com']

def to_zurich_datetime(timestamp):
    # 创建一个包含 'Europe/Athens' 时区信息的时间戳
    zurich_tz = pytz.timezone('Europe/Zurich')
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    timestamp = shanghai_tz.localize(datetime.datetime.fromtimestamp(timestamp))

    # 将时间戳转换到 'Asia/Shanghai' 时区
    zurich_time = timestamp.astimezone(zurich_tz)

    # 格式化时间并包含时区信息
    formatted_time = zurich_time.strftime('%Y-%m-%d %H:%M:%S')

    return formatted_time


def to_beijing_datetime(timestamp):
    zurich_tz = pytz.timezone('Europe/Zurich')
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    timestamp = zurich_tz.localize(datetime.datetime.fromtimestamp(timestamp))

    # 将时间戳转换到 'Asia/Shanghai' 时区
    shanghai_time = timestamp.astimezone(shanghai_tz)

    # 格式化时间
    formatted_time = shanghai_time.strftime('%Y-%m-%d %H:%M:%S')

    return formatted_time


def PeriodSeconds(period):
    return timeframe_seconds[period]


def on_tick(symbol, tick):
    global next_bar_open
    for period in periods:
        current = tick.time
        if current >= next_bar_open[symbol + str(period)]:
            bars = MT5.copy_rates_range(symbol, period, next_bar_open[symbol + str(period)] - PeriodSeconds(period),
                                        current)
            bars = bars[:-1]
            next_bar_open[symbol + str(period)] = current
            next_bar_open[symbol + str(period)] -= next_bar_open[symbol + str(period)] % PeriodSeconds(period)
            next_bar_open[symbol + str(period)] += PeriodSeconds(period)
            bars = pd.DataFrame(bars)
            bars.dropna(inplace=True)
            bars['time'] = pd.to_datetime(bars['time'], unit='s')
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
                on_bar(symbol, period, bar)


def on_bar(symbol, period, bar):
    print(
        f"北京时间:{to_beijing_datetime(bar.time.ts)} 瑞士时间:{datetime.datetime.fromtimestamp(bar.time.ts)}  on_bar {symbol} {period_name[period]}")
    chan = chans[symbol + str(period)]
    chan.trigger_load({period_map[period]: [bar]})
    bsp_list = chan.get_bsp(0)
    if bsp_list:
        last_bsp = bsp_list[-1]
        if chan[0][-1].idx == last_bsp.klu.klc.idx:
            if BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type or BSP_TYPE.T2 in last_bsp.type or \
                    BSP_TYPE.T2S in last_bsp.type or BSP_TYPE.T3A in last_bsp.type or BSP_TYPE.T3B in last_bsp.type:
                if enable_send_mail:
                    subject = f"外汇- {symbol} {period_name[period]} {' '.join([t.name for t in last_bsp.type])} {'买点' if last_bsp.is_buy else '卖点'}"
                    message = f"北京时间:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 瑞士时间:{to_zurich_datetime(datetime.datetime.now().timestamp())}"
                    send_email(to_emails, subject, message, chan)


def init_chan():
    for symbol in symbols:
        tick = MT5.symbol_info_tick(symbol)
        for period in periods:
            bars = MT5.copy_rates_from(symbol, period, tick.time, 500)
            bars = pd.DataFrame(bars[:-1])
            bars.dropna(inplace=True)
            bars['time'] = pd.to_datetime(bars['time'], unit='s')
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
                on_bar(symbol, period, bar)

            last_tick_time[symbol] = tick.time
            next_bar_open[symbol + str(period)] = last_tick_time[symbol]
            next_bar_open[symbol + str(period)] -= next_bar_open[symbol + str(period)] % PeriodSeconds(period)
            next_bar_open[symbol + str(period)] += PeriodSeconds(period)


def initialize_mt5():
    if not MT5.initialize():
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


def get_latest_tick(symbol):
    try:
        tick_info = MT5.symbol_info_tick(symbol)
        if tick_info is None:
            raise ValueError(f"无法获取 {symbol} 的价格信息")
        return tick_info
    except Exception as e:
        print(f"获取 {symbol} 的价格信息时发生异常: {e}")
        return None


if __name__ == "__main__":
    # 连接到MetaTrader 5
    if not MT5.initialize():
        print("initialize() failed")
        MT5.shutdown()
        exit(0)
    chans = {}
    last_tick_time = {}
    next_bar_open = {}
    for symbol in symbols:
        data_src = DATA_SRC.FOREX
        for period in periods:
            lv_list = [period_map[period]]
            config = CChanConfig({
                "trigger_step": True,  # 打开开关！
                "divergence_rate": 0.8,
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
    init_chan()
    enable_send_mail = True
    try:
        while True:
            for symbol in symbols:
                tick_info = get_latest_tick(symbol)
                if tick_info:
                    if tick_info.time > last_tick_time[symbol]:
                        last_tick_time[symbol] = tick_info.time
                        on_tick(symbol, tick_info)
            time.sleep(1)
            # 检查连接状态，如果连接丢失则尝试重新连接
            if not MT5.terminal_info():
                print("连接丢失，尝试重新连接...")
                MT5.shutdown()
                if not reconnect_mt5():
                    break
    except KeyboardInterrupt:
        print("用户中断程序")
    finally:
        MT5.shutdown()
