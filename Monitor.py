import datetime
import time

import MetaTrader5 as MT5
import pandas as pd
import pytz

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, AUTYPE, KL_TYPE, BSP_TYPE
from Common.func_util import str2float
from DataAPI.MT5ForexAPI import GetColumnNameFromFieldList, parse_time_column
from KLine.KLine_Unit import CKLine_Unit
from mail_tools import send_email

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
periods = [MT5.TIMEFRAME_H1, MT5.TIMEFRAME_M15, MT5.TIMEFRAME_M5]
enable_send_mail = False
to_emails = ['appleman4000@qq.com', 'xubin.njupt@foxmail.com', '375961433@qq.com', 'idbeny@163.com', 'jflzhao@163.com',
             '837801694@qq.com']


def TimeCurrent():
    local_current = datetime.datetime.now()
    time_struct_current = time.mktime(local_current.timetuple())
    utc_current = datetime.datetime.utcfromtimestamp(time_struct_current)
    return utc_current.timestamp()


def to_beijing_datetime(timestamp):
    # 创建一个包含 'Europe/Athens' 时区信息的时间戳
    zurich_tz = pytz.timezone('Europe/Zurich')
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    timestamp = zurich_tz.localize(datetime.datetime.fromtimestamp(timestamp))

    # 将时间戳转换到 'Asia/Shanghai' 时区
    shanghai_time = timestamp.astimezone(shanghai_tz)

    # 格式化时间并包含时区信息
    formatted_time = shanghai_time.strftime('%Y-%m-%d %H:%M:%S')

    return formatted_time


def PeriodSeconds(period):
    return timeframe_seconds[period]


def create_item_dict(data, column_name):
    for i in range(len(data)):
        data[i] = parse_time_column(data[i]) if i == 0 else str2float(data[i])
    return dict(zip(column_name, data))


def on_tick(symbol, tick):
    global next_bar_open
    for period in periods:
        if tick.time >= next_bar_open[symbol + str(period)]:
            next_bar_open[symbol + str(period)] = tick.time
            next_bar_open[symbol + str(period)] -= next_bar_open[symbol + str(period)] % PeriodSeconds(period)
            next_bar_open[symbol + str(period)] += PeriodSeconds(period)
            bars = MT5.copy_rates_from_pos(symbol, period, 0, 1)
            bars = pd.DataFrame(bars)
            bars.dropna(inplace=True)
            bars['time'] = pd.to_datetime(bars['time'], unit='s').tz_localize('Asia/Shanghai')
            bars['time'] = bars['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            bar = bars.iloc[0]
            fields = "time,open,high,low,close,volume"
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
        f"北京时间:{to_beijing_datetime(bar.time.ts)} 瑞士时间:{datetime.datetime.fromtimestamp(bar.time.ts)}  on_bar {symbol} {str(period)}")
    chan = chans[symbol + str(period)]
    chan.trigger_load({period_map[period]: [bar]})
    bsp_list = chan.get_bsp(0)
    if bsp_list:
        last_bsp = bsp_list[-1]
        if chan[0][-1].idx == last_bsp.klu.klc.idx:
            if BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type or BSP_TYPE.T2 in last_bsp.type or \
                    BSP_TYPE.T2S in last_bsp.type or BSP_TYPE.T3A in last_bsp.type or BSP_TYPE.T3B in last_bsp.type:
                if enable_send_mail:
                    message = f"北京时间:{to_beijing_datetime(bar.time.ts)} 瑞士时间:{datetime.datetime.fromtimestamp(bar.time.ts)} {symbol} {period_name[period]} {' '.join([t.name for t in last_bsp.type])} {'买' if last_bsp.is_buy else '卖'}"
                    print(message)
                    send_email(to_emails, message)


def init():
    for symbol in symbols:
        for period in periods:
            bars = MT5.copy_rates_from_pos(symbol, period, 0, 1000)
            last_tick_time[symbol] = bars[-1][0]
            next_bar_open[symbol + str(period)] = bars[-1][0]
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


if __name__ == "__main__":
    # 连接到MetaTrader 5
    if not MT5.initialize():
        print("initialize() failed")
        MT5.shutdown()
    chans = {}
    last_tick_time = {}
    next_bar_open = {}
    for symbol in symbols:
        data_src = DATA_SRC.FOREX
        for period in periods:
            lv_list = [period_map[period]]
            config = CChanConfig({
                "trigger_step": True,  # 打开开关！
                "bi_strict": True,
                "skip_step": 0,
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
                code=symbol,
                begin_time=None,
                end_time=None,
                data_src=data_src,
                lv_list=lv_list,
                config=config,
                autype=AUTYPE.NONE,
            )
            chans[symbol + str(period)] = chan
    init()
    enable_send_mail = True
    while True:
        for symbol in symbols:
            tick = MT5.symbol_info_tick(symbol)
            if tick.time > last_tick_time[symbol]:
                last_tick_time[symbol] = tick.time
                on_tick(symbol, tick)
        time.sleep(0.01)
    # 关闭MetaTrader 5
    MT5.shutdown()
