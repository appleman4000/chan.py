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
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, AUTYPE, KL_TYPE, BSP_TYPE, FX_TYPE
from DataAPI.MT5ForexAPI import GetColumnNameFromFieldList, create_item_dict
from KLine.KLine_Unit import CKLine_Unit
from Messenger import send_message

sys.setrecursionlimit(10000)
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
    mt5.TIMEFRAME_H1: KL_TYPE.K_1H,
    mt5.TIMEFRAME_H4: KL_TYPE.K_4H,
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
    "XAUUSD",
    "XAGUSD",
]
periods = [mt5.TIMEFRAME_D1, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_M15]
# to_emails = ['appleman4000@qq.com', 'xubin.njupt@foxmail.com', '375961433@qq.com', 'idbeny@163.com', 'jflzhao@163.com',
#              '837801694@qq.com', '1169006942@qq.com', 'vincent1122@126.com']
to_emails = ['appleman4000@qq.com']


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


def robot_trade(symbol, lot=0.01, is_buy=None, comment=""):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(symbol, "not found, can not call order_check()")
        return
    # if the symbol is unavailable in MarketWatch, add it
    if not symbol_info.visible:
        print(symbol, "is not visible, trying to switch on")
        if not mt5.symbol_select(symbol, True):
            print("symbol_select({}}) failed, exit", symbol)
            return
    for tries in range(10):
        point = mt5.symbol_info(symbol).point
        price = mt5.symbol_info_tick(symbol).ask if is_buy else mt5.symbol_info_tick(symbol).bid
        tp = round(price * 0.003 / point)
        sl = round(price * 0.01 / point)
        deviation = 30
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": price - sl * point if is_buy else price + sl * point,
            "tp": price + tp * point if is_buy else price - tp * point,
            "deviation": deviation,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_DAY,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        # send a trading request
        result = mt5.order_send(request)
        if result is None:
            return
        if result.retcode != mt5.TRADE_RETCODE_REQUOTE and result.retcode != mt5.TRADE_RETCODE_PRICE_OFF:
            if is_buy:
                print(f"{symbol} buy order placed successfully")
            else:
                print(f"{symbol} sell order placed successfully")
            break


def on_bar(symbol, period, bar, enable_send_message=False):
    print(
        f"北京时间:{bar.time} 瑞士时间:{shanghai_to_zurich_datetime(bar.time.ts)}  on_bar {symbol} {period_name[period]}")
    chan = chans[symbol + str(period)]
    chan.trigger_load({period_map[period]: [bar]})

    if enable_send_message and period == mt5.TIMEFRAME_M15:
        # 5分钟买卖点,底分型或者顶分型成立
        chan_m15 = chan[0]
        # 确保分型已确认
        if chan_m15[-2].fx not in [FX_TYPE.BOTTOM, FX_TYPE.TOP]:
            return
        # 1小时买卖点，分型待确认
        chan_h1 = chans[symbol + str(mt5.TIMEFRAME_H1)]
        bsp_list = chan_h1.get_bsp(0)
        if not bsp_list:
            return
        chan_h1 = chan_h1[0]
        last_bsp_h1 = bsp_list[-1]
        if BSP_TYPE.T1 not in last_bsp_h1.type and BSP_TYPE.T1P not in last_bsp_h1.type:
            return
        # if chan_h1[-1].idx - last_bsp_h1.klu.klc.idx != 0:
        #     return
        if last_bsp_h1.klu.time != chan_h1[-1][-1].time:
            return
        # 1小时买卖点和15分钟方向一致
        if (last_bsp_h1.is_buy and chan_m15[-2].fx != FX_TYPE.BOTTOM or
                not last_bsp_h1.is_buy and chan_m15[-2].fx != FX_TYPE.TOP):
            return

        price = f"{bar.close:.5f}".rstrip('0').rstrip('.')
        subject = f"外汇- {symbol} {period_name[mt5.TIMEFRAME_H1]} {' '.join([t.name for t in last_bsp_h1.type])} {'买点' if last_bsp_h1.is_buy else '卖点'} {price}"
        message = f"北京时间:{datetime.datetime.fromtimestamp(bar.time.ts + period_seconds(period)).strftime('%Y-%m-%d %H:%M')} 瑞士时间:{shanghai_to_zurich_datetime(bar.time.ts + period_seconds(period))}"
        send_message(subject, message, [chans[symbol + str(p)] for p in periods])
        comment = f"{last_bsp_h1.klu.time.to_str()} {' '.join([t.name for t in last_bsp_h1.type])}"
        robot_trade(symbol, 0.01, last_bsp_h1.is_buy, comment)


def init_chan():
    local_tz = pytz.timezone('Asia/Shanghai')
    for symbol in symbols:
        data_src = DATA_SRC.FOREX
        for period in periods:
            lv_list = [period_map[period]]
            config = CChanConfig({
                "trigger_step": True,  # 打开开关！
                "bi_strict": True,
                "gap_as_kl": True,
                "min_zs_cnt": 1,
                "divergence_rate": 0.8,
                "max_bs2_rate": 0.618,
                "macd_algo": "diff",
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
            bars = mt5.copy_rates_from_pos(symbol, period, 1, 1000)
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
                on_bar(symbol, period, bar, enable_send_message=False)
            next_bar_open[symbol + str(period)] = last_bar_time
            next_bar_open[symbol + str(period)] -= next_bar_open[symbol + str(period)] % period_seconds(period)
            next_bar_open[symbol + str(period)] += period_seconds(period)


def initialize_mt5():
    if not mt5.initialize(server="Swissquote-Server", login=6150644, password="Sj!i2zHy"):
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


def get_latest_bar(symbol, period):
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
                    bar = get_latest_bar(symbol, period)
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
