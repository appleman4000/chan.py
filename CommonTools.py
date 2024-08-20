# cython: language_level=3
# encoding:utf-8
import datetime
import io
import time

import MetaTrader5 as mt5
import matplotlib
import numpy as np
import pytz
from PIL import Image
from matplotlib import pyplot as plt

from Chan import CChan
from Common.CEnum import KL_TYPE, DATA_FIELD
from Common.CTime import CTime
from Common.func_util import str2float
from Plot.PlotDriver import CPlotDriver

local_timezone = 'Asia/Shanghai'
server_timezone = 'Europe/Zurich'

period_mt5_map = {
    KL_TYPE.K_1M: mt5.TIMEFRAME_M1,
    KL_TYPE.K_2M: mt5.TIMEFRAME_M2,
    KL_TYPE.K_3M: mt5.TIMEFRAME_M3,
    KL_TYPE.K_4M: mt5.TIMEFRAME_M4,
    KL_TYPE.K_5M: mt5.TIMEFRAME_M5,
    KL_TYPE.K_6M: mt5.TIMEFRAME_M6,
    KL_TYPE.K_10M: mt5.TIMEFRAME_M10,
    KL_TYPE.K_12M: mt5.TIMEFRAME_M12,
    KL_TYPE.K_15M: mt5.TIMEFRAME_M15,
    KL_TYPE.K_20M: mt5.TIMEFRAME_M20,
    KL_TYPE.K_30M: mt5.TIMEFRAME_M30,
    KL_TYPE.K_1H: mt5.TIMEFRAME_H1,
    KL_TYPE.K_2H: mt5.TIMEFRAME_H2,
    KL_TYPE.K_3H: mt5.TIMEFRAME_H3,
    KL_TYPE.K_4H: mt5.TIMEFRAME_H4,
    KL_TYPE.K_6H: mt5.TIMEFRAME_H6,
    KL_TYPE.K_8H: mt5.TIMEFRAME_H8,
    KL_TYPE.K_12H: mt5.TIMEFRAME_H12,
    KL_TYPE.K_DAY: mt5.TIMEFRAME_D1,
    KL_TYPE.K_WEEK: mt5.TIMEFRAME_W1
}
period_seconds = {
    KL_TYPE.K_1M: 60,
    KL_TYPE.K_2M: 120,
    KL_TYPE.K_3M: 180,
    KL_TYPE.K_4M: 240,
    KL_TYPE.K_5M: 300,
    KL_TYPE.K_6M: 360,
    KL_TYPE.K_10M: 600,
    KL_TYPE.K_12M: 720,
    KL_TYPE.K_15M: 900,
    KL_TYPE.K_20M: 1200,
    KL_TYPE.K_30M: 1800,
    KL_TYPE.K_1H: 3600,
    KL_TYPE.K_2H: 7200,
    KL_TYPE.K_3H: 10800,
    KL_TYPE.K_4H: 14400,
    KL_TYPE.K_6H: 21600,
    KL_TYPE.K_8H: 28800,
    KL_TYPE.K_12H: 43200,
    KL_TYPE.K_DAY: 86400,
    KL_TYPE.K_WEEK: 604800,
}

period_name = {
    KL_TYPE.K_1M: "1分钟",
    KL_TYPE.K_2M: "2分钟",
    KL_TYPE.K_3M: "3分钟",
    KL_TYPE.K_4M: "4分钟",
    KL_TYPE.K_5M: "5分钟",
    KL_TYPE.K_6M: "6分钟",
    KL_TYPE.K_10M: "10分钟",
    KL_TYPE.K_12M: "12分钟",
    KL_TYPE.K_15M: "15分钟",
    KL_TYPE.K_20M: "20分钟",
    KL_TYPE.K_30M: "30分钟",
    KL_TYPE.K_1H: "1小时",
    KL_TYPE.K_2H: "2小时",
    KL_TYPE.K_3H: "3小时",
    KL_TYPE.K_4H: "4小时",
    KL_TYPE.K_6H: "6小时",
    KL_TYPE.K_8H: "8小时",
    KL_TYPE.K_12H: "12小时",
    KL_TYPE.K_DAY: "1天",
    KL_TYPE.K_WEEK: "1周",
}


def shanghai_to_zurich_datetime(timestamp):
    zurich_tz = pytz.timezone('Europe/Zurich')
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    timestamp = shanghai_tz.localize(datetime.datetime.fromtimestamp(timestamp))

    # 将时间戳转换到 'Asia/Shanghai' 时区
    zurich_time = timestamp.astimezone(zurich_tz)

    # 格式化时间
    formatted_time = zurich_time.strftime('%Y-%m-%d %H:%M')

    return formatted_time


def create_item_dict(data, column_name):
    for i in range(len(data)):
        data[i] = parse_time_column(data[i]) if i == 0 else str2float(data[i])
    return dict(zip(column_name, data))


def parse_time_column(inp):
    # 20210902113000000
    # 2021-09-13
    if len(inp) == 10:
        year = int(inp[:4])
        month = int(inp[5:7])
        day = int(inp[8:10])
        hour = minute = 0
    elif len(inp) == 17:
        year = int(inp[:4])
        month = int(inp[4:6])
        day = int(inp[6:8])
        hour = int(inp[8:10])
        minute = int(inp[10:12])
    elif len(inp) == 19:
        year = int(inp[:4])
        month = int(inp[5:7])
        day = int(inp[8:10])
        hour = int(inp[11:13])
        minute = int(inp[14:16])
    else:
        raise Exception(f"unknown time column from mt5:{inp}")
    return CTime(year=year, month=month, day=day, hour=hour, minute=minute, second=0, auto=False)


def GetColumnNameFromFieldList(fileds: str):
    _dict = {
        "time": DATA_FIELD.FIELD_TIME,
        "open": DATA_FIELD.FIELD_OPEN,
        "high": DATA_FIELD.FIELD_HIGH,
        "low": DATA_FIELD.FIELD_LOW,
        "close": DATA_FIELD.FIELD_CLOSE,
        "volume": DATA_FIELD.FIELD_VOLUME,
    }
    return [_dict[x] for x in fileds.split(",")]


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
        bar = mt5.copy_rates_from_pos(symbol, period_mt5_map[period], 1, 1)[0]
        if bar is None:
            raise ValueError(f"无法获取 {symbol} 的价格信息")
        return bar
    except Exception as e:
        print(f"获取 {symbol} 的价格信息时发生异常: {e}")
        return None


def robot_trade(symbol, lot=0.01, is_buy=None, comment=""):
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        print(f"Currency pair {symbol} has open positions.")
        return
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


def chan_to_png(chan: CChan, plot_config, plot_para, file_path=""):
    matplotlib.use('Agg')
    g = CPlotDriver(chan, plot_config, plot_para)
    # 移除标题
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
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

    g.figure.tight_layout()
    if file_path:
        g.figure.savefig(file_path, format='PNG', bbox_inches='tight', pad_inches=0.1)
        plt.close(g.figure)
        img = Image.open(file_path).resize((224, 224))
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(file_path)
        return None
    else:
        buf = io.BytesIO()
        g.figure.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        plt.close(g.figure)
        img = Image.open(buf).resize((224, 224))
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        # 将图片转换为 NumPy 数组
        img_array = np.array(img)
        return img_array
