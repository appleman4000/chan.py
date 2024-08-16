# cython: language_level=3
# encoding:utf-8
import datetime
import time

import MetaTrader5 as mt5
import baostock as bs
import pandas as pd

from Common.CEnum import DATA_FIELD, KL_TYPE
from Common.CTime import CTime
from Common.func_util import str2float
from DataAPI.CommonForexAPI import CCommonForexApi
from KLine.KLine_Unit import CKLine_Unit


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


def period_seconds(period):
    return timeframe_seconds[period]


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


class CandleIterator:
    def __init__(self, symbol, period, start_date):
        self.symbol = symbol
        self.period = period
        self.start_date = start_date
        self.last_time = start_date
        self.data = self._fetch_candles(self.start_date)
        self.index = 0
        self.next_bar_open = start_date
        self.next_bar_open -= datetime.timedelta(seconds=self.next_bar_open.timestamp() % period_seconds(self.period))
        self.next_bar_open += datetime.timedelta(seconds=period_seconds(self.period))

    def _fetch_candles(self, start_date):
        # 获取从 start_date 开始的K线数据
        bars = mt5.copy_rates_range(self.symbol, self.period, start_date,
                                    datetime.datetime.now() + datetime.timedelta(hours=2))
        if bars is None:
            print(f"获取数据失败: {mt5.last_error()}")
            return None
        bars = pd.DataFrame(bars)
        bars['time'] = pd.to_datetime(bars['time'], unit='s')
        # bars['time'] += datetime.timedelta(seconds=timeframe_seconds[self.period])
        bars['time'] = bars['time'].dt.tz_localize('Europe/Zurich')
        bars['time'] = bars['time'].dt.tz_convert("Asia/Shanghai")
        bars['time'] = bars['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        return bars

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            # 如果已经到达数据末尾，则获取新的数据
            self.data = self._fetch_candles(self.last_time)
            if self.data is None or self.data.empty:
                raise StopIteration
            self.index = 0

        candle = self.data.iloc[self.index]
        self.index += 1
        self.last_time = datetime.datetime.strptime(candle['time'], "%Y-%m-%d %H:%M:%S") + datetime.timedelta(hours=2)
        self.next_bar_open = self.last_time
        self.next_bar_open -= datetime.timedelta(
            seconds=self.next_bar_open.timestamp() % period_seconds(self.period))
        self.next_bar_open += datetime.timedelta(seconds=period_seconds(self.period))
        return candle


class CMT5ForexOnlineAPI(CCommonForexApi):
    is_connect = None

    def __init__(self, code, k_type, begin_date=None, end_date=None, autype=None):
        super(CMT5ForexOnlineAPI, self).__init__(code, k_type, begin_date, end_date)
        # 建立MetaTrader 5到指定交易账户的连接
        # connect to MetaTrader 5
        if not reconnect_mt5():
            print("initialize() failed")
            mt5.shutdown()
            exit(0)

        # request connection status and parameters
        print(mt5.terminal_info())
        # get data on MetaTrader 5 version
        print(mt5.version())
        end_date = datetime.datetime.now() + datetime.timedelta(hours=2)
        start_date = end_date - datetime.timedelta(days=5)
        if k_type == KL_TYPE.K_1M:
            period = mt5.TIMEFRAME_M1
        elif k_type == KL_TYPE.K_3M:
            period = mt5.TIMEFRAME_M3
        elif k_type == KL_TYPE.K_5M:
            period = mt5.TIMEFRAME_M5
        elif k_type == KL_TYPE.K_15M:
            period = mt5.TIMEFRAME_M15
        elif k_type == KL_TYPE.K_30M:
            period = mt5.TIMEFRAME_M30
        elif k_type == KL_TYPE.K_1H:
            period = mt5.TIMEFRAME_H1
        elif k_type == KL_TYPE.K_4H:
            period = mt5.TIMEFRAME_H4
        elif k_type == KL_TYPE.K_DAY:
            period = mt5.TIMEFRAME_D1
        else:
            raise Exception("不支持的时间框")
        self.period = period
        self.iterator = CandleIterator(code, period, start_date)

    def get_kl_data(self):
        fields = "time,open,high,low,close,volume"
        interval = 1
        while True:
            if (datetime.datetime.now() + datetime.timedelta(
                    hours=2)).timestamp() >= self.iterator.next_bar_open.timestamp():
                candle = self.iterator.__next__()
                # print(candle)  # 打印每根K线的数据
                data = [
                    candle["time"],
                    candle["open"],
                    candle["high"],
                    candle["low"],
                    candle["close"],
                    candle["tick_volume"],
                ]
                bar = CKLine_Unit(create_item_dict(data, GetColumnNameFromFieldList(fields)))
                yield bar
            else:
                time.sleep(interval)  # 等待指定的时间间隔再获取数据
            if not mt5.terminal_info():
                print("连接丢失，尝试重新连接...")
                mt5.shutdown()
                if not reconnect_mt5():
                    break

    def SetBasciInfo(self):
        self.name = self.code
        self.is_stock = False

    @classmethod
    def do_init(cls):
        if not cls.is_connect:
            cls.is_connect = bs.login()

    @classmethod
    def do_close(cls):
        if cls.is_connect:
            bs.logout()
            cls.is_connect = None

    def __convert_type(self):
        _dict = {
            KL_TYPE.K_DAY: 'd',
            KL_TYPE.K_WEEK: 'w',
            KL_TYPE.K_MON: 'm',
            KL_TYPE.K_1M: '1',
            KL_TYPE.K_5M: '5',
            KL_TYPE.K_15M: '15',
            KL_TYPE.K_30M: '30',
            KL_TYPE.K_1H: '60',
            KL_TYPE.K_4H: '240',
        }
        return _dict[self.k_type]
