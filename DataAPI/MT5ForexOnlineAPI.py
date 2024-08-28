# cython: language_level=3
# encoding:utf-8
import datetime
import time

import MetaTrader5 as mt5
import baostock as bs
import pandas as pd

from CommonTools import period_seconds, server_timezone, local_timezone, create_item_dict, GetColumnNameFromFieldList, \
    reconnect_mt5, period_name, period_mt5_map
from DataAPI.CommonForexAPI import CCommonForexApi
from KLine.KLine_Unit import CKLine_Unit


class CandleIterator:
    def __init__(self, symbol, period, start_date):
        self.symbol = symbol
        self.period = period
        self.start_date = start_date
        self.last_time = start_date
        self.data = self._fetch_candles(self.start_date)
        self.index = 0
        self.next_bar_open = start_date
        self.next_bar_open -= datetime.timedelta(seconds=self.next_bar_open.timestamp() % period_seconds[self.period])
        self.next_bar_open += datetime.timedelta(seconds=period_seconds[self.period])

    def _fetch_candles(self, start_date):
        # 获取从 start_date 开始的K线数据
        current = datetime.datetime.now() + datetime.timedelta(hours=2)
        current -= datetime.timedelta(seconds=current.timestamp() % period_seconds[self.period])
        current -= datetime.timedelta(seconds=period_seconds[self.period])
        while True:
            bars = mt5.copy_rates_range(self.symbol, period_mt5_map[self.period], start_date, current)
            if bars is None:
                if mt5.connection_status()[0] != mt5.TRADE_RETCODE_DONE:
                    mt5.shutdown()
                    if not reconnect_mt5():
                        return None
                    time.sleep(5)
                else:
                    return None
            else:
                break

        bars = pd.DataFrame(bars)
        bars['time'] = pd.to_datetime(bars['time'], unit='s')
        bars['time'] += datetime.timedelta(seconds=period_seconds[self.period])  # 开盘时间转收盘时间
        bars['time'] = bars['time'].dt.tz_localize(tz=server_timezone)
        bars['time'] = bars['time'].dt.tz_convert(tz=local_timezone).dt.tz_localize(None)
        bars['time'] = bars['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        return bars

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            # 如果已经到达数据末尾，则获取新的数据
            self.data = self._fetch_candles(self.last_time)
            if self.data is None or self.data.empty:
                return None
            self.index = 0

        candle = self.data.iloc[self.index]
        self.index += 1
        self.last_time = datetime.datetime.strptime(candle['time'], "%Y-%m-%d %H:%M:%S") + datetime.timedelta(hours=2)
        self.next_bar_open = self.last_time
        self.next_bar_open -= datetime.timedelta(
            seconds=self.next_bar_open.timestamp() % period_seconds[self.period])
        self.next_bar_open += datetime.timedelta(seconds=period_seconds[self.period])
        return candle


class CMT5ForexOnlineAPI(CCommonForexApi):
    is_connect = None

    def __init__(self, code, k_type, begin_date=None, end_date=None, autype=None):

        # 建立MetaTrader 5到指定交易账户的连接
        if not reconnect_mt5():
            print("initialize() failed")
            mt5.shutdown()
            exit(0)

        # request connection status and parameters
        print(mt5.terminal_info())
        # get data on MetaTrader 5 version
        print(mt5.version())

        if begin_date is None:
            begin_date = datetime.datetime.now() - datetime.timedelta(days=10)

        self.iterator = CandleIterator(code, k_type, begin_date)
        super(CMT5ForexOnlineAPI, self).__init__(code, k_type, begin_date, end_date=None)

    def get_kl_data(self):
        fields = "time,open,high,low,close,volume"
        interval = 1
        while True:
            current = datetime.datetime.now() + datetime.timedelta(hours=2)  # 服务器时间比本地时间多两个小时
            current -= datetime.timedelta(seconds=current.timestamp() % period_seconds[self.k_type])
            if current.timestamp() >= self.iterator.next_bar_open.timestamp():
                candle = self.iterator.__next__()
                if candle is None:
                    time.sleep(interval)  # 等待指定的时间间隔再获取数据
                    continue
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
                # print(f"{self.k_type}:{bar}")
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
        return period_name[self.k_type]
