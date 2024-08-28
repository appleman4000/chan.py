# cython: language_level=3
# encoding:utf-8
import datetime
import time

import MetaTrader5 as mt5
import pandas as pd

from CommonTools import period_mt5_map, period_seconds, period_name, create_item_dict, GetColumnNameFromFieldList, \
    server_timezone, local_timezone, reconnect_mt5
from DataAPI.CommonForexAPI import CCommonForexApi
from KLine.KLine_Unit import CKLine_Unit


class CMT5ForexAPI(CCommonForexApi):
    is_connect = None

    def __init__(self, code, k_type, begin_date=None, end_date=None, autype=None):
        super(CMT5ForexAPI, self).__init__(code, k_type, begin_date, end_date)
        # 建立MetaTrader 5到指定交易账户的连接
        # connect to MetaTrader 5
        if not reconnect_mt5():
            print("initialize() failed")
            mt5.shutdown()
            exit(0)
        time.sleep(100)

    def get_kl_data(self):
        local_time_format = '%Y-%m-%d %H:%M:%S'

        # 解析时间字符串为datetime对象
        begin_date = datetime.datetime.strptime(self.begin_date, local_time_format)
        end_date = datetime.datetime.strptime(self.end_date, local_time_format)
        end_date = end_date + datetime.timedelta(hours=2)
        while True:
            bars = mt5.copy_rates_range(self.code, period_mt5_map[self.k_type], begin_date, end_date)
            if bars is None:
                mt5.shutdown()
                if not reconnect_mt5():
                    break
                time.sleep(10)
            else:
                break
        mt5.shutdown()
        bars = pd.DataFrame(bars)
        bars['time'] = pd.to_datetime(bars['time'], unit='s')
        bars['time'] += datetime.timedelta(seconds=period_seconds[self.k_type])  # 开盘时间转收盘时间
        bars['time'] = bars['time'].dt.tz_localize(tz=server_timezone)
        bars['time'] = bars['time'].dt.tz_convert(tz=local_timezone).dt.tz_localize(None)
        bars['time'] = bars['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        fields = "time,open,high,low,close,volume"
        for index, row in bars.iterrows():
            data = [
                row["time"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["tick_volume"],
            ]
            yield CKLine_Unit(create_item_dict(data, GetColumnNameFromFieldList(fields)))

    def SetBasciInfo(self):
        self.name = self.code
        self.is_stock = False

    def __convert_type(self):
        return period_name[self.k_type]
