# cython: language_level=3
# encoding:utf-8
import datetime

import MetaTrader5 as mt5
import baostock as bs
import pandas as pd

from CommonTools import period_mt5_map, period_seconds, period_name, create_item_dict, GetColumnNameFromFieldList, \
    server_timezone, local_timezone
from DataAPI.CommonForexAPI import CCommonForexApi
from KLine.KLine_Unit import CKLine_Unit


class CMT5ForexAPI(CCommonForexApi):
    is_connect = None

    def __init__(self, code, k_type, begin_date=None, end_date=None, autype=None):
        super(CMT5ForexAPI, self).__init__(code, k_type, begin_date, end_date)
        # 建立MetaTrader 5到指定交易账户的连接
        # connect to MetaTrader 5
        if not mt5.initialize(server="Swissquote-Server", login=6150644, password="Sj!i2zHy"):
            print("initialize() failed")
            mt5.shutdown()
            exit(0)

        # request connection status and parameters
        print(mt5.terminal_info())
        # get data on MetaTrader 5 version
        print(mt5.version())

    def get_kl_data(self):
        local_time_format = '%Y-%m-%d %H:%M:%S'

        # 解析时间字符串为datetime对象
        begin_date = datetime.datetime.strptime(self.begin_date, local_time_format)
        end_date = datetime.datetime.strptime(self.end_date, local_time_format)
        end_date = end_date + datetime.timedelta(hours=2)

        bars = mt5.copy_rates_range(self.code, period_mt5_map[self.k_type], begin_date, end_date)
        bars = pd.DataFrame(bars)
        bars.dropna(inplace=True)
        bars['time'] = pd.to_datetime(bars['time'], unit='s')
        bars['time'] = bars['time'] + datetime.timedelta(seconds=period_seconds[self.k_type])
        bars['time'] = bars['time'].dt.tz_localize(server_timezone)
        bars['time'] = bars['time'].dt.tz_convert(local_timezone)
        bars['time'] = bars['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # bars.set_index('time', inplace=True)
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
