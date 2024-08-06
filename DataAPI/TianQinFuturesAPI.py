# cython: language_level=3
# encoding:utf-8
import baostock as bs
import pandas as pd
from tqsdk import TqApi, TqKq, TqAuth

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
        raise Exception(f"unknown time column from tianqin:{inp}")
    return CTime(year, month, day, hour, minute)


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


class CTianQinFuturesAPI(CCommonForexApi):
    is_connect = None

    def __init__(self, code, k_type, begin_date=None, end_date=None, autype=None):
        super(CTianQinFuturesAPI, self).__init__(code, k_type, begin_date, end_date)
        user_name = "appleman4000"
        password = "123654"
        self.api = TqApi(TqKq(), auth=TqAuth(user_name, password), web_gui=False, disable_print=True, debug=False)
        kq_symbol = "KQ.m@" + code
        self.symbol = self.api.get_quote(kq_symbol).underlying_symbol

    def get_kl_data(self):
        base_length = 30
        if self.k_type == KL_TYPE.K_1M:
            duration_seconds = 60
            delta = pd.Timedelta(minutes=0, seconds=59)
            data_length = base_length * 24 * 60
        elif self.k_type == KL_TYPE.K_3M:
            duration_seconds = 60 * 3
            data_length = base_length * 24 * 20
        elif self.k_type == KL_TYPE.K_5M:
            duration_seconds = 60 * 5
            data_length = base_length * 24 * 6
        elif self.k_type == KL_TYPE.K_15M:
            duration_seconds = 60 * 15
            data_length = base_length * 24 * 4
        elif self.k_type == KL_TYPE.K_30M:
            duration_seconds = 60 * 30
            data_length = base_length * 24 * 2
        elif self.k_type == KL_TYPE.K_60M:
            duration_seconds = 60 * 60
            delta = pd.Timedelta(minutes=59, seconds=59)
            data_length = base_length * 24
        elif self.k_type == KL_TYPE.K_DAY:
            duration_seconds = 60 * 60 * 24
            data_length = base_length
        else:
            raise Exception("不支持的时间框")

        bars = self.api.get_kline_serial(symbol=self.symbol, duration_seconds=duration_seconds, data_length=data_length)
        bars = pd.DataFrame(bars)
        bars.dropna(inplace=True)
        bars['time'] = pd.to_datetime(bars['datetime'], unit='ns')
        bars['time'] = bars['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        fields = "time,open,high,low,close,volume"

        for index, row in bars.iterrows():
            data = [
                row["time"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
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
        _dict = {
            KL_TYPE.K_DAY: 'd',
            KL_TYPE.K_WEEK: 'w',
            KL_TYPE.K_MON: 'm',
            KL_TYPE.K_5M: '5',
            KL_TYPE.K_15M: '15',
            KL_TYPE.K_30M: '30',
            KL_TYPE.K_60M: '60',
        }
        return _dict[self.k_type]
