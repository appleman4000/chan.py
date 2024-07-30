# cython: language_level=3
import datetime

import MetaTrader5 as MT5
import baostock as bs
import pandas as pd
import pytz

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


class CMT5ForexAPI(CCommonForexApi):
    is_connect = None

    def __init__(self, code, k_type, begin_date=None, end_date=None, autype=None):
        super(CMT5ForexAPI, self).__init__(code, k_type, begin_date, end_date)
        # 建立MetaTrader 5到指定交易账户的连接
        # connect to MetaTrader 5
        if not MT5.initialize(server="Swissquote-Server", login=6150644, password="Sj!i2zHy"):
            print("initialize() failed")
            MT5.shutdown()
            exit(0)

        # request connection status and parameters
        print(MT5.terminal_info())
        # get data on MetaTrader 5 version
        print(MT5.version())

    def get_kl_data(self):
        if self.k_type == KL_TYPE.K_1M:
            timeframe = MT5.TIMEFRAME_M1
        elif self.k_type == KL_TYPE.K_3M:
            timeframe = MT5.TIMEFRAME_M3
        elif self.k_type == KL_TYPE.K_5M:
            timeframe = MT5.TIMEFRAME_M5
        elif self.k_type == KL_TYPE.K_15M:
            timeframe = MT5.TIMEFRAME_M15
        elif self.k_type == KL_TYPE.K_30M:
            timeframe = MT5.TIMEFRAME_M30
        elif self.k_type == KL_TYPE.K_60M:
            timeframe = MT5.TIMEFRAME_H1
        elif self.k_type == KL_TYPE.K_DAY:
            timeframe = MT5.TIMEFRAME_D1
        else:
            raise Exception("不支持的时间框")
        local_time_format = '%Y-%m-%d %H:%M:%S'

        local_tz = pytz.timezone('Asia/Shanghai')
        zurich_tz = pytz.timezone('Europe/Zurich')
        # create 'datetime' objects in UTC time zone to avoid the implementation of a local time zone offset
        # 解析时间字符串为datetime对象
        begin_date = datetime.datetime.strptime(self.begin_date, local_time_format)
        # 本地时区

        # 将datetime对象本地化为本地时区时间
        begin_date = local_tz.localize(begin_date)
        # 将本地时区时间转换为UTC时间
        begin_date = begin_date.astimezone(pytz.utc)

        end_date = datetime.datetime.strptime(self.end_date, local_time_format)

        # 将datetime对象本地化为本地时区时间
        end_date = local_tz.localize(end_date)
        # 将本地时区时间转换为UTC时间
        end_date = end_date.astimezone(pytz.utc)

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

        # bars = MT5.copy_rates_range(self.code, timeframe, begin_date, end_date)
        bars = MT5.copy_rates_from_pos(self.code, timeframe, 0, 1000)
        bars = pd.DataFrame(bars)
        bars.dropna(inplace=True)
        bars['time'] = pd.to_datetime(bars['time'], unit='s')
        bars['time'] = bars['time'].dt.tz_localize('Europe/Zurich')
        bars['time'] = bars['time'].dt.tz_convert(local_tz)
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
        _dict = {
            KL_TYPE.K_DAY: 'd',
            KL_TYPE.K_WEEK: 'w',
            KL_TYPE.K_MON: 'm',
            KL_TYPE.K_1M: '1',
            KL_TYPE.K_5M: '5',
            KL_TYPE.K_15M: '15',
            KL_TYPE.K_30M: '30',
            KL_TYPE.K_60M: '60',
        }
        return _dict[self.k_type]
