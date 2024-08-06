# cython: language_level=3
# encoding:utf-8
import datetime

import akshare as ak
import baostock as bs
import pandas as pd

from Common.CEnum import AUTYPE, DATA_FIELD, KL_TYPE
from Common.CTime import CTime
from Common.func_util import str2float
from KLine.KLine_Unit import CKLine_Unit
from .CommonStockAPI import CCommonStockApi


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
        raise Exception(f"unknown time column from baostock:{inp}")
    return CTime(year, month, day, hour, minute)


def GetColumnNameFromFieldList(fileds: str):
    _dict = {
        "time": DATA_FIELD.FIELD_TIME,
        "date": DATA_FIELD.FIELD_TIME,
        "open": DATA_FIELD.FIELD_OPEN,
        "high": DATA_FIELD.FIELD_HIGH,
        "low": DATA_FIELD.FIELD_LOW,
        "close": DATA_FIELD.FIELD_CLOSE,
        "volume": DATA_FIELD.FIELD_VOLUME,
        "amount": DATA_FIELD.FIELD_TURNOVER,
        "turn": DATA_FIELD.FIELD_TURNRATE,
    }
    return [_dict[x] for x in fileds.split(",")]


fund_name_em = None


def load_fund_name_em():
    global fund_name_em
    if fund_name_em is None:
        fund_name_em = ak.fund_name_em()
        return fund_name_em
    else:
        return fund_name_em


class CAKShareETFAPI(CCommonStockApi):
    is_connect = None

    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=AUTYPE.QFQ):
        super(CAKShareETFAPI, self).__init__(code, k_type, begin_date, end_date, autype)

    def get_kl_data(self):
        autype_dict = {AUTYPE.QFQ: "qfq", AUTYPE.HFQ: "hfq", AUTYPE.NONE: ""}
        if self.k_type == KL_TYPE.K_DAY:
            local_from = datetime.datetime.strptime(self.begin_date, '%Y-%m-%d %H:%M:%S')
            local_to = datetime.datetime.strptime(self.end_date, '%Y-%m-%d %H:%M:%S')
            start_date = local_from.strftime("%Y%m%d")
            end_date = local_to.strftime("%Y%m%d")
            bars = ak.fund_etf_hist_em(symbol=self.code, period="daily", start_date=start_date,
                                       end_date=end_date, adjust=autype_dict[self.autype])
        elif self.k_type == KL_TYPE.K_1M:
            bars = ak.fund_etf_hist_min_em(symbol=self.code, period="1", start_date=self.begin_date,
                                           end_date=self.end_date, adjust=autype_dict[self.autype])
        elif self.k_type == KL_TYPE.K_5M:
            bars = ak.fund_etf_hist_min_em(symbol=self.code, period="5", start_date=self.begin_date,
                                           end_date=self.end_date, adjust=autype_dict[self.autype])
        elif self.k_type == KL_TYPE.K_15M:
            bars = ak.fund_etf_hist_min_em(symbol=self.code, period="15", start_date=self.begin_date,
                                           end_date=self.end_date, adjust=autype_dict[self.autype])
        elif self.k_type == KL_TYPE.K_30M:
            bars = ak.fund_etf_hist_min_em(symbol=self.code, period="30", start_date=self.begin_date,
                                           end_date=self.end_date, adjust=autype_dict[self.autype])
        elif self.k_type == KL_TYPE.K_60M:
            bars = ak.fund_etf_hist_min_em(symbol=self.code, period="60", start_date=self.begin_date,
                                           end_date=self.end_date, adjust=autype_dict[self.autype])
        bars.dropna(inplace=True)
        if self.k_type == KL_TYPE.K_DAY:
            bars['时间'] = pd.to_datetime(bars['日期']).dt.strftime('%Y-%m-%d %H:%M:%S')
            del bars['日期']
        for index, row in bars.iterrows():
            data = [
                row["时间"],
                row["开盘"],
                row["最高"],
                row["最低"],
                row["收盘"],
                row["成交量"],
            ]
            fields = "time,open,high,low,close,volume"
            yield CKLine_Unit(create_item_dict(data, GetColumnNameFromFieldList(fields)))

    def SetBasciInfo(self):
        rs = load_fund_name_em()
        rs.where(rs["基金代码"] == self.code, inplace=True)
        rs.dropna(inplace=True)
        code_name = rs["基金简称"].iloc[0]
        stock_type = "1"
        self.name = code_name
        self.is_stock = (stock_type == '1')

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
