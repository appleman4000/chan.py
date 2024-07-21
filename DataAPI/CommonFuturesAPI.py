# cython: language_level=3
import abc
from typing import Iterable

from KLine.KLine_Unit import CKLine_Unit


class CCommonForexApi:
    def __init__(self, code, k_type, begin_date, end_date):
        self.code = code
        self.name = None
        self.is_stock = False
        self.k_type = k_type
        self.begin_date = begin_date
        self.end_date = end_date
        self.SetBasciInfo()

    @abc.abstractmethod
    def get_kl_data(self) -> Iterable[CKLine_Unit]:
        pass

    @abc.abstractmethod
    def SetBasciInfo(self):
        pass

    @classmethod
    @abc.abstractmethod
    def do_init(cls):
        pass

    @classmethod
    @abc.abstractmethod
    def do_close(cls):
        pass
