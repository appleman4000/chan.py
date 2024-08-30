# cython: language_level=3
# encoding:utf-8
import numpy as np
import talib


class TaIndicators:
    def __init__(self, N=200):
        assert N > 1
        self.N = N
        self.arr = []

    def add(self, high, low, close) -> dict:
        if len(self.arr) == 0:
            self.arr = [[high, low, close]] * self.N
        else:
            self.arr.append([high, low, close])
        if len(self.arr) > self.N:
            del self.arr[0]
        highest = np.array(self.arr)[:, 0].astype(np.float64)
        lowest = np.array(self.arr)[:, 1].astype(np.float64)
        closing = np.array(self.arr)[:, 2].astype(np.float64)
        returns = dict()
        returns["ADX"] = talib.ADX(highest, lowest, closing, timeperiod=14)[-1]
        returns["ADXR"] = talib.ADXR(highest, lowest, closing, timeperiod=14)[-1]
        returns["APO"] = talib.APO(closing, fastperiod=12, slowperiod=26)[-1]
        AROONDOWN, AROONUP = talib.AROON(highest, lowest, timeperiod=14)
        returns["AROONDOWN"] = AROONDOWN[-1]
        returns["AROONUP"] = AROONUP[-1]
        returns["AROONOSC"] = talib.AROONOSC(highest, lowest, timeperiod=14)[-1]
        UPPERBAND, MIDDLEBAND, LOWERBAND = \
            talib.BBANDS(closing, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        returns["UPPERBAND"] = UPPERBAND[-1] / close - 1
        returns["MIDDLEBAND"] = MIDDLEBAND[-1] / close - 1
        returns["LOWERBAND"] = LOWERBAND[-1] / close - 1
        MACD_DIF, MACD_DEA, MACD_BAR = talib.MACD(closing, fastperiod=12, slowperiod=26, signalperiod=9)
        returns["MACD_DIF"] = MACD_DIF[-1]
        returns["MACD_DEA"] = MACD_DEA[-1]
        returns["MACD_BAR"] = MACD_BAR[-1]
        returns["PPO"] = talib.PPO(closing, fastperiod=12, slowperiod=26, matype=0)[-1]
        returns["RSI"] = talib.RSI(closing, timeperiod=14)[-1] / 100.0
        returns["SAR"] = talib.SAR(highest, lowest, acceleration=0.02, maximum=0.2)[-1] / close - 1
        returns["WILLR"] = talib.WILLR(highest, lowest, closing, timeperiod=14)[-1]
        returns["MOM"] = talib.MOM(closing, timeperiod=10)[-1]
        periods = [6, 12, 26]
        for period in periods:
            returns[f"DEMA{period}"] = talib.DEMA(closing, timeperiod=period)[-1] / close - 1
            returns[f"EMA{period}"] = talib.EMA(closing, timeperiod=period)[-1] / close - 1
            returns[f"TEMA{period}"] = talib.TEMA(closing, timeperiod=period)[-1] / close - 1
        periods = [14, 20]
        for period in periods:
            returns[f"ATR{period}"] = talib.ATR(highest, lowest, closing, timeperiod=period)[-1]
        periods = [14]
        for period in periods:
            returns[f"CCI{period}"] = talib.CCI(highest, lowest, closing, timeperiod=period)[-1] / 100.0
        periods = [12, 26]
        for period in periods:
            returns[f"CMO{period}"] = talib.CMO(closing, timeperiod=period)[-1]
        periods = [6, 20]
        for period in periods:
            returns[f"ROC{period}"] = talib.ROC(closing, timeperiod=period)[-1]
        periods = [6, 20]
        for period in periods:
            returns[f"ROCP{period}"] = talib.ROCP(closing, timeperiod=period)[-1]
        periods = [10, 20, 40, 80]
        for period in periods:
            returns[f"MAX{period}"] = talib.MAX(highest, timeperiod=period)[-1] / close - 1
            returns[f"MIN{period}"] = talib.MIN(lowest, timeperiod=period)[-1] / close - 1
        return returns
