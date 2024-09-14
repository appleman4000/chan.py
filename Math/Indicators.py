# cython: language_level=3
# encoding:utf-8

import numpy as np
import talib


def calculate_all_cdl_patterns(opening, highest, lowest, closing):
    """
    计算所有 TA-Lib 提供的形态指标。

    参数:
    - data: pd.DataFrame，包含 'open', 'high', 'low', 'close' 列的 DataFrame。

    返回:
    - pd.DataFrame，包含所有形态指标计算结果的 DataFrame。
    """
    cdl_patterns = {}
    # 获取所有 TA-Lib 形态识别函数名称
    pattern_functions = [func for func in dir(talib) if func.startswith('CDL')]

    # 对每个形态识别函数计算结果
    for pattern in pattern_functions:
        pattern_function = getattr(talib, pattern)
        cdl_patterns[pattern] = pattern_function(opening, highest, lowest, closing)[-1]
    return cdl_patterns


class TaIndicators:
    def __init__(self, N):
        assert N > 1
        self.N = N
        self.arr = []

    def add(self, open, high, low, close) -> dict:
        self.arr.append([open, high, low, close])

        if len(self.arr) > self.N:
            del self.arr[0]
        opening = np.array(self.arr)[:, 0].astype(np.float64)
        highest = np.array(self.arr)[:, 1].astype(np.float64)
        lowest = np.array(self.arr)[:, 2].astype(np.float64)
        closing = np.array(self.arr)[:, 3].astype(np.float64)
        returns = dict()

        returns["WILLR"] = talib.WILLR(highest, lowest, closing, timeperiod=14)[-1]
        periods = [6, 20]
        for period in periods:
            returns[f"ROC{period}"] = talib.ROCP(closing, timeperiod=period)[-1]
        periods = [10, 20]
        for period in periods:
            returns[f"MAX{period}"] = talib.MAX(highest, timeperiod=period)[-1] / close-1
            returns[f"MIN{period}"] = talib.MIN(lowest, timeperiod=period)[-1] / close-1
        UPPERBAND, MIDDLEBAND, LOWERBAND = \
            talib.BBANDS(closing, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        returns["UPPERBAND"] = UPPERBAND[-1] / close-1
        returns["MIDDLEBAND"] = MIDDLEBAND[-1] / close-1
        returns["LOWERBAND"] = LOWERBAND[-1] / close-1
        MACD_DIF, MACD_DEA, MACD_BAR = talib.MACD(closing, fastperiod=12, slowperiod=26, signalperiod=9)
        returns["MACD_DIF"] = MACD_DIF[-1]
        returns["MACD_DEA"] = MACD_DEA[-1]
        returns["MACD_BAR"] = MACD_BAR[-1]
        returns["RSI"] = talib.RSI(closing, timeperiod=14)[-1] / 100.0
        returns["CCI"] = talib.CCI(highest, lowest, closing, timeperiod=14)[-1] / 100.0

        K, D = talib.STOCH(highest, lowest, closing,
                           fastk_period=9,  # K 线的周期
                           slowk_period=3,  # K 线的平滑周期
                           slowk_matype=0,  # 使用简单移动平均
                           slowd_period=3,  # D 线的周期
                           slowd_matype=0)  # 使用简单移动平均
        returns["K"] = K[-1]
        returns["D"] = D[-1]
        returns["J"] = 3 * K[-1] - 2 * D[-1]
        returns["AROONOSC"] = talib.AROONOSC(highest, lowest, timeperiod=14)[-1] / 100.0
        returns["PPO"] = talib.PPO(closing, fastperiod=12, slowperiod=26, matype=0)[-1] / 100.0
        periods = [12, 26]
        for period in periods:
            returns[f"CMO{period}"] = talib.CMO(closing, timeperiod=period)[-1] / 100.0
        returns["ADX"] = talib.ADX(highest, lowest, closing, timeperiod=14)[-1]
        returns["ADXR"] = talib.ADXR(highest, lowest, closing, timeperiod=14)[-1]
        returns["APO"] = talib.APO(closing, fastperiod=12, slowperiod=26)[-1]
        AROONDOWN, AROONUP = talib.AROON(highest, lowest, timeperiod=14)
        returns["AROONDOWN"] = AROONDOWN[-1] / 100.0
        returns["AROONUP"] = AROONUP[-1] / 100.0

        returns["SAR"] = talib.SAR(highest, lowest, acceleration=0.02, maximum=0.2)[-1] / close-1

        returns["MOM"] = talib.MOM(closing, timeperiod=10)[-1]
        periods = [6, 12, 26]
        for period in periods:
            returns[f"DEMA{period}"] = talib.DEMA(closing, timeperiod=period)[-1] / close-1
            returns[f"EMA{period}"] = talib.EMA(closing, timeperiod=period)[-1] / close-1
            returns[f"TEMA{period}"] = talib.TEMA(closing, timeperiod=period)[-1] / close-1
        periods = [14, 20]
        for period in periods:
            returns[f"ATR{period}"] = talib.ATR(highest, lowest, closing, timeperiod=period)[-1]
        return returns
