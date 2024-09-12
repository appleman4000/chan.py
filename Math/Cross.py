# cython: language_level=3
# encoding:utf-8
import numpy as np

import talib as ta


class CrossIndicators:
    def __init__(self, T=100):
        self.T = T
        self.arr = []

    def add(self, value) -> float:
        if len(self.arr) == 0:
            self.arr = [value] * self.T
        self.arr.append(value)

        if len(self.arr) > self.T:
            self.arr = self.arr[-self.T:]
        periods = np.arange(8, 31)
        returns = dict()
        for period in periods:
            closing = np.array(self.arr, dtype=np.float64)
            ema = ta.EMA(closing, timeperiod=period)
            if closing[-2] < ema[-2] and closing[-1] >= ema[-1]:
                returns[f"Cross{period}"] = 1
            elif closing[-2] > ema[-2] and closing[-1] <= ema[-1]:
                returns[f"Cross{period}"] = -1
            else:
                returns[f"Cross{period}"] = 0
        return returns
