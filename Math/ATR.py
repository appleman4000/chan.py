import numpy as np
import talib


class ATR:
    def __init__(self, n=200, period: int = 20):
        super(ATR, self).__init__()
        self.arr = []
        self.period = period
        self.n = n

    def add(self, high, low, close) -> dict:
        self.arr.append([high, low, close])

        if len(self.arr) > self.n:
            del self.arr[0]
        highest = np.array(self.arr)[:, 0].astype(np.float64)
        lowest = np.array(self.arr)[:, 1].astype(np.float64)
        closing = np.array(self.arr)[:, 2].astype(np.float64)
        atr = talib.ATR(highest, lowest, closing, timeperiod=self.period)[-1]
        return atr
