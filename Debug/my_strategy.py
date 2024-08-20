# cython: language_level=3
import datetime
import io
import time

import MetaTrader5 as mt5
import backtrader as bt
import keras
import matplotlib
import numpy as np
import pandas as pd
import pytz
from backtrader import num2date
from matplotlib import pyplot as plt

from Chan import CChan
from Common.CEnum import KL_TYPE, BSP_TYPE, FX_TYPE, DATA_FIELD
from Common.CTime import CTime
from Debug.GenerateTrainData import plot_config, plot_para, config
from KLine.KLine_Unit import CKLine_Unit
from Plot.PlotDriver import CPlotDriver


def initialize_mt5():
    if not mt5.initialize():
        print("初始化失败")
        return False
    return True


def reconnect_mt5(retry_interval=5, max_retries=5):
    retries = 0
    while retries < max_retries:
        print(f"尝试重新连接 MT5 (第 {retries + 1} 次尝试)...")
        if initialize_mt5():
            print("重新连接 MT5 成功")
            return True
        retries += 1
        time.sleep(retry_interval)
    print("重新连接 MT5 失败")
    return False


class CZSCStrategy(bt.Strategy):  # BOLL策略程序
    def __init__(self):  # 初始化
        self.model = keras.models.load_model("./model.h5")
        self.data_close = self.datas[0].close  # 指定价格序列
        # 初始化交易指令、买卖价格和手续费
        code = " "
        self.kl_type = KL_TYPE.K_1H
        # 快照
        self.chan = CChan(
            code=code,
            data_src=None,
            lv_list=[self.kl_type],
            config=config,
        )
        self.order = None
        self.signals = set()
        self.long_orders = []
        self.short_orders = []
        self.fee = 1.0002
        self.money = 10000

    def next(self):  # 买卖策略

        dt = num2date(self.datas[0].datetime[0])
        o = self.datas[0].open[0]
        h = self.datas[0].high[0]
        l = self.datas[0].low[0]
        c = self.datas[0].close[0]
        v = self.datas[0].volume[0]
        fields = "time,open,high,low,close,volume"
        data = [
            dt,
            o,
            h,
            l,
            c,
            v,
        ]

        def str2float(s):
            try:
                return float(s)
            except ValueError:
                return 0.0

        def parse_time_column(inp):
            return CTime(year=inp.year, month=inp.month, day=inp.day, hour=inp.hour, minute=inp.minute, second=0,
                         auto=False)

        def create_item_dict(data, column_name):
            for i in range(len(data)):
                data[i] = parse_time_column(data[i]) if i == 0 else str2float(data[i])
            return dict(zip(column_name, data))

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

        klu = CKLine_Unit(create_item_dict(data, GetColumnNameFromFieldList(fields)))
        self.chan.trigger_load({self.kl_type: [klu]})
        bsp_list = self.chan.get_bsp()
        if not bsp_list:
            return
        last_bsp = bsp_list[-1]
        profit = 0
        if not self.position:
            if last_bsp.klu.time == self.chan[0][-1][-1].time and \
                    (BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
                matplotlib.use('Agg')
                g = CPlotDriver(self.chan, plot_config, plot_para)
                # 移除标题
                for ax in g.figure.axes:
                    ax.set_title("", loc="left")
                    # 移除 x 轴和 y 轴标签
                    ax.set_xlabel('')
                    ax.set_ylabel('')

                    # 移除 x 轴和 y 轴的刻度标签
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # 移除 x 轴和 y 轴的刻度线
                    ax.tick_params(axis='both', which='both', length=0)

                    # 移除网格线
                    ax.grid(False)
                g.figure.tight_layout()
                buf = io.BytesIO()
                g.figure.savefig(buf, format='png')
                plt.close(g.figure)
                buf.seek(0)
                outputs = self.model.predict(np.expand_dims(buf.getvalue(), axis=0))[0]
                if outputs > 0.7:
                    if last_bsp.is_buy and self.chan[0][-2].fx == FX_TYPE.BOTTOM:
                        self.log(f"BUY CREATE, {self.data_close[0]}")
                        self.order = self.buy()
                        self.long_orders.append(self.order)
                    if not last_bsp.is_buy and self.chan[0][-2].fx == FX_TYPE.TOP:
                        self.log(f"SELL CREATE, {self.data_close[0]}")
                        self.order = self.sell()
                        self.short_orders.append(self.order)
        else:
            if len(self.long_orders) > 0:
                # 止盈
                close_price = round(self.data_close[0] / self.fee, 5)
                long_orders_copy = self.long_orders.copy()
                for order in long_orders_copy:
                    long_profit = close_price / order.executed.price - 1
                    tp = long_profit >= 0.003
                    sl = long_profit <= -0.003
                    if tp or sl:
                        profit += round(long_profit * self.money, 2)
                        self.log(f"Close Order, {self.data_close[0]}")
                        self.order = self.close()
                        self.long_orders.remove(order)
                        print(
                            f'{self.chan[0][-1][-1].time}:sell price = {close_price}, profit = {long_profit * self.money:.2f}')

            if len(self.short_orders) > 0:
                close_price = round(self.data_close[0] * self.fee, 5)
                short_orders_copy = self.short_orders.copy()
                for order in short_orders_copy:
                    short_profit = order.executed.price / close_price - 1
                    tp = short_profit >= 0.003
                    sl = short_profit <= -0.003
                    if tp or sl:
                        profit += round(short_profit * self.money, 2)
                        self.log(f"Close Order, {self.data_close[0]}")
                        self.order = self.close()
                        self.short_orders.remove(order)
                        print(
                            f'{self.chan[0][-1][-1].time}:sell price = {close_price}, profit = {short_profit * self.money:.2f}')

    def log(self, txt, dt=None, do_print=False):  # 日志函数
        dt = dt or bt.num2date(self.datas[0].datetime[0])
        print("%s, %s" % (dt, txt))

    def notify_order(self, order):  # 记录交易执行情况
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Rejected:
            self.log(
                f"Rejected : order_ref:{order.ref}  data_name:{order.p.data._name}"
            )

        if order.status == order.Margin:
            self.log(f"Margin : order_ref:{order.ref}  data_name:{order.p.data._name}")

        if order.status == order.Cancelled:
            self.log(
                f"Concelled : order_ref:{order.ref}  data_name:{order.p.data._name}"
            )

        if order.status == order.Partial:
            self.log(f"Partial : order_ref:{order.ref}  data_name:{order.p.data._name}")

        if order.status == order.Completed:
            if order.isbuy():
                self.log(
                    f" BUY : data_name:{order.p.data._name} price : {order.executed.price} , cost : {order.executed.value} , commission : {order.executed.comm}"
                )

            else:  # Sell
                self.log(
                    f" SELL : data_name:{order.p.data._name} price : {order.executed.price} , cost : {order.executed.value} , commission : {order.executed.comm}"
                )

    def notify_trade(self, trade):  # 记录交易收益情况
        if not trade.isclosed:
            return
        self.log(f"策略收益：\n毛收益 {trade.pnl:.5f}, 净收益 {trade.pnlcomm:.5f}")

    def stop(self):  # 回测结束后输出结果
        self.log(
            "期末总资金 %.5f" % (self.broker.getvalue()),
            do_print=True,
        )


if __name__ == "__main__":
    local_tz = pytz.timezone('Asia/Shanghai')
    reconnect_mt5()
    cerebro = bt.Cerebro()
    begin_time = "2021-01-02 00:00:00"
    end_time = "2024-08-01 00:00:00"
    bars = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_H1,
                                datetime.datetime.strptime(begin_time, "%Y-%m-%d %H:%M:%S"),
                                datetime.datetime.strptime(begin_time, "%Y-%m-%d %H:%M:%S"))
    bars = pd.DataFrame(bars)
    bars.dropna(inplace=True)
    bars['time'] = pd.to_datetime(bars['time'], unit='s')
    bars['time'] = bars['time'].dt.tz_localize('Europe/Zurich')
    bars['time'] = bars['time'].dt.tz_convert("Asia/Shanghai")
    bars['time'] = bars['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    bars.rename(columns={'time': 'datetime', 'tick_volume': 'volume'}, inplace=True)
    del bars["real_volume"]
    del bars["spread"]
    bars.index = pd.to_datetime(bars.datetime)
    cerebro.adddata(
        bt.feeds.PandasData(
            dataname=bars, timeframe=bt.TimeFrame.Minutes
        )
    )

    cerebro.addstrategy(CZSCStrategy)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
    # 添加最大回撤分析器
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.0002, leverage=100)
    results = cerebro.run(runonce=True)
    # 提取分析器结果
    trade_analysis = results[0].analyzers.tradeanalyzer.get_analysis()

    win_rate = trade_analysis.won.total / (trade_analysis.won.total + trade_analysis.lost.total) if (
                                                                                                            trade_analysis.won.total + trade_analysis.lost.total) > 0 else 0
    avg_win = trade_analysis.won.pnl.total / trade_analysis.won.total if trade_analysis.won.total > 0 else 0
    avg_loss = trade_analysis.lost.pnl.total / trade_analysis.lost.total if trade_analysis.lost.total > 0 else 0
    profit_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')
    # 获取最终的投资组合价值
    final_value = cerebro.broker.getvalue()

    # 打印最终的投资组合价值
    print(f"Final Portfolio Value: {final_value:.5f}")
    print(f"Win Rate: {win_rate * 100:.5f}%")
    print(f"Average Win: {avg_win:.5f}")
    print(f"Average Loss: {avg_loss:.5f}")
    print(f"Profit/Loss Ratio: {profit_loss_ratio:.5f}")
    # 获取最大回撤分析器结果
    drawdown = results[0].analyzers.drawdown.get_analysis()

    # 打印最大回撤信息
    print(f"最大回撤比例: {drawdown.max.drawdown:.5f}%")
    print(f"最大回撤金额: {drawdown.max.moneydown:.5f}")

    print(f"回撤持续时间: {drawdown.max.len}")
    print('最终资金: %.5f' % cerebro.broker.getvalue())
    cerebro.plot()
