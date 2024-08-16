# cython: language_level=3
import sys
from typing import List

import MetaTrader5 as mt5
import numpy as np
from empyrical import max_drawdown
from matplotlib import pyplot as plt

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_FIELD, DATA_SRC, KL_TYPE
from DataAPI.MT5ForexAPI import CMT5ForexAPI
from KLine.KLine_Unit import CKLine_Unit

sys.setrecursionlimit(1000000)
period_map = {
    mt5.TIMEFRAME_M1: KL_TYPE.K_1M,
    mt5.TIMEFRAME_M3: KL_TYPE.K_3M,
    mt5.TIMEFRAME_M5: KL_TYPE.K_5M,
    mt5.TIMEFRAME_M15: KL_TYPE.K_15M,
    mt5.TIMEFRAME_M30: KL_TYPE.K_30M,
    mt5.TIMEFRAME_H1: KL_TYPE.K_1H,
    mt5.TIMEFRAME_H4: KL_TYPE.K_4H,
    mt5.TIMEFRAME_D1: KL_TYPE.K_DAY,

}
timeframe_seconds = {
    mt5.TIMEFRAME_M1: 60,
    mt5.TIMEFRAME_M2: 120,
    mt5.TIMEFRAME_M3: 180,
    mt5.TIMEFRAME_M4: 240,
    mt5.TIMEFRAME_M5: 300,
    mt5.TIMEFRAME_M6: 360,
    mt5.TIMEFRAME_M10: 600,
    mt5.TIMEFRAME_M12: 720,
    mt5.TIMEFRAME_M15: 900,
    mt5.TIMEFRAME_M20: 1200,
    mt5.TIMEFRAME_M30: 1800,
    mt5.TIMEFRAME_H1: 3600,
    mt5.TIMEFRAME_H2: 7200,
    mt5.TIMEFRAME_H3: 10800,
    mt5.TIMEFRAME_H4: 14400,
    mt5.TIMEFRAME_H6: 21600,
    mt5.TIMEFRAME_H8: 28800,
    mt5.TIMEFRAME_H12: 43200,
    mt5.TIMEFRAME_D1: 86400,
    mt5.TIMEFRAME_W1: 604800,
    mt5.TIMEFRAME_MN1: 2592000  # 大约的月时间秒数，具体月的秒数会有所不同
}


def combine_middle_klu_from_bottom(klu_bottom_lst: List[CKLine_Unit]) -> CKLine_Unit:
    return CKLine_Unit(
        {
            DATA_FIELD.FIELD_TIME: klu_bottom_lst[-1].time,
            DATA_FIELD.FIELD_OPEN: klu_bottom_lst[0].open,
            DATA_FIELD.FIELD_CLOSE: klu_bottom_lst[-1].close,
            DATA_FIELD.FIELD_HIGH: max(klu.high for klu in klu_bottom_lst),
            DATA_FIELD.FIELD_LOW: min(klu.low for klu in klu_bottom_lst),
        }
    )


def main():
    """
    代码不能直接跑，仅用于展示如何实现小级别K线更新直接刷新CChan结果
    """
    code = "EURUSD"
    begin_time = "2024-01-01 00:00:00"
    end_time = "2024-07-10 00:00:00"
    data_src_type = DATA_SRC.FOREX_ONLINE
    bottom_kl_type = mt5.TIMEFRAME_M1
    middle_kl_type = mt5.TIMEFRAME_M5
    top_kl_type = mt5.TIMEFRAME_M15
    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "skip_step": 200,
        "divergence_rate": 1.0,
        "min_zs_cnt": 0,
        "macd_algo": "slope",
        "kl_data_check": False,
        "bi_end_is_peak": False,
        "bsp2_follow_1": True,
        "bsp3_follow_1": True,
        "bs_type": '1,1p,2,2s,3a,3b',
    })

    # 快照
    chan = CChan(
        code=code,
        data_src=data_src_type,
        lv_list=[period_map[top_kl_type], period_map[middle_kl_type], period_map[bottom_kl_type]],
        config=config,
        begin_time=begin_time,
        end_time=end_time,

    )

    capital = 10000
    lots = 0.1
    money = 100000 * lots
    capitals = np.array([])
    profits = np.array([])
    fee = 1.0002
    long_orders = []
    short_orders = []
    history_long_orders = 0
    history_short_orders = 0
    for chan_snapshot in chan.step_load():
        top_lv_chan = chan_snapshot[0]
        middle_lv_chan = chan_snapshot[1]
        bottom_lv_chan = chan_snapshot[2]

        """
        策略开始：
        这里基于chan实现你的策略
        """
        profit = 0
        top_bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not top_bsp_list:
            continue
        top_last_bsp = top_bsp_list[-1]
        middle_bsp_list = chan.get_bsp(1)  # 获取买卖点列表
        if not middle_bsp_list:
            continue
        middle_last_bsp = middle_bsp_list[-1]
        bottom_bsp_list = chan.get_bsp(2)  # 获取买卖点列表
        if not bottom_bsp_list:
            continue
        bottom_last_bsp = bottom_bsp_list[-1]
        entry_rule = top_lv_chan[-1].idx == top_last_bsp.klu.klc.idx
        entry_rule = entry_rule and middle_lv_chan[-1].idx == middle_last_bsp.klu.klc.idx
        entry_rule = entry_rule and bottom_lv_chan[-1].idx == bottom_last_bsp.klu.klc.idx

        if len(long_orders) > 0:
            # 止盈
            close_price = round(bottom_lv_chan[-1][-1].close / fee, 5)
            long_orders_copy = long_orders.copy()
            for order in long_orders_copy:
                long_profit = close_price / order - 1
                tp = long_profit >= 0.002
                sl = long_profit <= -0.002
                if tp or sl:
                    long_orders.remove(order)
                    profit += round(long_profit * money, 2)
                    print(
                        f'{bottom_lv_chan[-1][-1].time}:sell price = {close_price}, profit = {long_profit * money:.2f}')
                    history_long_orders += 1

        if len(short_orders) > 0:
            close_price = round(bottom_lv_chan[-1][-1].close * fee, 5)
            short_orders_copy = short_orders.copy()
            for order in short_orders_copy:
                short_profit = order / close_price - 1
                tp = short_profit >= 0.002
                sl = short_profit <= -0.002
                if tp or sl:
                    short_orders.remove(order)
                    profit += round(short_profit * money, 2)
                    print(
                        f'{bottom_lv_chan[-1][-1].time}:sell price = {close_price}, profit = {short_profit * money:.2f}')
                    history_short_orders += 1

        if len(long_orders) == 0 and len(short_orders) == 0:
            if entry_rule and top_last_bsp.is_buy and middle_last_bsp.is_buy and bottom_last_bsp.is_buy:
                long_orders.append(round(bottom_lv_chan[-1][-1].close * fee, 5))
                print(f'{bottom_lv_chan[-1][-1].time}:buy long price = {long_orders[-1]}')

        if len(short_orders) == 0 and len(long_orders) == 0:
            if entry_rule and not top_last_bsp.is_buy and not middle_last_bsp.is_buy and not bottom_last_bsp.is_buy:
                short_orders.append(round(bottom_lv_chan[-1][-1].close / fee, 5))
                print(f'{bottom_lv_chan[-1][-1].time}:buy short price = {short_orders[-1]}')
        capital += profit
        if profit != 0:
            profits = np.append(profits, profit)
            capitals = np.append(capitals, capital)
            print(f"capital:{capital}")
            win_rate = len(profits[profits > 0]) / len(profits) * 100
            print(f"胜率: {win_rate}")
    CMT5ForexAPI.do_close()
    # 最后显示总盈利历史图表
    plt.rcParams['font.sans-serif'] = ['simhei']
    plt.rcParams['axes.unicode_minus'] = False
    win_rate = len(profits[profits > 0]) / (len(profits[profits > 0]) + len(profits[profits < 0])) * 100
    print(f"胜率: {win_rate}")
    win_loss_radio = profits[profits > 0].mean() / -profits[profits < 0].mean()
    print(f"盈亏比 {win_loss_radio}")
    # annualization = 12 * 4 * 5 * 24 * 4
    # sharperatio = sharpe_ratio(
    #     capitals[1:] / capitals[:-1] - 1,
    #     risk_free=0.0,
    #     period=None,
    #     annualization=annualization,
    # )
    # print(f"夏普率 {sharperatio}")
    maxdrawdown = - max_drawdown(capitals[1:] / capitals[:-1] - 1) * 100
    print(f"最大回撤 {maxdrawdown}")
    plt.suptitle(
        "%s 资金:%.2f 胜率:%.2f%% 盈亏比:%.2f 最大回撤:%.2f%% 多(%d) 空(%d)"
        % (
            code,
            round(capitals[-1], 2),
            win_rate,
            win_loss_radio,
            maxdrawdown,
            history_long_orders,
            history_short_orders,
        ),
        fontsize=9,
    )
    plt.plot(range(len(capitals)), capitals)
    plt.show()


if __name__ == "__main__":
    main()
