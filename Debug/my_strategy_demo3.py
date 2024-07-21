# cython: language_level=3
import datetime
import sys
from typing import List

import numpy as np
from empyrical import max_drawdown
from matplotlib import pyplot as plt

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_FIELD, DATA_SRC, KL_TYPE, BSP_TYPE
from DataAPI.MT5ForexAPI import CMT5ForexAPI
from KLine.KLine_Unit import CKLine_Unit

sys.setrecursionlimit(1000000)


def combine_higher_klu_from_lower(klu_lower_lst: List[CKLine_Unit]) -> CKLine_Unit:
    return CKLine_Unit(
        {
            DATA_FIELD.FIELD_TIME: klu_lower_lst[-1].time,
            DATA_FIELD.FIELD_OPEN: klu_lower_lst[0].open,
            DATA_FIELD.FIELD_CLOSE: klu_lower_lst[-1].close,
            DATA_FIELD.FIELD_HIGH: max(klu.high for klu in klu_lower_lst),
            DATA_FIELD.FIELD_LOW: min(klu.low for klu in klu_lower_lst),
        }
    )


def main():
    """
    代码不能直接跑，仅用于展示如何实现小级别K线更新直接刷新CChan结果
    """
    code = "EURGBP"
    begin_time = "2021-01-01 00:00:00"
    end_time = "2024-07-10 00:00:00"
    data_src_type = DATA_SRC.FOREX
    lower_kl_type = KL_TYPE.K_1M
    higher_kl_type = KL_TYPE.K_5M
    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "divergence_rate": 0.8,
        "min_zs_cnt": 1,
        "kl_data_check": False,
    })

    # 快照
    chan_lower = CChan(
        code=code,
        data_src=data_src_type,
        lv_list=[lower_kl_type],
        config=config,
    )
    chan_higher = CChan(
        code=code,
        data_src=data_src_type,
        lv_list=[higher_kl_type],
        config=config,
    )

    lower_data_src = CMT5ForexAPI(code, k_type=lower_kl_type, begin_date=begin_time, end_date=end_time,
                                  autype=AUTYPE.NONE)  # 获取最小级别
    higher_data_src = CMT5ForexAPI(code, k_type=higher_kl_type, begin_date=begin_time, end_date=end_time,
                                   autype=AUTYPE.NONE)  # 获取最小级别

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
    lower_last_bsp = None
    higher_last_bsp = None
    higher_data = higher_data_src.get_kl_data()
    klu_higher = next(higher_data)
    for klu_lower in lower_data_src.get_kl_data():  # 获取单根1分钟K线
        now = datetime.datetime.now()
        chan_lower.trigger_load({lower_kl_type: [klu_lower]})
        lower_lv_chan = chan_lower[0]  # 低级别K线

        while klu_lower.time.ts >= klu_higher.time.ts:
            chan_higher.trigger_load({higher_kl_type: [klu_higher]})
            klu_higher = next(higher_data)
        higher_lv_chan = chan_higher[0]  # 高级别K线

        """
        策略开始：
        这里基于chan实现你的策略
        """
        profit = 0
        higher_bsp_list = chan_higher.get_bsp(0)  # 获取买卖点列表
        lower_bsp_list = chan_lower.get_bsp(0)  # 获取买卖点列表
        if higher_bsp_list:
            higher_last_bsp = higher_bsp_list[-1]
        if lower_bsp_list:
            lower_last_bsp = lower_bsp_list[-1]

        higher_bsp_rule = higher_last_bsp and (
                BSP_TYPE.T1 in higher_last_bsp.type or BSP_TYPE.T1P in higher_last_bsp.type)

        lower_bsp_rule = lower_last_bsp and (
                BSP_TYPE.T1 in lower_last_bsp.type or BSP_TYPE.T1P in lower_last_bsp.type)

        if len(long_orders) > 0:

            rule = higher_bsp_rule and not higher_last_bsp.is_buy
            if rule:
                close_price = round(lower_lv_chan[-1][-1].close / fee, 5)
                for order in long_orders:
                    long_profit = close_price / order - 1
                    print(
                        f'{lower_lv_chan[-1][-1].time}:sell price = {close_price}, profit = {long_profit * money:.2f}')
                    profit += round(long_profit * money, 2)
                    history_long_orders += 1
                long_orders.clear()
            else:
                # 止盈
                close_price = round(lower_lv_chan[-1][-1].close / fee, 5)
                long_orders_copy = long_orders.copy()
                for order in long_orders_copy:
                    long_profit = close_price / order - 1
                    tp = long_profit >= 0.01
                    sl = long_profit <= -0.01
                    if tp or sl:
                        long_orders.remove(order)
                        profit += round(long_profit * money, 2)
                        print(
                            f'{lower_lv_chan[-1][-1].time}:sell price = {close_price}, profit = {long_profit * money:.2f}')
                        history_long_orders += 1

        if len(short_orders) > 0:
            rule = higher_bsp_rule and higher_last_bsp.is_buy
            if rule:
                close_price = round(lower_lv_chan[-1][-1].close * fee, 5)
                for order in short_orders:
                    short_profit = order / close_price - 1
                    profit += round(short_profit * money, 2)
                    print(
                        f'{lower_lv_chan[-1][-1].time}:sell price = {close_price}, profit = {short_profit * money:.2f}')
                    history_short_orders += 1
                short_orders.clear()
            else:
                close_price = round(lower_lv_chan[-1][-1].close * fee, 5)
                short_orders_copy = short_orders.copy()
                for order in short_orders_copy:
                    short_profit = order / close_price - 1
                    tp = short_profit >= 0.01
                    sl = short_profit <= -0.01
                    if tp or sl:
                        short_orders.remove(order)
                        profit += round(short_profit * money, 2)
                        print(
                            f'{lower_lv_chan[-1][-1].time}:sell price = {close_price}, profit = {short_profit * money:.2f}')
                        history_short_orders += 1

        if len(long_orders) == 0:
            if lower_bsp_rule and higher_bsp_rule and lower_last_bsp.is_buy and higher_last_bsp.is_buy:
                long_orders.append(round(lower_lv_chan[-1][-1].close * fee, 5))
                print(f'{lower_lv_chan[-1][-1].time}:buy long price = {long_orders[-1]}')

        if len(short_orders) == 0:
            if lower_bsp_rule and higher_bsp_rule and not lower_last_bsp.is_buy and not higher_last_bsp.is_buy:
                short_orders.append(round(lower_lv_chan[-1][-1].close / fee, 5))
                print(f'{lower_lv_chan[-1][-1].time}:buy short price = {short_orders[-1]}')
        capital += profit
        if profit != 0:
            profits = np.append(profits, profit)
            capitals = np.append(capitals, capital)
            print(f"capital:{capital}")
            win_rate = len(profits[profits > 0]) / len(profits) * 100
            print(f"胜率: {win_rate}")

        # print(f"{klu_lower.time}:{datetime.datetime.now() - now}")

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
