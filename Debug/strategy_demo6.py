# cython: language_level=3
import json
import pickle
from typing import Dict, TypedDict

import numpy as np
import xgboost as xgb
from empyrical import max_drawdown
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix

from BuySellPoint.BS_Point import CBS_Point
from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, BSP_TYPE
from Common.CTime import CTime
from DataAPI.MT5ForexAPI import CMT5ForexAPI
from Debug.FeatureEngineering import FeatureFactors
from strategy_demo5 import get_factors


class T_SAMPLE_INFO(TypedDict):
    feature: CFeatures
    is_buy: bool
    open_time: CTime


def predict_bsp(model, last_bsp: CBS_Point, meta: Dict[str, int]):
    missing = -9999999
    feature_arr = [missing] * len(meta)
    for feat_name, feat_value in last_bsp.features.items():
        if feat_name in meta:
            feature_arr[meta[feat_name]] = feat_value
    feature_arr = [feature_arr]
    dtest = xgb.DMatrix(feature_arr, missing=missing)
    return model.predict_proba(csr_matrix(dtest.get_data()).toarray())[0][1]
    # return model.predict(dtest)


if __name__ == "__main__":
    """
    本demo主要演示如何在实盘中把策略产出的买卖点，对接到demo5中训练好的离线模型上
    """
    code = "EURUSD"
    begin_time = "2021-07-01 00:00:00"
    end_time = "2024-07-10 00:00:00"
    data_src = DATA_SRC.FOREX
    lv_list = [KL_TYPE.K_1H]

    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "bi_strict": True,
        "skip_step": 500,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
    })

    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )

    # model = xgb.Booster()
    # model.load_model("model.json")
    # meta = json.load(open("feature.meta", "r"))
    # # 打开文件以二进制读模式
    with open("model.hdf5", 'rb') as file:
        # 使用 pickle.load 加载对象
        model = pickle.load(file)
    meta = json.load(open("feature.meta", "r"))
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
    treated_bsp_idx = set()
    for chan_snapshot in chan.step_load():
        # 策略逻辑要对齐demo5
        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        cur_lv_chan = chan_snapshot[0]
        profit = 0
        if len(long_orders) > 0:
            # 止盈
            close_price = round(cur_lv_chan[-1][-1].close / fee, 5)
            long_orders_copy = long_orders.copy()
            for order in long_orders_copy:
                long_profit = close_price / order - 1
                tp = long_profit >= 0.003
                sl = long_profit <= -0.003
                if tp or sl:
                    long_orders.remove(order)
                    profit = round(long_profit * money, 2)
                    print(
                        f'{cur_lv_chan[-1][-1].time}:sell price = {close_price}, profit = {long_profit * money:.2f}')
                    history_long_orders += 1

        if len(short_orders) > 0:
            close_price = round(cur_lv_chan[-1][-1].close * fee, 5)
            short_orders_copy = short_orders.copy()
            for order in short_orders_copy:
                short_profit = order / close_price - 1
                tp = short_profit >= 0.003
                sl = short_profit <= -0.003
                if tp or sl:
                    short_orders.remove(order)
                    profit = round(short_profit * money, 2)
                    print(
                        f'{cur_lv_chan[-1][-1].time}:sell price = {close_price}, profit = {short_profit * money:.2f}')
                    history_short_orders += 1

        if len(long_orders) <= 0 and len(short_orders) <= 0:
            if last_bsp.klu.idx not in treated_bsp_idx and last_bsp.klu.time == cur_lv_chan[-1][-1].time and \
                    (BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
                factors = get_factors(FeatureFactors(chan))
                for key in factors.keys():
                    last_bsp.features.add_feat(key, factors[key])
                    # 买卖点打分，应该和demo5最后的predict结果完全一致才对
                # print(last_bsp.klu.time, predict_bsp(model, last_bsp, meta))
                value = predict_bsp(model, last_bsp, meta)
                treated_bsp_idx.add(last_bsp.klu.idx)
                if last_bsp.is_buy and value > 0.63:
                    long_orders.append(round(cur_lv_chan[-1][-1].close * fee, 5))
                    print(f'{cur_lv_chan[-1][-1].time}:buy long price = {long_orders[-1]}')

                if not last_bsp.is_buy and value > 0.63:
                    short_orders.append(round(cur_lv_chan[-1][-1].close / fee, 5))
                    print(f'{cur_lv_chan[-1][-1].time}:buy short price = {short_orders[-1]}')

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
