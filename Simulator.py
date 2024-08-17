# cython: language_level=3
import datetime
import sys
from multiprocessing import Process
from multiprocessing import Value

import MetaTrader5 as mt5
import numpy as np
import pytz

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, KL_TYPE
from Messenger import send_message

sys.setrecursionlimit(1000000)
app_id = 'cli_a63ae160c79d500b'
app_secret = 'BvtLvfCEPEePrqdw4vddScwhKVWSCtAx'
webhook_url = 'https://open.feishu.cn/open-apis/bot/v2/hook/d8ea8601-259b-4f48-a310-1e715909232e'
symbols = [
    # Major
    "EURUSD",
    "GBPUSD",
    "AUDUSD",
    "NZDUSD",
    "USDJPY",
    "USDCAD",
    "USDCHF",
    # Crosses
    "AUDCHF",
    "AUDJPY",
    "AUDNZD",
    "CADCHF",
    "CADJPY",
    "CHFJPY",
    "EURAUD",
    "EURCAD",
    "AUDCAD",
    "EURCHF",
    "GBPNZD",
    "GBPCAD",
    "GBPCHF",
    "GBPJPY",
    "USDCNH",
    "XAUUSD",
    "XAGUSD",
]
period_map = {
    mt5.TIMEFRAME_M1: KL_TYPE.K_1M,
    mt5.TIMEFRAME_M3: KL_TYPE.K_3M,
    mt5.TIMEFRAME_M5: KL_TYPE.K_5M,
    mt5.TIMEFRAME_M10: KL_TYPE.K_10M,
    mt5.TIMEFRAME_M15: KL_TYPE.K_15M,
    mt5.TIMEFRAME_M30: KL_TYPE.K_30M,
    mt5.TIMEFRAME_H1: KL_TYPE.K_1H,
    mt5.TIMEFRAME_H2: KL_TYPE.K_2H,
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


def period_seconds(period):
    return timeframe_seconds[period]


def robot_trade(symbol, lot=0.01, is_buy=None, comment=""):
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        print(f"Currency pair {symbol} has open positions.")
        return
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(symbol, "not found, can not call order_check()")
        return
    # if the symbol is unavailable in MarketWatch, add it
    if not symbol_info.visible:
        print(symbol, "is not visible, trying to switch on")
        if not mt5.symbol_select(symbol, True):
            print("symbol_select({}}) failed, exit", symbol)
            return
    for tries in range(10):
        point = mt5.symbol_info(symbol).point
        price = mt5.symbol_info_tick(symbol).ask if is_buy else mt5.symbol_info_tick(symbol).bid
        tp = round(price * 0.003 / point)
        sl = round(price * 0.003 / point)
        deviation = 30
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": price - sl * point if is_buy else price + sl * point,
            "tp": price + tp * point if is_buy else price - tp * point,
            "deviation": deviation,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_DAY,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        # send a trading request
        result = mt5.order_send(request)
        if result is None:
            return
        if result.retcode != mt5.TRADE_RETCODE_REQUOTE and result.retcode != mt5.TRADE_RETCODE_PRICE_OFF:
            if is_buy:
                print(f"{symbol} buy order placed successfully")
            else:
                print(f"{symbol} sell order placed successfully")
            break


def shanghai_to_zurich_datetime(timestamp):
    zurich_tz = pytz.timezone('Europe/Zurich')
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    timestamp = shanghai_tz.localize(datetime.datetime.fromtimestamp(timestamp))

    # 将时间戳转换到 'Asia/Shanghai' 时区
    zurich_time = timestamp.astimezone(zurich_tz)

    # 格式化时间
    formatted_time = zurich_time.strftime('%Y-%m-%d %H:%M')

    return formatted_time


def strategy(code, global_profit):
    data_src_type = DATA_SRC.FOREX_ONLINE
    bottom_kl_type = mt5.TIMEFRAME_M3
    top_kl_type = mt5.TIMEFRAME_M30
    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "skip_step": 500,
        "divergence_rate": float("inf"),
        "min_zs_cnt": 1,
        "macd_algo": "slope",
        "kl_data_check": False,
        "bi_end_is_peak": True,
        "bsp1_only_multibi_zs": True,
        "max_bs2_rate": 0.999,
        "bs1_peak": True,
        "bs_type": "1,1p,2,2s,3a,3b",
        "bsp2_follow_1": True,
        "bsp3_follow_1": True,
        "bsp3_peak": True,
        "bsp2s_follow_2": True,
        "max_bsp2s_lv": None,
        "strict_bsp3": True,
    })
    end_date = datetime.datetime.now()
    end_date = end_date.timestamp()
    end_date -= end_date % timeframe_seconds[top_kl_type]
    end_date -= timeframe_seconds[top_kl_type]
    end_date = datetime.datetime.fromtimestamp(end_date)
    # 快照
    chan = CChan(
        code=code,
        data_src=data_src_type,
        lv_list=[period_map[top_kl_type], period_map[bottom_kl_type]],
        config=config,
        begin_time=None,
        end_time=end_date
    )

    capital = 10000
    lots = 0.1
    money = 100000 * lots
    capitals = np.array([])
    profits = np.array([])
    fee = 1.0004
    long_orders = []
    short_orders = []
    history_long_orders = 0
    history_short_orders = 0
    for id, chan_snapshot in enumerate(chan.step_load()):

        top_lv_chan = chan_snapshot[0]
        bottom_lv_chan = chan_snapshot[1]
        # print(datetime.datetime.now())
        # print(
        #     f"北京时间:{top_lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(top_lv_chan[-1][-1].time.ts)}  on_bar {code}")
        # print(
        #     f"北京时间:{middle_lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(middle_lv_chan[-1][-1].time.ts)}  on_bar {code}")
        # print(
        #     f"北京时间:{bottom_lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(bottom_lv_chan[-1][-1].time.ts)}  on_bar {code}")
        """
        策略开始：
        这里基于chan实现你的策略
        """
        profit = 0
        top_bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not top_bsp_list:
            continue
        top_last_bsp = top_bsp_list[-1]
        bottom_bsp_list = chan.get_bsp(1)  # 获取买卖点列表
        if not bottom_bsp_list:
            continue
        bottom_last_bsp = bottom_bsp_list[-1]
        top_entry_rule = top_lv_chan[-1].idx == top_last_bsp.klu.klc.idx
        botton_entry_rule = bottom_lv_chan[-1].idx == bottom_last_bsp.klu.klc.idx

        if len(long_orders) > 0:
            # 止盈
            close_price = round(bottom_lv_chan[-1][-1].close / fee, 5)
            long_orders_copy = long_orders.copy()
            for order in long_orders_copy:
                long_profit = close_price / order - 1
                tp = long_profit >= 0.01
                sl = long_profit <= -0.01
                if top_entry_rule and not top_last_bsp.is_buy and botton_entry_rule and not bottom_last_bsp.is_buy:
                    long_orders.remove(order)
                    profit += round(long_profit * money, 2)
                    print(
                        f'{code} {bottom_lv_chan[-1][-1].time}:sell price = {close_price}, profit = {long_profit * money:.2f}')
                    history_long_orders += 1

        if len(short_orders) > 0:
            close_price = round(bottom_lv_chan[-1][-1].close * fee, 5)
            short_orders_copy = short_orders.copy()
            for order in short_orders_copy:
                short_profit = order / close_price - 1
                tp = short_profit >= 0.01
                sl = short_profit <= -0.01
                if top_entry_rule and top_last_bsp.is_buy and botton_entry_rule and bottom_last_bsp.is_buy:
                    short_orders.remove(order)
                    profit += round(short_profit * money, 2)
                    print(
                        f'{code} {bottom_lv_chan[-1][-1].time}:sell price = {close_price}, profit = {short_profit * money:.2f}')
                    history_short_orders += 1

        if len(long_orders) == 0 and len(short_orders) == 0:
            if top_entry_rule and botton_entry_rule and top_last_bsp.is_buy and bottom_last_bsp.is_buy:
                long_orders.append(round(bottom_lv_chan[-1][-1].close * fee, 5))
                print(f'{code} {bottom_lv_chan[-1][-1].time}:buy long price = {long_orders[-1]}')
                if len(profits) > 0:
                    win_rate = len(profits[profits > 0]) / len(profits) * 100
                    price = f"{bottom_lv_chan[-1][-1].close:.5f}".rstrip('0').rstrip('.')
                    subject = f"{code} 胜率: {win_rate:.1f}% {'买点' if top_last_bsp.is_buy else '卖点'} {price}"
                    message = f"北京时间:{bottom_lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(bottom_lv_chan[-1][-1].time.ts)}"
                else:
                    price = f"{bottom_lv_chan[-1][-1].close:.5f}".rstrip('0').rstrip('.')
                    subject = f"{code} {'买点' if top_last_bsp.is_buy else '卖点'} {price}"
                    message = f"北京时间:{bottom_lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(bottom_lv_chan[-1][-1].time.ts)}"
                # send_message(app_id, app_secret, webhook_url, subject, message, [chan])

        if len(short_orders) == 0 and len(long_orders) == 0:
            if top_entry_rule and botton_entry_rule and not top_last_bsp.is_buy and not bottom_last_bsp.is_buy:
                short_orders.append(round(bottom_lv_chan[-1][-1].close / fee, 5))
                print(f'{code} {bottom_lv_chan[-1][-1].time}:buy short price = {short_orders[-1]}')
                if len(profits) > 0:
                    win_rate = len(profits[profits > 0]) / len(profits) * 100
                    price = f"{bottom_lv_chan[-1][-1].close:.5f}".rstrip('0').rstrip('.')
                    subject = f"{code} 胜率: {win_rate:.1f}% {'买点' if top_last_bsp.is_buy else '卖点'} {price}"
                    message = f"北京时间:{bottom_lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(bottom_lv_chan[-1][-1].time.ts)}"
                else:
                    price = f"{bottom_lv_chan[-1][-1].close:.5f}".rstrip('0').rstrip('.')
                    subject = f"{code} {'买点' if top_last_bsp.is_buy else '卖点'} {price}"
                    message = f"北京时间:{bottom_lv_chan[-1][-1].time} 瑞士时间:{shanghai_to_zurich_datetime(bottom_lv_chan[-1][-1].time.ts)}"
                # send_message(app_id, app_secret, webhook_url, subject, message, [chan])

        capital += profit
        if profit != 0:
            profits = np.append(profits, profit)
            with global_profit.get_lock():  # 使用锁来确保操作的原子性
                global_profit.value += profit

            capitals = np.append(capitals, capital)
            print(f"{code} capital:{capital}")
            win_rate = len(profits[profits > 0]) / len(profits) * 100
            print(f"{code} 胜率: {win_rate}")
            print(f"global_profit:{global_profit.value}")


if __name__ == "__main__":
    global_profit = Value('f', 0)
    processes = []
    for symbol in symbols:
        process = Process(target=strategy, args=(symbol, global_profit))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
