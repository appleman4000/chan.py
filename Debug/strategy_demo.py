# cython: language_level=3
from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, FX_TYPE, KL_TYPE

if __name__ == "__main__":
    """
    一个极其弱智的策略，只交易一类买卖点，底分型形成后就开仓，直到一类卖点顶分型形成后平仓
    只用做展示如何自己实现策略，做回测用~
    """
    code = "EURUSD"
    begin_time = "2021-01-01 00:00:00"
    end_time = "2024-01-01 00:00:00"
    data_src = DATA_SRC.FOREX
    lv_list = [KL_TYPE.K_1H]

    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "skip_step": 500,
        "divergence_rate": 1.0,
        "min_zs_cnt": 0,
        "macd_algo": "slope",
        "kl_data_check": False,
        "bi_end_is_peak": True,
        "bsp2_follow_1": True,
        "bsp3_follow_1": True,
        "bs_type": '1,1p,2,2s,3a,3b',
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

    is_hold = False
    last_buy_price = None
    profit = 0
    for chan_snapshot in chan.step_load():  # 每增加一根K线，返回当前静态精算结果
        bsp_list = chan_snapshot.get_bsp()  # 获取买卖点列表
        if not bsp_list:  # 为空
            continue
        last_bsp = bsp_list[-1]  # 最后一个买卖点
        if BSP_TYPE.T1 not in last_bsp.type and BSP_TYPE.T1P not in last_bsp.type:  # 假如只做1类买卖点
            continue
        cur_lv_chan = chan_snapshot[0]
        if last_bsp.klu.klc.idx != cur_lv_chan[-2].idx:
            continue
        if cur_lv_chan[-2].fx == FX_TYPE.BOTTOM and last_bsp.is_buy and not is_hold:  # 底分型形成后开仓
            last_buy_price = cur_lv_chan[-1][-1].close  # 开仓价格为最后一根K线close
            print(f'{cur_lv_chan[-1][-1].time}:buy price = {last_buy_price}')
            is_hold = True
        elif cur_lv_chan[-2].fx == FX_TYPE.TOP and not last_bsp.is_buy and is_hold:  # 顶分型形成后平仓
            sell_price = cur_lv_chan[-1][-1].close
            profit += (sell_price - last_buy_price) / last_buy_price * 100
            print(f'{cur_lv_chan[-1][-1].time}:sell price = {sell_price}, profit rate = {profit:.2f}%')
            is_hold = False
