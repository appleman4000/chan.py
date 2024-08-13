# encoding:utf-8
from Chan import CChan
from Common.CEnum import FX_TYPE, BI_DIR, MACD_ALGO

N = 1


def open_klu_rate(chan: CChan):
    last_klu = chan[0][-1][-1]
    return {
        "open_klu_rate": (last_klu.close - last_klu.open) / last_klu.open,
    }


# def last_bsp_type(chan: CChan):
#     bsp_list = chan.get_bsp()
#     if not bsp_list:
#         return {"last_bsp_type": None}
#     last_bsp = bsp_list[-1]
#     return {"last_bsp_type": list(BSP_TYPE).index(last_bsp.type[0])}


def last_fx(chan: CChan):
    fx = chan[0][-1].fx
    return {"last_fx": list(FX_TYPE).index(fx)}


def last_bi_dir(chan: CChan):
    returns = dict()
    for i in range(1, N):
        bi = chan[0].bi_list[-i]
        returns[f"last_bi_dir{i}"] = list(BI_DIR).index(bi.dir)
    return returns


def last_bi_is_sure(chan: CChan):
    returns = dict()
    for i in range(1, N):
        bi = chan[0].bi_list[-i]
        returns[f"last_bi_is_sure{i}"] = int(bi.is_sure)
    return returns


def last_bi_slope(chan: CChan):
    returns = dict()
    for i in range(1, N):
        bi = chan[0].bi_list[-i]
        slope = (bi.get_end_val() - bi.get_begin_val()) / bi.get_begin_val()
        returns[f"last_bi_slope{i}"] = slope
    return returns


def last_bi_macd_metric(chan: CChan):
    returns = dict()
    for i in range(1, N):
        bi = chan[0].bi_list[-i]
        metric = bi.cal_macd_metric(MACD_ALGO.DIFF, is_reverse=False)
        returns[f"last_bi_macd_metric{i}"] = metric
    return returns
