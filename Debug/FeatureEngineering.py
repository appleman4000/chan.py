# encoding:utf-8
from Chan import CChan
from Common.CEnum import FX_TYPE, BI_DIR, MACD_ALGO

BI_N = 6
ZS_N = 2


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

# 笔
def last_fx(chan: CChan):
    fx = chan[0][-1].fx
    return {"last_fx": list(FX_TYPE).index(fx)}


def last_bi_dir(chan: CChan):
    returns = dict()
    for i in range(1, BI_N):
        bi = chan[0].bi_list[-i]
        returns[f"last_bi_dir{i}"] = list(BI_DIR).index(bi.dir)
    return returns


def last_bi_is_sure(chan: CChan):
    returns = dict()
    for i in range(1, BI_N):
        bi = chan[0].bi_list[-i]
        returns[f"last_bi_is_sure{i}"] = int(bi.is_sure)
    return returns


def last_bi_high(chan: CChan):
    last_klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, BI_N):
        bi = chan[0].bi_list[-i]
        returns[f"last_bi_high{i}"] = bi._high() / last_klu.close - 1
    return returns


def last_bi_low(chan: CChan):
    last_klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, BI_N):
        bi = chan[0].bi_list[-i]
        returns[f"last_bi_low{i}"] = bi._low() / last_klu.close - 1
    return returns


def last_bi_mid(chan: CChan):
    last_klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, BI_N):
        bi = chan[0].bi_list[-i]
        returns[f"last_bi_mid{i}"] = bi._mid() / last_klu.close - 1
    return returns


def last_bi_begin(chan: CChan):
    last_klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, BI_N):
        bi = chan[0].bi_list[-i]
        returns[f"last_bi_begin{i}"] = bi.get_begin_val() / last_klu.close - 1
    return returns


def last_bi_end(chan: CChan):
    last_klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, BI_N):
        bi = chan[0].bi_list[-i]
        returns[f"last_bi_end{i}"] = bi.get_end_val() / last_klu.close - 1
    return returns


def last_bi_slope(chan: CChan):
    returns = dict()
    for i in range(1, BI_N):
        bi = chan[0].bi_list[-i]
        slope = (bi.get_begin_val() - bi.get_end_val()) / bi.get_end_val()
        returns[f"last_bi_slope{i}"] = slope
    return returns


def last_bi_macd_metric(chan: CChan):
    returns = dict()
    for i in range(1, BI_N):
        bi = chan[0].bi_list[-i]
        metric = bi.cal_macd_metric(MACD_ALGO.PEAK, is_reverse=False)
        returns[f"last_bi_macd_metric{i}"] = metric
    return returns


# 中枢
def last_zs_is_sure(chan: CChan):
    returns = dict()
    for i in range(1, ZS_N):
        zs = chan[0].zs_list[-i]
        returns[f"last_zs_is_sure{i}"] = int(zs.is_sure)
    return returns


def last_zs_is_one_bi_zs(chan: CChan):
    returns = dict()
    for i in range(1, ZS_N):
        zs = chan[0].zs_list[-i]
        returns[f"last_zs_is_one_bi_zs{i}"] = int(zs.is_one_bi_zs())
    return returns


def last_zs_high(chan: CChan):
    last_klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, ZS_N):
        zs = chan[0].zs_list[-i]
        returns[f"last_zs_high{i}"] = zs.high / last_klu.close - 1
    return returns


def last_zs_low(chan: CChan):
    last_klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, ZS_N):
        zs = chan[0].zs_list[-i]
        returns[f"last_zs_low{i}"] = zs.low / last_klu.close - 1
    return returns


def last_zs_mid(chan: CChan):
    last_klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, ZS_N):
        zs = chan[0].zs_list[-i]
        returns[f"last_zs_mid{i}"] = zs.mid / last_klu.close - 1
    return returns


# def last_zs_begin(chan: CChan):
#     last_klu = chan[0][-1][-1]
#     returns = dict()
#     for i in range(1, ZS_N):
#         zs = chan[0].zs_list[-i]
#         returns[f"last_zs_begin{i}"] = zs.begin / last_klu.close - 1
#     return returns
#
#
# def last_zs_end(chan: CChan):
#     last_klu = chan[0][-1][-1]
#     returns = dict()
#     for i in range(1, ZS_N):
#         zs = chan[0].zs_list[-i]
#         returns[f"last_zs_end{i}"] = zs.end / last_klu.close - 1
#     return returns


def last_zs_peak_high(chan: CChan):
    last_klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, ZS_N):
        zs = chan[0].zs_list[-i]
        returns[f"last_zs_peak_high{i}"] = zs.peak_high / last_klu.close - 1
    return returns


def last_zs_peak_low(chan: CChan):
    last_klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, ZS_N):
        zs = chan[0].zs_list[-i]
        returns[f"last_zs_peak_low{i}"] = zs.peak_low / last_klu.close - 1
    return returns
