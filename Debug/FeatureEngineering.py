# encoding:utf-8
from Chan import CChan
from Common.CEnum import MACD_ALGO, BI_DIR, BSP_TYPE, FX_TYPE

N_BI = 5
N_ZS = 2
N_SEG = 1


def open_klu_rate(chan: CChan):
    klu = chan[0][-1][-1]
    return {
        "open_klu_rate": klu.close/ klu.open,
    }


def bsp_type(chan: CChan):
    bsp_list = chan.get_bsp()
    if not bsp_list:
        return {"bsp_type": None}
    bsp = bsp_list[-1]
    return {"bsp_type": list(BSP_TYPE).index(bsp.type[0])}


def fx(chan: CChan):
    fx = chan[0][-1].fx
    return {"fx": list(FX_TYPE).index(fx)}


############################### 笔 ####################################################
def bi_dir(chan: CChan):
    returns = dict()
    for i in range(1, N_BI + 1):
        bi = chan[0].bi_list[-i]
        returns[f"bi_dir{i}"] = list(BI_DIR).index(bi.dir)
    return returns


def bi_is_sure(chan: CChan):
    returns = dict()
    for i in range(1, N_BI + 1):
        bi = chan[0].bi_list[-i]
        returns[f"bi_is_sure{i}"] = int(bi.is_sure)
    return returns


def bi_high(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_BI + 1):
        bi = chan[0].bi_list[-i]
        returns[f"bi_high{i}"] = bi._high() / klu.close
    return returns


def bi_low(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_BI + 1):
        bi = chan[0].bi_list[-i]
        returns[f"bi_low{i}"] = bi._low() / klu.close
    return returns


def bi_mid(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_BI + 1):
        bi = chan[0].bi_list[-i]
        returns[f"bi_mid{i}"] = bi._mid() / klu.close
    return returns


def bi_begin(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_BI + 1):
        bi = chan[0].bi_list[-i]
        returns[f"bi_begin{i}"] = bi.get_begin_val() / klu.close
    return returns


def bi_end(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_BI + 1):
        bi = chan[0].bi_list[-i]
        returns[f"bi_end{i}"] = bi.get_end_val() / klu.close
    return returns


def bi_slope(chan: CChan):
    returns = dict()
    for i in range(1, N_BI + 1):
        bi = chan[0].bi_list[-i]
        slope = bi.get_end_val() / bi.get_begin_val()
        returns[f"bi_slope{i}"] = slope
    return returns


def bi_klu_cnt(chan: CChan):
    returns = dict()
    for i in range(1, N_BI + 1):
        bi = chan[0].bi_list[-i]
        returns[f"bi_klu_cnt{i}"] = bi.get_klu_cnt()
    return returns


def bi_klc_cnt(chan: CChan):
    returns = dict()
    for i in range(1, N_BI + 1):
        bi = chan[0].bi_list[-i]
        returns[f"bi_klc_cnt{i}"] = bi.get_klc_cnt()
    return returns


def divergence_macd_metric(chan: CChan):
    returns = dict()
    for macd_algo in [MACD_ALGO.AREA,
                      MACD_ALGO.PEAK,
                      MACD_ALGO.FULL_AREA,
                      MACD_ALGO.DIFF,
                      MACD_ALGO.SLOPE,
                      MACD_ALGO.AMP]:
        bi_in_metric = chan[0].zs_list[-1].bi_in.cal_macd_metric(macd_algo, is_reverse=False)
        bi_out_metric = chan[0].zs_list[-1].bi_out.cal_macd_metric(macd_algo, is_reverse=True)
        returns[f"bi_in_divergence_macd_metric{macd_algo.name}"] = bi_in_metric
        returns[f"bi_out_divergence_macd_metric{macd_algo.name}"] = bi_out_metric
        returns[f"divergence_macd_metric{macd_algo.name}"] = bi_out_metric / bi_in_metric
    return returns


############################### 中枢 ####################################################
def zs_high(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_ZS + 1):
        zs = chan[0].zs_list[-i]
        returns[f"zs_high{i}"] = zs.high / klu.close
    return returns


def zs_low(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_ZS + 1):
        zs = chan[0].zs_list[-i]
        returns[f"zs_low{i}"] = zs.low / klu.close
    return returns


def zs_mid(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_ZS + 1):
        zs = chan[0].zs_list[-i]
        returns[f"zs_mid{i}"] = zs.mid / klu.close
    return returns


def zs_peak_high(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_ZS + 1):
        zs = chan[0].zs_list[-i]
        returns[f"zs_peak_high{i}"] = zs.peak_high / klu.close
    return returns


def zs_peak_low(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_ZS + 1):
        zs = chan[0].zs_list[-i]
        returns[f"zs_peak_low{i}"] = zs.peak_low / klu.close
    return returns


def zs_is_inside(chan: CChan):
    returns = dict()
    for i in range(1, N_ZS + 1):
        zs = chan[0].zs_list[-i]
        for j in range(1, N_SEG + 1):
            seg = chan[0].seg_list[-j]
            returns[f"zs_is_inside{i} {j}"] = int(zs.is_inside(seg))
    return returns


############################### 线段 ####################################################
def seg_cal_klu_slope(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_SEG + 1):
        seg = chan[0].seg_list[-i]
        returns[f"seg_cal_klu_slope{i}"] = seg.cal_klu_slope()
    return returns


def seg_cal_amp(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_SEG + 1):
        seg = chan[0].seg_list[-i]
        returns[f"seg_cal_amp{i}"] = seg.cal_amp()
    return returns


def seg_cal_bi_cnt(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_SEG + 1):
        seg = chan[0].seg_list[-i]
        returns[f"seg_cal_bi_cnt{i}"] = seg.cal_bi_cnt()
    return returns


def seg_low(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_SEG + 1):
        seg = chan[0].seg_list[-i]
        returns[f"seg_low{i}"] = seg._low() / klu.close
    return returns


def seg_high(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_SEG + 1):
        seg = chan[0].seg_list[-i]
        returns[f"seg_high{i}"] = seg._high() / klu.close
    return returns


def seg_is_down(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_SEG + 1):
        seg = chan[0].seg_list[-i]
        returns[f"seg_is_down{i}"] = int(seg.is_down())
    return returns


def seg_begin_val(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_SEG + 1):
        seg = chan[0].seg_list[-i]
        returns[f"seg_begin_val{i}"] = seg.get_begin_val() / klu.close
    return returns


def seg_end_val(chan: CChan):
    klu = chan[0][-1][-1]
    returns = dict()
    for i in range(1, N_SEG + 1):
        seg = chan[0].seg_list[-i]
        returns[f"seg_begin_val{i}"] = seg.get_end_val() / klu.close
    return returns


def seg_amp(chan: CChan):
    returns = dict()
    for i in range(1, N_SEG + 1):
        seg = chan[0].seg_list[-i]
        returns[f"seg_amp{i}"] = seg.get_end_val() / seg.get_begin_val()
    return returns


def seg_klu_cnt(chan: CChan):
    returns = dict()
    for i in range(1, N_SEG + 1):
        seg = chan[0].seg_list[-i]
        returns[f"seg_klu_cnt{i}"] = seg.get_klu_cnt()
    return returns


def seg_macd_slope(chan: CChan):
    returns = dict()
    for i in range(1, N_SEG + 1):
        seg = chan[0].seg_list[-i]
        returns[f"seg_macd_slope{i}"] = seg.Cal_MACD_slope()
    return returns


def seg_macd_amp(chan: CChan):
    returns = dict()
    for i in range(1, N_SEG + 1):
        seg = chan[0].seg_list[-i]
        returns[f"seg_macd_amp{i}"] = seg.Cal_MACD_amp()
    return returns
