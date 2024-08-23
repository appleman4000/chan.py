# cython: language_level=3
# encoding:utf-8
from Common.CEnum import MACD_ALGO, BI_DIR, FX_TYPE

MAX_BI = 1
MAX_ZS = 1
MAX_SEG = 1
MAX_SEGSEG = 1
MAX_SEGZS = 1


class FeatureFactors:
    def __init__(self, chan):
        self.chan = chan

    def get_factors(self):
        returns = dict()
        returns.update(self.bi())
        returns.update(self.zs())
        returns.update(self.seg())
        returns.update(self.open_klu_rate())
        returns.update(self.macd())
        returns.update(self.rsi())
        returns.update(self.kdj())
        returns.update(self.boll())
        # returns.update(self.fx())
        return returns

    # 最后K线涨跌率
    def open_klu_rate(self):
        returns = dict()
        for i in range(1, len(self.chan[-1]) + 1):
            klu = self.chan[-1][-i]
            returns[f"open_klu_rate{i}"] = klu.close / klu.open - 1
        return returns

    # 最后一个分型
    def fx(self):
        fx = self.chan[-1].fx
        return {"fx": list(FX_TYPE).index(fx)}

    ############################### 笔 ########################################
    def bi(self):
        def bi_begin(i, bi):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"bi_begin{i}"] = (klu.idx - bi.get_begin_klu().idx + 1)
            returns[f"bi_begin_rate{i}"] = bi.get_begin_val() / klu.close - 1
            returns[f"bi_begin_slope{i}"] = (bi.get_begin_val() - klu.close) / (
                    klu.idx - bi.get_begin_klu().idx + 1)
            return returns

        def bi_end(i, bi):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"bi_end{i}"] = (klu.idx - bi.get_end_klu().idx + 1)
            returns[f"bi_end_rate{i}"] = bi.get_end_val() / klu.close - 1
            returns[f"bi_end_slope{i}"] = (bi.get_end_val() - klu.close) / (klu.idx - bi.get_end_klu().idx + 1)
            return returns

        def bi_dir(i, bi):
            returns = dict()
            returns[f"bi_dir{i}"] = list(BI_DIR).index(bi.dir)
            return returns

        def bi_is_sure(i, bi):
            returns = dict()
            returns[f"bi_is_sure{i}"] = int(bi.is_sure)
            return returns

        def bi_high(i, bi):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"bi_high_rate{i}"] = bi._high() / klu.close - 1
            return returns

        def bi_low(i, bi):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"bi_low_rate{i}"] = bi._low() / klu.close - 1
            return returns

        def bi_mid(i, bi):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"bi_mid_rate{i}"] = bi._mid() / klu.close - 1
            return returns

        def bi_rate_slope(i, bi):
            returns = dict()
            returns[f"bi_rate{i}"] = bi.get_end_val() / bi.get_begin_val() - 1
            returns[f"bi_slope{i}"] = (bi.get_end_val() - bi.get_begin_val()) / bi.get_klu_cnt()
            return returns

        def bi_klu_cnt(i, bi):
            returns = dict()
            returns[f"bi_klu_cnt{i}"] = bi.get_klu_cnt()
            return returns

        def bi_klc_cnt(i, bi):
            returns = dict()
            returns[f"bi_klc_cnt{i}"] = bi.get_klc_cnt()
            return returns

        returns = dict()
        for i in range(1, MAX_BI + 1):
            if i < len(self.chan.bi_list):
                bi = self.chan.bi_list[-i]
                returns.update(bi_begin(i, bi))
                returns.update(bi_end(i, bi))
                # returns.update(bi_dir(i, bi))
                # returns.update(bi_is_sure(i, bi))
                returns.update(bi_high(i, bi))
                returns.update(bi_low(i, bi))
                returns.update(bi_mid(i, bi))
                returns.update(bi_rate_slope(i, bi))
                returns.update(bi_klu_cnt(i, bi))
                returns.update(bi_klc_cnt(i, bi))
        return returns

    ############################### 中枢 ####################################
    def zs(self):
        def zs_begin(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_begin{i}"] = klu.idx - zs.begin.idx + 1
            return returns

        def zs_end(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_end{i}"] = klu.idx - zs.end.idx + 1
            return returns

        def zs_high(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_high_rate{i}"] = zs.high / klu.close - 1
            return returns

        def zs_low(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_low_rate{i}"] = zs.low / klu.close - 1
            return returns

        def zs_mid(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_mid_rate{i}"] = zs.mid / klu.close - 1
            return returns

        def zs_peak_high(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_peak_high_rate{i}"] = zs.peak_high / klu.close - 1
            return returns

        def zs_peak_low(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_peak_low_rate{i}"] = zs.peak_low / klu.close - 1
            return returns

        def divergence(i, zs):
            returns = dict()

            for macd_algo in [MACD_ALGO.AREA,
                              MACD_ALGO.PEAK,
                              MACD_ALGO.FULL_AREA,
                              MACD_ALGO.DIFF,
                              MACD_ALGO.SLOPE,
                              MACD_ALGO.AMP]:
                bi_in_metric = zs.bi_in.cal_macd_metric(macd_algo, is_reverse=False)
                bi_out_metric = zs.bi_out.cal_macd_metric(macd_algo, is_reverse=True)
                returns[f"bi_in_{macd_algo.name}{i}"] = bi_in_metric
                returns[f"bi_out_{macd_algo.name}{i}"] = bi_out_metric
                returns[f"divergence_{macd_algo.name}{i}"] = bi_out_metric / bi_in_metric
            return returns

        returns = dict()
        if len(self.chan.zs_list) > 0:
            for i in range(1, MAX_ZS + 1):
                if i < len(self.chan.zs_list):
                    zs = self.chan.zs_list[-i]
                    returns.update(zs_begin(i, zs))
                    returns.update(zs_end(i, zs))
                    returns.update(zs_high(i, zs))
                    returns.update(zs_low(i, zs))
                    returns.update(zs_mid(i, zs))
                    returns.update(zs_peak_high(i, zs))
                    returns.update(zs_peak_low(i, zs))
                    returns.update(divergence(i, zs))

        if len(self.chan.segzs_list) > 0:
            for i in range(1, MAX_SEGZS + 1):
                if i < len(self.chan.segzs_list):
                    segzs = self.chan.segzs_list[-i]
                    returns.update(zs_begin(i + MAX_ZS, segzs))
                    returns.update(zs_end(i + MAX_ZS, segzs))
                    returns.update(zs_high(i + MAX_ZS, segzs))
                    returns.update(zs_low(i + MAX_ZS, segzs))
                    returns.update(zs_mid(i + MAX_ZS, segzs))
                    returns.update(zs_peak_high(i + MAX_ZS, segzs))
                    returns.update(zs_peak_low(i + MAX_ZS, segzs))
                    for macd_algo in [
                        MACD_ALGO.SLOPE,
                        MACD_ALGO.AMP]:
                        bi_in_metric = segzs.bi_in.cal_macd_metric(macd_algo, is_reverse=False)
                        bi_out_metric = segzs.bi_out.cal_macd_metric(macd_algo, is_reverse=True)
                        returns[f"bi_in_{macd_algo.name}{i + MAX_ZS}"] = bi_in_metric
                        returns[f"bi_out_{macd_algo.name}{i + MAX_ZS}"] = bi_out_metric
                        returns[f"divergence_{macd_algo.name}{i + MAX_ZS}"] = bi_out_metric / bi_in_metric - 1
        return returns

    ############################### 线段 ######################################
    def seg(self):
        def seg_begin(i, seg):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"seg_begin{i}"] = klu.idx - seg.get_begin_klu().idx + 1
            returns[f"seg_begin_rate{i}"] = seg.get_begin_val() / klu.close - 1
            returns[f"seg_begin_slope{i}"] = (seg.get_begin_val() - klu.close) / (
                    klu.idx - seg.get_begin_klu().idx + 1)
            return returns

        def seg_end(i, seg):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"seg_end{i}"] = klu.idx - seg.get_end_klu().idx + 1
            returns[f"seg_end_val_rate{i}"] = seg.get_end_val() / klu.close - 1
            returns[f"seg_end_val_slope{i}"] = (seg.get_end_val() - klu.close) / (
                    klu.idx - seg.get_end_klu().idx + 1)
            return returns

        def seg_rate_slope(i, seg):
            returns = dict()
            returns[f"seg_rate{i}"] = seg.get_end_val() / seg.get_begin_val() - 1
            returns[f"seg_slope{i}"] = (seg.get_end_val() - seg.get_begin_val()) / (
                    seg.get_end_klu().idx - seg.get_begin_klu().idx)
            return returns

        def seg_bi_cnt(i, seg):
            returns = dict()
            returns[f"seg_bi_cnt{i}"] = seg.cal_bi_cnt()
            return returns

        def seg_low(i, seg):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"seg_low_rate{i}"] = seg._low() / klu.close - 1
            return returns

        def seg_high(i, seg):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"seg_high_rate{i}"] = seg._high() / klu.close - 1
            return returns

        def seg_is_down(i, seg):
            returns = dict()
            returns[f"seg_is_down{i}"] = int(seg.is_down())
            return returns

        def seg_klu_cnt(i, seg):
            returns = dict()
            returns[f"seg_klu_cnt{i}"] = seg.get_klu_cnt()
            return returns

        def seg_macd(i, seg):
            returns = dict()
            returns[f"seg_macd_slope{i}"] = seg.Cal_MACD_slope()
            returns[f"seg_macd_amp{i}"] = seg.Cal_MACD_amp()
            return returns

        returns = dict()
        if len(self.chan.seg_list) > 0:
            for i in range(1, MAX_SEG + 1):
                if i < len(self.chan.seg_list):
                    seg = self.chan.seg_list[-i]
                    returns.update(seg_begin(i, seg))
                    returns.update(seg_end(i, seg))
                    returns.update(seg_rate_slope(i, seg))
                    returns.update(seg_bi_cnt(i, seg))
                    returns.update(seg_low(i, seg))
                    returns.update(seg_high(i, seg))
                    # returns.update(seg_is_down(i, seg))
                    returns.update(seg_klu_cnt(i, seg))
                    returns.update(seg_macd(i, seg))

        if len(self.chan.segseg_list) > 0:
            for i in range(1, MAX_SEGSEG + 1):
                if i < len(self.chan.segseg_list):
                    segseg = self.chan.segseg_list[-i]
                    returns.update(seg_begin(i + MAX_SEG, segseg))
                    returns.update(seg_end(i + MAX_SEG, segseg))
                    returns.update(seg_rate_slope(i + MAX_SEG, segseg))
                    returns.update(seg_bi_cnt(i + MAX_SEG, segseg))
                    returns.update(seg_low(i + MAX_SEG, segseg))
                    returns.update(seg_high(i + MAX_SEG, segseg))
                    returns.update(seg_is_down(i + MAX_SEG, segseg))
                    returns.update(seg_klu_cnt(i + MAX_SEG, segseg))
                    returns.update(seg_macd(i + MAX_SEG, segseg))
        return returns

    def macd(self):
        returns = dict()
        returns["macd"] = self.chan[-1][-1].macd.macd
        returns["DIF"] = self.chan[-1][-1].macd.DIF
        returns["DE"] = self.chan[-1][-1].macd.DEA
        return returns

    def rsi(self):
        returns = dict()
        returns["rsi"] = self.chan[-1][-1].rsi
        return returns

    def kdj(self):
        returns = dict()
        returns["k"] = self.chan[-1][-1].kdj.k
        returns["d"] = self.chan[-1][-1].kdj.d
        returns["j"] = self.chan[-1][-1].kdj.j
        return returns

    def boll(self):
        klu = self.chan[-1][-1]
        returns = dict()
        returns["k"] = self.chan[-1][-1].boll.UP / klu.close - 1
        returns["d"] = self.chan[-1][-1].boll.MID / klu.close - 1
        returns["j"] = self.chan[-1][-1].boll.DOWN / klu.close - 1
        return returns
