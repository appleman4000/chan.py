# cython: language_level=3
# encoding:utf-8
from Common.CEnum import BI_DIR, FX_TYPE, MACD_ALGO


class FeatureFactors:
    def __init__(self, chan,
                 pip_value,
                 MAX_BI=2,
                 MAX_ZS=1,
                 MAX_SEG=1,
                 MAX_SEGSEG=1,
                 MAX_SEGZS=1):
        self.chan = chan
        self.pip_value = pip_value
        self.MAX_BI = MAX_BI
        self.MAX_ZS = MAX_ZS
        self.MAX_SEG = MAX_SEG
        self.MAX_SEGSEG = MAX_SEGSEG
        self.MAX_SEGZS = MAX_SEGZS

    def get_factors(self):
        returns = dict()
        returns.update(self.bi())
        returns.update(self.zs())
        returns.update(self.seg())
        # returns.update(self.open_klu_rate())
        # returns.update(self.fx())
        return returns
    # def open_klu_rate(self):
    #     returns = dict()
    #     klu = self.chan[-1][-1]
    #     returns[f"open_klu_rate"] = (klu.close - klu.open) / self.pip_value
    #     return returns
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
            returns[f"bi_begin_rate{i}"] = (bi.get_begin_val() / klu.close)
            returns[f"bi_begin_slope{i}"] = (bi.get_begin_val() - klu.close) / (
                    klu.idx - bi.get_begin_klu().idx + 1)
            return returns

        def bi_end(i, bi):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"bi_end{i}"] = (klu.idx - bi.get_end_klu().idx + 1)
            returns[f"bi_end_rate{i}"] = bi.get_end_val() / klu.close
            returns[f"bi_end_slope{i}"] = (bi.get_end_val() - klu.close) / (
                    klu.idx - bi.get_end_klu().idx + 1)
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
            returns[f"bi_high{i}"] = bi._high() / bi._low()
            returns[f"bi_high_rate{i}"] = bi._high() / klu.close
            return returns

        def bi_low(i, bi):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"bi_low_rate{i}"] = bi._low() / klu.close
            return returns

        def bi_mid(i, bi):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"bi_mid_rate{i}"] = bi._mid() / klu.close
            return returns

        def bi_rate(i, bi):
            returns = dict()
            returns[f"bi_rate{i}"] = bi.get_end_val() / bi.get_begin_val()
            returns[f"bi_slope{i}"] = (bi.get_end_val() - bi.get_begin_val())/ bi.get_klu_cnt()
            return returns

        def bi_klu_cnt(i, bi):
            returns = dict()
            returns[f"bi_klu_cnt{i}"] = bi.get_klu_cnt()
            return returns

        def bi_klc_cnt(i, bi):
            returns = dict()
            returns[f"bi_klc_cnt{i}"] = bi.get_klc_cnt()
            return returns

        def indicators(i, klu):
            returns = dict()
            for key, value in klu.indicators.items():
                returns[f"bi_end_{key}{i}"] = value
            return returns

        returns = dict()
        for i in range(1, self.MAX_BI + 1):
            if i < len(self.chan.bi_list):
                bi = self.chan.bi_list[-i]
                returns.update(bi_begin(i, bi))
                if i > 1:
                    returns.update(bi_end(i, bi))
                # returns.update(bi_dir(i, bi))
                # returns.update(bi_is_sure(i, bi))
                returns.update(bi_high(i, bi))
                returns.update(bi_low(i, bi))
                returns.update(bi_mid(i, bi))
                returns.update(bi_rate(i, bi))
                returns.update(bi_klu_cnt(i, bi))
                returns.update(bi_klc_cnt(i, bi))
                returns.update(indicators(i, bi.end_klc[-1]))
        return returns

    ############################### 中枢 ####################################
    def zs(self):
        def zs_begin(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_begin{i}"] = klu.idx - zs.begin.idx + 1
            for key, value in zs.begin.indicators.items():
                returns[f"zs_begin_{key}{i}"] = value
            return returns

        def zs_end(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_end{i}"] = zs.begin.idx - zs.end.idx + 1
            for key, value in zs.end.indicators.items():
                returns[f"zs_end_{key}{i}"] = value
            return returns

        def zs_length(i, zs):
            returns = dict()
            returns[f"zs_length{i}"] = zs.end.idx - zs.begin.idx + 1
            return returns

        def zs_high(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_high{i}"] = zs.high / zs.low
            returns[f"zs_high_rate{i}"] = zs.high / klu.close
            return returns

        def zs_low(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_low_rate{i}"] = zs.low / klu.close
            return returns

        def zs_mid(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_mid_rate{i}"] = zs.mid / klu.close
            return returns

        def zs_peak_high(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_peak_high{i}"] = zs.peak_high / zs.peak_low
            returns[f"zs_peak_high_rate{i}"] = zs.peak_high / klu.close
            return returns

        def zs_peak_low(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_peak_low_rate{i}"] = zs.peak_low / klu.close
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
                returns[f"bi_in_{macd_algo.name}{i}"] = bi_in_metric
                if zs.bi_out:
                    bi_out_metric = zs.bi_out.cal_macd_metric(macd_algo, is_reverse=True)
                    returns[f"bi_out_{macd_algo.name}{i}"] = bi_out_metric
                    returns[f"divergence_{macd_algo.name}{i}"] = bi_out_metric / (bi_in_metric+1e-7)
            return returns

        returns = dict()
        if len(self.chan.zs_list) > 0:
            for i in range(1, self.MAX_ZS + 1):
                if i < len(self.chan.zs_list):
                    zs = self.chan.zs_list[-i]
                    returns.update(zs_begin(i, zs))
                    returns.update(zs_end(i, zs))
                    returns.update(zs_length(i, zs))
                    returns.update(zs_high(i, zs))
                    returns.update(zs_low(i, zs))
                    returns.update(zs_mid(i, zs))
                    returns.update(zs_peak_high(i, zs))
                    returns.update(zs_peak_low(i, zs))
                    returns.update(divergence(i, zs))
        if len(self.chan.segzs_list) > 0:
            for i in range(1, self.MAX_SEGZS + 1):
                if i < len(self.chan.segzs_list):
                    segzs = self.chan.segzs_list[-i]
                    returns.update(zs_begin(f"_segzs{i}", segzs))
                    returns.update(zs_end(f"_segzs{i}", segzs))
                    returns.update(zs_high(f"_segzs{i}", segzs))
                    returns.update(zs_low(f"_segzs{i}", segzs))
                    returns.update(zs_mid(f"_segzs{i}", segzs))
                    returns.update(zs_peak_high(f"_segzs{i}", segzs))
                    returns.update(zs_peak_low(f"_segzs{i}", segzs))
                    for macd_algo in [
                        MACD_ALGO.SLOPE,
                        MACD_ALGO.AMP]:
                        bi_in_metric = segzs.bi_in.cal_macd_metric(macd_algo, is_reverse=False)
                        bi_out_metric = segzs.bi_out.cal_macd_metric(macd_algo, is_reverse=True)
                        returns[f"segzs_bi_in_{macd_algo.name}{i}"] = bi_in_metric
                        returns[f"segzs_bi_out_{macd_algo.name}{i}"] = bi_out_metric
                        returns[f"segzs_divergence_{macd_algo.name}{i}"] = bi_out_metric - bi_in_metric

        return returns

    ############################### 线段 ######################################
    def seg(self):
        def seg_begin(i, seg):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"seg_begin{i}"] = klu.idx - seg.get_begin_klu().idx + 1
            returns[f"seg_begin_rate{i}"] = seg.get_begin_val() / klu.close
            returns[f"seg_begin_slope{i}"] = (seg.get_begin_val() - klu.close) / (
                    klu.idx - seg.get_begin_klu().idx + 1)
            return returns

        def seg_end(i, seg):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"seg_end{i}"] = klu.idx - seg.get_end_klu().idx + 1
            returns[f"seg_end_val_rate{i}"] = seg.get_end_val() / klu.close
            returns[f"seg_end_val_slope{i}"] = (seg.get_end_val() - klu.close) / (
                    klu.idx - seg.get_end_klu().idx + 1)
            return returns

        def seg_rate(i, seg):
            returns = dict()
            returns[f"seg_rate{i}"] = seg.get_end_val() / seg.get_begin_val()
            returns[f"seg_slope{i}"] = (seg.get_end_val() - seg.get_begin_val()) / (
                    seg.get_end_klu().idx - seg.get_begin_klu().idx)
            return returns

        def seg_bi_cnt(i, seg):
            returns = dict()
            returns[f"seg_bi_cnt{i}"] = seg.cal_bi_cnt()
            return returns

        def seg_high(i, seg):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"seg_high{i}"] = seg._high() / seg._low()
            returns[f"seg_high_rate{i}"] = seg._high() / klu.close
            return returns

        def seg_low(i, seg):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"seg_low_rate{i}"] = seg._low() / klu.close
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
            for i in range(1, self.MAX_SEG + 1):
                if i < len(self.chan.seg_list):
                    seg = self.chan.seg_list[-i]
                    returns.update(seg_begin(i, seg))
                    returns.update(seg_end(i, seg))
                    returns.update(seg_rate(i, seg))
                    returns.update(seg_bi_cnt(i, seg))
                    returns.update(seg_low(i, seg))
                    returns.update(seg_high(i, seg))
                    # returns.update(seg_is_down(i, seg))
                    returns.update(seg_klu_cnt(i, seg))
                    returns.update(seg_macd(i, seg))

        if len(self.chan.segseg_list) > 0:
            for i in range(1, self.MAX_SEGSEG + 1):
                if i < len(self.chan.segseg_list):
                    segseg = self.chan.segseg_list[-i]
                    returns.update(seg_begin(f"_segseg{i}", segseg))
                    returns.update(seg_end(f"_segseg{i}", segseg))
                    returns.update(seg_rate(f"_segseg{i}", segseg))
                    returns.update(seg_bi_cnt(f"_segseg{i}", segseg))
                    returns.update(seg_low(f"_segseg{i}", segseg))
                    returns.update(seg_high(f"_segseg{i}", segseg))
                    # returns.update(seg_is_down(f"_segseg{i}", segseg))
                    returns.update(seg_klu_cnt(f"_segseg{i}", segseg))
                    returns.update(seg_macd(f"_segseg{i}", segseg))
        return returns
