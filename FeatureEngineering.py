# cython: language_level=3
# encoding:utf-8
from Common.CEnum import BI_DIR, FX_TYPE, MACD_ALGO


class FeatureFactors:
    def __init__(self, chan,
                 MAX_BI=2):
        self.chan = chan
        self.MAX_BI = MAX_BI

    def get_factors(self):
        bis = []
        for i in range(1, self.MAX_BI + 1):
            if i < len(self.chan.bi_list):
                bi = self.chan.bi_list[-i]
                bis.append(bi)
        segs = []
        for bi in bis:
            if bi.parent_seg:
                seg = bi.parent_seg
                if seg not in segs:
                    segs.append(seg)
        segsegs = []
        for seg in segs:
            if seg.parent_seg:
                segseg = seg.parent_seg
                if segseg not in segsegs:
                    segsegs.append(segseg)

        returns = dict()
        returns.update(self.bi(bis))
        returns.update(self.zs(segs, segsegs))
        returns.update(self.seg(segs, segsegs))
        return returns

    # 最后一个分型
    def fx(self):
        fx = self.chan[-1].fx
        return {"fx": list(FX_TYPE).index(fx)}

    ############################### 笔 ########################################
    def bi(self, bis):
        def open_klu_rate(i, bi):
            returns = dict()
            returns[f"open_klu_rate"] = bi.get_end_klu().close / bi.get_end_klu().open - 1
            return returns

        def bi_begin(i, bi):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"bi_begin_klu_cnt{i}"] = (klu.idx - bi.get_begin_klu().idx + 1)
            returns[f"bi_begin_klc_cnt{i}"] = (klu.klc.idx - bi.get_begin_klu().klc.idx + 1)
            returns[f"bi_begin_amp{i}"] = bi.get_begin_val() / klu.close - 1
            returns[f"bi_begin_slope{i}"] = (bi.get_begin_val() / klu.close - 1) / (
                    klu.idx - bi.get_begin_klu().idx + 1)
            return returns

        def bi_end(i, bi):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"bi_end_klu_cnt{i}"] = (klu.idx - bi.get_end_klu().idx + 1)
            returns[f"bi_end_klc_cnt{i}"] = (klu.klc.idx - bi.get_end_klu().klc.idx + 1)
            returns[f"bi_end_amp{i}"] = bi.get_end_val() / klu.close - 1
            returns[f"bi_end_slope{i}"] = (bi.get_end_val() / klu.close - 1) / (
                    klu.idx - bi.get_end_klu().idx + 1)
            return returns

        def bi_high(i, bi):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"bi_high{i}"] = (bi._high() - bi._low()) / -1
            returns[f"bi_high_amp{i}"] = bi._high() / klu.close - 1
            returns[f"bi_high_low_amp{i}"] = (bi._high() / bi._low()) - 1
            return returns

        def bi_low(i, bi):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"bi_low_amp{i}"] = (bi._low() / klu.close) - 1
            return returns

        def bi_mid(i, bi):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"bi_mid_amp{i}"] = (bi._mid() / klu.close) - 1
            return returns

        def bi_amp(i, bi):
            returns = dict()
            returns[f"bi_amp{i}"] = (bi.get_end_val() / bi.get_begin_val()) - 1
            returns[f"bi_slope{i}"] = (bi.get_end_val() / bi.get_begin_val() - 1) / bi.get_klu_cnt()
            return returns

        def bi_dir(i, bi):
            returns = dict()
            returns[f"bi_dir{i}"] = list(BI_DIR).index(bi.dir)
            return returns

        def bi_is_sure(i, bi):
            returns = dict()
            returns[f"bi_is_sure{i}"] = int(bi.is_sure)
            return returns

        def bi_klu_cnt(i, bi):
            returns = dict()
            returns[f"bi_klu_cnt{i}"] = bi.get_klu_cnt()
            return returns

        def bi_klc_cnt(i, bi):
            returns = dict()
            returns[f"bi_klc_cnt{i}"] = bi.get_klc_cnt()
            return returns

        def macd(i, bi):
            returns = dict()
            returns[f"macd{i}"] = bi.get_end_klu().macd.macd
            returns[f"dif{i}"] = bi.get_end_klu().macd.DIF
            returns[f"dea{i}"] = bi.get_end_klu().macd.DEA

            return returns

        def rsi(i, bi):
            returns = dict()
            returns[f"rsi{i}"] = bi.get_end_klu().rsi
            return returns

        def kdj(i, bi):
            returns = dict()
            returns[f"k{i}"] = bi.get_end_klu().kdj.k
            returns[f"d{i}"] = bi.get_end_klu().kdj.d
            returns[f"j{i}"] = bi.get_end_klu().kdj.j
            return returns

        def boll(i, bi):
            returns = dict()
            returns[f"up{i}"] = (bi.get_end_klu().boll.UP / bi.get_end_klu().close)
            returns[f"mid{i}"] = (bi.get_end_klu().boll.MID / bi.get_end_klu().close)
            returns[f"down{i}"] = (bi.get_end_klu().boll.DOWN / bi.get_end_klu().close)
            return returns

        returns = dict()
        for i, bi in enumerate(bis):
            returns.update(open_klu_rate(i, bi))
            returns.update(bi_begin(i, bi))
            returns.update(bi_end(i, bi))
            returns.update(bi_high(i, bi))
            returns.update(bi_low(i, bi))
            returns.update(bi_mid(i, bi))
            returns.update(bi_amp(i, bi))
            returns.update(bi_klu_cnt(i, bi))
            returns.update(bi_klc_cnt(i, bi))
            returns.update(macd(i, bi))
            returns.update(rsi(i, bi))
            returns.update(kdj(i, bi))
            returns.update(boll(i, bi))
        return returns

    ############################### 中枢 ####################################
    def zs(self, segs, segsegs):
        def zs_begin(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_begin_klu_cnt{i}"] = klu.idx - zs.begin.idx + 1
            returns[f"zs_begin_klc_cnt{i}"] = klu.klc.idx - zs.begin.klc.idx + 1
            return returns

        def zs_end(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_end_klu_cnt{i}"] = klu.idx - zs.end.idx + 1
            returns[f"zs_end_klc_cnt{i}"] = klu.klc.idx - zs.end.klc.idx + 1
            return returns

        def zs_klu_cnt(i, zs):
            returns = dict()
            returns[f"zs_klu_cnt{i}"] = zs.end.idx - zs.begin.idx + 1
            return returns

        def zs_klc_cnt(i, zs):
            returns = dict()
            returns[f"zs_klc_cnt{i}"] = zs.end.klc.idx - zs.begin.klc.idx + 1
            return returns

        def zs_high(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_high_low_amp{i}"] = zs.high / zs.low - 1
            returns[f"zs_high_amp{i}"] = zs.high / klu.close - 1
            returns[f"zs_slope{i}"] = (zs.high / zs.low - 1) / (zs.end.idx - zs.begin.idx + 1)
            return returns

        def zs_low(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_low_amp{i}"] = zs.low / klu.close - 1
            return returns

        def zs_mid(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_mid_amp{i}"] = zs.mid / klu.close - 1
            return returns

        def zs_peak_high(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_peak_high_low_amp{i}"] = zs.peak_high / zs.peak_low - 1
            returns[f"zs_peak_high_amp{i}"] = zs.peak_high / klu.close - 1
            returns[f"zs_peak_high_slope{i}"] = (zs.peak_high / zs.peak_low - 1) / (zs.end.idx - zs.begin.idx + 1)
            return returns

        def zs_peak_low(i, zs):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"zs_peak_low_amp{i}"] = zs.peak_low / klu.close - 1
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
                # returns[f"bi_in_{macd_algo.name}{i}"] = bi_in_metric
                if zs.bi_out:
                    bi_out_metric = zs.bi_out.cal_macd_metric(macd_algo, is_reverse=True)
                    # returns[f"bi_out_{macd_algo.name}{i}"] = bi_out_metric
                    returns[f"divergence_{macd_algo.name}{i}"] = bi_out_metric / (bi_in_metric + 1e-7) - 1
            return returns

        returns = dict()

        for i, seg in enumerate(segs):
            zs_lst = seg.zs_lst[::-1]
            for j in range(len(zs_lst)):
                zs = zs_lst[j]
                returns.update(zs_begin(f"{i}{j}", zs))
                returns.update(zs_end(f"{i}{j}", zs))
                returns.update(zs_klu_cnt(f"{i}{j}", zs))
                returns.update(zs_klc_cnt(f"{i}{j}", zs))
                returns.update(zs_high(f"{i}{j}", zs))
                returns.update(zs_low(f"{i}{j}", zs))
                returns.update(zs_mid(f"{i}{j}", zs))
                returns.update(zs_peak_high(f"{i}{j}", zs))
                returns.update(zs_peak_low(f"{i}{j}", zs))
                returns.update(divergence(f"{i}{j}", zs))

        for i, segseg in enumerate(segsegs):
            zs_lst = segseg.zs_lst[::-1]
            for j in range(len(zs_lst)):
                segzs = zs_lst[j]
                returns.update(zs_begin(f"_segzs{i}{j}", segzs))
                returns.update(zs_end(f"_segzs{i}{j}", segzs))
                returns.update(zs_klu_cnt(f"_segzs{i}{j}", segzs))
                returns.update(zs_klc_cnt(f"_segzs{i}{j}", segzs))
                returns.update(zs_high(f"_segzs{i}{j}", segzs))
                returns.update(zs_low(f"_segzs{i}{j}", segzs))
                returns.update(zs_mid(f"_segzs{i}{j}", segzs))
                returns.update(zs_peak_high(f"_segzs{i}{j}", segzs))
                returns.update(zs_peak_low(f"_segzs{i}{j}", segzs))
                for macd_algo in [
                    MACD_ALGO.SLOPE,
                    MACD_ALGO.AMP]:
                    bi_in_metric = segzs.bi_in.cal_macd_metric(macd_algo, is_reverse=False)
                    bi_out_metric = segzs.bi_out.cal_macd_metric(macd_algo, is_reverse=True)
                    # returns[f"segzs_bi_in_{macd_algo.name}{i}"] = bi_in_metric
                    # returns[f"segzs_bi_out_{macd_algo.name}{i}"] = bi_out_metric
                    returns[f"segzs_divergence_{macd_algo.name}{i}{j}"] = bi_out_metric / (bi_in_metric + 1e-7) - 1

        return returns

    ############################### 线段 ######################################
    def seg(self, segs, segsegs):
        def seg_begin(i, seg):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"seg_begin_klu_cnt{i}"] = klu.idx - seg.get_begin_klu().idx + 1
            returns[f"seg_begin_klc_cnt{i}"] = klu.klc.idx - seg.get_begin_klu().klc.idx + 1
            returns[f"seg_begin_amp{i}"] = seg.get_begin_val() / klu.close - 1
            returns[f"seg_begin_slope{i}"] = (seg.get_begin_val() / klu.close - 1) / (
                    klu.idx - seg.get_begin_klu().idx + 1)
            return returns

        def seg_end(i, seg):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"seg_end_klu_cnt{i}"] = klu.idx - seg.get_end_klu().idx + 1
            returns[f"seg_end_klc_cnt{i}"] = klu.klc.idx - seg.get_end_klu().klc.idx + 1
            returns[f"seg_end_val_amp{i}"] = seg.get_end_val() / klu.close - 1
            returns[f"seg_end_val_slope{i}"] = (seg.get_end_val() / klu.close - 1) / (
                    klu.idx - seg.get_end_klu().idx + 1)
            return returns

        def seg_amp(i, seg):
            returns = dict()
            returns[f"seg_amp{i}"] = seg.get_end_val() / seg.get_begin_val() - 1
            returns[f"seg_slope{i}"] = (seg.get_end_val() / seg.get_begin_val() - 1) / (
                    seg.get_end_klu().idx - seg.get_begin_klu().idx + 1)
            return returns

        def seg_high(i, seg):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"seg_high_low_amp{i}"] = (seg._high() / seg._low()) - 1
            returns[f"seg_high_amp{i}"] = (seg._high() / klu.close) - 1
            return returns

        def seg_low(i, seg):
            klu = self.chan[-1][-1]
            returns = dict()
            returns[f"seg_low_amp{i}"] = (seg._low() / klu.close) - 1
            return returns

        def seg_bi_cnt(i, seg):
            returns = dict()
            returns[f"seg_bi_cnt{i}"] = seg.cal_bi_cnt()
            return returns

        def seg_is_down(i, seg):
            returns = dict()
            returns[f"seg_is_down{i}"] = int(seg.is_down())
            return returns

        def seg_klu_cnt(i, seg):
            returns = dict()
            returns[f"seg_klu_cnt{i}"] = seg.get_klu_cnt()
            return returns

        def seg_klc_cnt(i, seg):
            returns = dict()
            returns[f"seg_klc_cnt{i}"] = seg.get_end_klu().klc.idx - seg.get_begin_klu().klc.idx + 1
            return returns

        def seg_macd(i, seg):
            returns = dict()
            returns[f"seg_macd_slope{i}"] = seg.Cal_MACD_slope()
            returns[f"seg_macd_amp{i}"] = seg.Cal_MACD_amp()
            return returns

        returns = dict()
        for i, seg in enumerate(segs):
            returns.update(seg_begin(i, seg))
            returns.update(seg_end(i, seg))
            returns.update(seg_amp(i, seg))
            returns.update(seg_high(i, seg))
            returns.update(seg_low(i, seg))
            returns.update(seg_bi_cnt(i, seg))
            # returns.update(seg_is_down(i, seg))
            returns.update(seg_klu_cnt(i, seg))
            returns.update(seg_klc_cnt(i, seg))
            returns.update(seg_macd(i, seg))

        for i, segseg in enumerate(segsegs):
            returns.update(seg_begin(f"_segseg{i}", segseg))
            returns.update(seg_end(f"_segseg{i}", segseg))
            returns.update(seg_amp(f"_segseg{i}", segseg))
            returns.update(seg_high(f"_segseg{i}", segseg))
            returns.update(seg_low(f"_segseg{i}", segseg))
            returns.update(seg_bi_cnt(f"_segseg{i}", segseg))
            # returns.update(seg_is_down(f"_segseg{i}", segseg))
            returns.update(seg_klu_cnt(f"_segseg{i}", segseg))
            returns.update(seg_klc_cnt(f"_segseg{i}", segseg))
            returns.update(seg_macd(f"_segseg{i}", segseg))
        return returns
