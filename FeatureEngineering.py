# cython: language_level=3
# encoding:utf-8
from Chan import CChan
from Common.CEnum import BI_DIR, FX_TYPE, MACD_ALGO

coefficient = 100


class FeatureFactors:
    def __init__(self, chan_snapshot: CChan,
                 MAX_BI=2,
                 MAX_ZS=1,
                 MAX_SEG=1,
                 MAX_SEGSEG=1,
                 MAX_SEGZS=1):
        self.lv0_chan = chan_snapshot[0]
        # self.lv1_chan = chan_snapshot[1]
        self.MAX_BI = MAX_BI
        self.MAX_ZS = MAX_ZS
        self.MAX_SEG = MAX_SEG
        self.MAX_SEGSEG = MAX_SEGSEG
        self.MAX_SEGZS = MAX_SEGZS

    def get_factors(self):
        bis = []
        for i in range(1, self.MAX_BI + 1):
            if i < len(self.lv0_chan.bi_list):
                bi = self.lv0_chan.bi_list[-i]
                bis.append(bi)
        if self.MAX_SEG <= 0:
            segs = []
        else:
            segs = self.lv0_chan.seg_list[-min(self.MAX_SEG, len(self.lv0_chan.seg_list)):]
        if self.MAX_SEGSEG <= 0:
            segsegs = []
        else:
            segsegs = self.lv0_chan.segseg_list[-min(self.MAX_SEGSEG, len(self.lv0_chan.segseg_list)):]
        if self.MAX_ZS <= 0:
            zss = []
        else:
            zss = self.lv0_chan.zs_list[-min(self.MAX_ZS, len(self.lv0_chan.zs_list)):]
        if self.MAX_SEGZS <= 0:
            segzss = []
        else:
            segzss = self.lv0_chan.segzs_list[-min(self.MAX_SEGZS, len(self.lv0_chan.segzs_list)):]

        factors = dict()
        factors.update(self.bi(bis, self.lv0_chan[-1][-1]))
        factors.update(self.seg(segs, segsegs, self.lv0_chan[-1][-1]))
        factors.update(self.zs(zss, segzss, self.lv0_chan[-1][-1]))

        # factors1 = dict()
        # bis = []
        # for i in range(1, self.MAX_BI + 1):
        #     if i < len(self.lv1_chan.bi_list):
        #         bi = self.lv1_chan.bi_list[-i]
        #         bis.append(bi)
        # factors1.update(self.bi(bis, self.lv1_chan[-1][-1]))
        # factors = dict()
        # for key, value in factors0.items():
        #     factors.update({f"lv0_{key}": value})
        # for key, value in factors1.items():
        #     factors.update({f"lv1_{key}": value})
        return factors

    # 最后一个分型
    def fx(self):
        fx = self.lv0_chan[-1].fx
        return {"fx": list(FX_TYPE).index(fx)}

    ############################### 笔 ########################################
    def bi(self, bis, klu):

        def bi_begin(i, bi):
            factors = dict()
            factors[f"bi_begin_klu_cnt{i}"] = (klu.idx - bi.get_begin_klu().idx + 1) / coefficient
            factors[f"bi_begin_klc_cnt{i}"] = (klu.klc.idx - bi.get_begin_klu().klc.idx + 1) / coefficient
            factors[f"bi_begin_amp{i}"] = (bi.get_begin_val() / klu.close - 1) * coefficient
            factors[f"bi_begin_slope{i}"] = (bi.get_begin_val() - klu.close) * coefficient / (
                    klu.idx - bi.get_begin_klu().idx + 1)
            return factors

        def bi_end(i, bi):
            factors = dict()
            factors[f"bi_end_klu_cnt{i}"] = (klu.idx - bi.get_end_klu().idx + 1) / coefficient
            factors[f"bi_end_klc_cnt{i}"] = (klu.klc.idx - bi.get_end_klu().klc.idx + 1) / coefficient
            factors[f"bi_end_amp{i}"] = (bi.get_end_val() / klu.close - 1) * coefficient
            factors[f"bi_end_slope{i}"] = (bi.get_end_val() / klu.close - 1) * coefficient / (
                    klu.idx - bi.get_end_klu().idx + 1)
            return factors

        def bi_high(i, bi):
            factors = dict()
            factors[f"bi_high{i}"] = (bi._high() / bi._low() - 1) * coefficient
            factors[f"bi_high_amp{i}"] = (bi._high() / klu.close - 1) * coefficient
            factors[f"bi_high_low_amp{i}"] = (bi._high() / bi._low() - 1) * coefficient
            return factors

        def bi_low(i, bi):
            factors = dict()
            factors[f"bi_low_amp{i}"] = (bi._low() / klu.close - 1) * coefficient
            return factors

        def bi_mid(i, bi):
            factors = dict()
            factors[f"bi_mid_amp{i}"] = (bi._mid() / klu.close - 1) * coefficient
            return factors

        def bi_amp(i, bi):
            factors = dict()
            factors[f"bi_amp{i}"] = (bi.get_end_val() / bi.get_begin_val() - 1) * coefficient
            factors[f"bi_slope{i}"] = (
                                              bi.get_end_val() / bi.get_begin_val() - 1) * coefficient / bi.get_klu_cnt()
            return factors


        def bi_klu_cnt(i, bi):
            factors = dict()
            factors[f"bi_klu_cnt{i}"] = bi.get_klu_cnt() / coefficient
            return factors

        def bi_klc_cnt(i, bi):
            factors = dict()
            factors[f"bi_klc_cnt{i}"] = bi.get_klc_cnt() / coefficient
            return factors

        def bi_indicators(i, bi):
            factors = dict()
            for key, value in bi.get_end_klu().indicators.items():
                factors[f"bi{i}_{key}"] = value
            return factors

        factors = dict()
        for i, bi in enumerate(bis):
            factors.update(bi_begin(i, bi))
            factors.update(bi_end(i, bi))
            factors.update(bi_high(i, bi))
            factors.update(bi_low(i, bi))
            factors.update(bi_mid(i, bi))
            factors.update(bi_amp(i, bi))
            factors.update(bi_klu_cnt(i, bi))
            factors.update(bi_klc_cnt(i, bi))
            factors.update((bi_indicators(i, bi)))
        return factors

    ############################### 中枢 ####################################
    def zs(self, zss, segzss, klu):
        def zs_begin(i, zs):

            factors = dict()
            factors[f"zs_begin_klu_cnt{i}"] = (klu.idx - zs.begin.idx + 1) / coefficient
            factors[f"zs_begin_klc_cnt{i}"] = (klu.klc.idx - zs.begin.klc.idx + 1) / coefficient
            return factors

        def zs_end(i, zs):

            factors = dict()
            factors[f"zs_end_klu_cnt{i}"] = (klu.idx - zs.end.idx + 1) / coefficient
            factors[f"zs_end_klc_cnt{i}"] = (klu.klc.idx - zs.end.klc.idx + 1) / coefficient
            return factors

        def zs_klu_cnt(i, zs):
            factors = dict()
            factors[f"zs_klu_cnt{i}"] = (zs.end.idx - zs.begin.idx + 1) / coefficient
            return factors

        def zs_klc_cnt(i, zs):
            factors = dict()
            factors[f"zs_klc_cnt{i}"] = (zs.end.klc.idx - zs.begin.klc.idx + 1) / coefficient
            return factors

        def zs_high(i, zs):

            factors = dict()
            factors[f"zs_high_low_amp{i}"] = (zs.high / zs.low - 1) * coefficient
            factors[f"zs_high_amp{i}"] = (zs.high / klu.close - 1) * coefficient
            factors[f"zs_slope{i}"] = (zs.high / zs.low - 1) * coefficient / (zs.end.idx - zs.begin.idx + 1)
            return factors

        def zs_low(i, zs):

            factors = dict()
            factors[f"zs_low_amp{i}"] = (zs.low / klu.close - 1) * coefficient
            return factors

        def zs_mid(i, zs):

            factors = dict()
            factors[f"zs_mid_amp{i}"] = (zs.mid / klu.close - 1) * coefficient
            return factors

        def zs_peak_high(i, zs):

            factors = dict()
            factors[f"zs_peak_high_low_amp{i}"] = (zs.peak_high / zs.peak_low - 1) * coefficient
            factors[f"zs_peak_high_amp{i}"] = (zs.peak_high / klu.close - 1) * coefficient
            factors[f"zs_peak_high_slope{i}"] = (zs.peak_high / zs.peak_low - 1) * coefficient / (
                    zs.end.idx - zs.begin.idx + 1)
            return factors

        def zs_peak_low(i, zs):

            factors = dict()
            factors[f"zs_peak_low_amp{i}"] = (zs.peak_low / klu.close - 1) * coefficient
            return factors

        def divergence(i, zs):
            factors = dict()
            for macd_algo in [
                MACD_ALGO.AREA,
                MACD_ALGO.PEAK,
                MACD_ALGO.FULL_AREA,
                MACD_ALGO.DIFF,
                MACD_ALGO.SLOPE,
                MACD_ALGO.AMP
            ]:
                bi_in_metric = zs.bi_in.cal_macd_metric(macd_algo, is_reverse=False)
                factors[f"bi_in_{macd_algo.name}{i}"] = bi_in_metric * coefficient
                if zs.bi_out:
                    bi_out_metric = zs.bi_out.cal_macd_metric(macd_algo, is_reverse=True)
                    factors[f"bi_out_{macd_algo.name}{i}"] = bi_out_metric * coefficient
                    factors[f"divergence_{macd_algo.name}{i}"] = bi_out_metric / (bi_in_metric + 1e-7)
            return factors

        factors = dict()

        for i, zs in enumerate(zss):
            factors.update(zs_begin(f"{i}", zs))
            factors.update(zs_end(f"{i}", zs))
            factors.update(zs_klu_cnt(f"{i}", zs))
            factors.update(zs_klc_cnt(f"{i}", zs))
            factors.update(zs_high(f"{i}", zs))
            factors.update(zs_low(f"{i}", zs))
            factors.update(zs_mid(f"{i}", zs))
            factors.update(zs_peak_high(f"{i}", zs))
            factors.update(zs_peak_low(f"{i}", zs))
            factors.update(divergence(f"{i}", zs))

        for i, segzs in enumerate(segzss):
            factors.update(zs_begin(f"_segzs{i}", segzs))
            factors.update(zs_end(f"_segzs{i}", segzs))
            factors.update(zs_klu_cnt(f"_segzs{i}", segzs))
            factors.update(zs_klc_cnt(f"_segzs{i}", segzs))
            factors.update(zs_high(f"_segzs{i}", segzs))
            factors.update(zs_low(f"_segzs{i}", segzs))
            factors.update(zs_mid(f"_segzs{i}", segzs))
            factors.update(zs_peak_high(f"_segzs{i}", segzs))
            factors.update(zs_peak_low(f"_segzs{i}", segzs))
            for macd_algo in [
                MACD_ALGO.SLOPE,
                MACD_ALGO.AMP
            ]:
                bi_in_metric = segzs.bi_in.cal_macd_metric(macd_algo, is_reverse=False)
                bi_out_metric = segzs.bi_out.cal_macd_metric(macd_algo, is_reverse=True)
                factors[f"segzs_bi_in_{macd_algo.name}{i}"] = bi_in_metric * coefficient
                factors[f"segzs_bi_out_{macd_algo.name}{i}"] = bi_out_metric * coefficient
                factors[f"segzs_divergence_{macd_algo.name}{i}"] = bi_out_metric / (bi_in_metric + 1e-7)

        return factors

    ############################### 线段 ######################################
    def seg(self, segs, segsegs, klu):
        def seg_begin(i, seg):
            factors = dict()
            factors[f"seg_begin_klu_cnt{i}"] = (klu.idx - seg.get_begin_klu().idx + 1) / coefficient
            factors[f"seg_begin_klc_cnt{i}"] = (klu.klc.idx - seg.get_begin_klu().klc.idx + 1) / coefficient
            factors[f"seg_begin_amp{i}"] = (seg.get_begin_val() / klu.close - 1) * coefficient
            factors[f"seg_begin_slope{i}"] = (seg.get_begin_val() / klu.close - 1) * coefficient / (
                    klu.idx - seg.get_begin_klu().idx + 1)
            return factors

        def seg_end(i, seg):
            factors = dict()
            factors[f"seg_end_klu_cnt{i}"] = (klu.idx - seg.get_end_klu().idx + 1) / coefficient
            factors[f"seg_end_klc_cnt{i}"] = (klu.klc.idx - seg.get_end_klu().klc.idx + 1) / coefficient
            factors[f"seg_end_val_amp{i}"] = (seg.get_end_val() / klu.close - 1) * coefficient
            factors[f"seg_end_val_slope{i}"] = (seg.get_end_val() / klu.close - 1) * coefficient / (
                    klu.idx - seg.get_end_klu().idx + 1)
            return factors

        def seg_amp(i, seg):
            factors = dict()
            factors[f"seg_amp{i}"] = (seg.get_end_val() / seg.get_begin_val() - 1) * coefficient
            factors[f"seg_slope{i}"] = (seg.get_end_val() / seg.get_begin_val() - 1) * coefficient / (
                    seg.get_end_klu().idx - seg.get_begin_klu().idx + 1)
            return factors

        def seg_high(i, seg):
            factors = dict()
            factors[f"seg_high_low_amp{i}"] = (seg._high() / seg._low() - 1) * coefficient
            factors[f"seg_high_amp{i}"] = (seg._high() / klu.close - 1) * coefficient
            return factors

        def seg_low(i, seg):

            factors = dict()
            factors[f"seg_low_amp{i}"] = (seg._low() / klu.close - 1) * coefficient
            return factors

        def seg_bi_cnt(i, seg):
            factors = dict()
            factors[f"seg_bi_cnt{i}"] = seg.cal_bi_cnt() / coefficient
            return factors

        def seg_is_down(i, seg):
            factors = dict()
            factors[f"seg_is_down{i}"] = int(seg.is_down())
            return factors

        def seg_klu_cnt(i, seg):
            factors = dict()
            factors[f"seg_klu_cnt{i}"] = seg.get_klu_cnt() / coefficient
            return factors

        def seg_klc_cnt(i, seg):
            factors = dict()
            factors[f"seg_klc_cnt{i}"] = (seg.get_end_klu().klc.idx - seg.get_begin_klu().klc.idx + 1) / coefficient
            return factors

        def seg_macd(i, seg):
            factors = dict()
            factors[f"seg_macd_slope{i}"] = seg.Cal_MACD_slope() * coefficient
            factors[f"seg_macd_amp{i}"] = seg.Cal_MACD_amp() * coefficient
            return factors

        factors = dict()
        for i, seg in enumerate(segs):
            factors.update(seg_begin(i, seg))
            factors.update(seg_end(i, seg))
            factors.update(seg_amp(i, seg))
            factors.update(seg_high(i, seg))
            factors.update(seg_low(i, seg))
            factors.update(seg_bi_cnt(i, seg))
            # factors.update(seg_is_down(i, seg))
            factors.update(seg_klu_cnt(i, seg))
            factors.update(seg_klc_cnt(i, seg))
            factors.update(seg_macd(i, seg))

        for i, segseg in enumerate(segsegs):
            factors.update(seg_begin(f"_segseg{i}", segseg))
            factors.update(seg_end(f"_segseg{i}", segseg))
            factors.update(seg_amp(f"_segseg{i}", segseg))
            factors.update(seg_high(f"_segseg{i}", segseg))
            factors.update(seg_low(f"_segseg{i}", segseg))
            factors.update(seg_bi_cnt(f"_segseg{i}", segseg))
            # factors.update(seg_is_down(f"_segseg{i}", segseg))
            factors.update(seg_klu_cnt(f"_segseg{i}", segseg))
            factors.update(seg_klc_cnt(f"_segseg{i}", segseg))
            factors.update(seg_macd(f"_segseg{i}", segseg))
        return factors
