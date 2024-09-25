# cython: language_level=3
# encoding:utf-8
from Chan import CChan
from Common.CEnum import MACD_ALGO


class FeatureLib:
    def __init__(self, chan_snapshot: CChan,
                 pip_value=0.0001,
                 L0_BI=7,
                 L0_ZS=1,
                 L0_SEG=1,
                 L0_SEGSEG=1,
                 L0_SEGZS=1,
                 L1_BI=7,
                 L1_ZS=1,
                 L1_SEG=1,
                 L1_SEGSEG=1,
                 L1_SEGZS=1,
                 L2_BI=7,
                 L2_ZS=1,
                 L2_SEG=1,
                 L2_SEGSEG=1,
                 L2_SEGZS=1
                 ):
        self.lv0_chan = chan_snapshot[0]
        self.lv1_chan = chan_snapshot[1]
        self.lv2_chan = chan_snapshot[2]
        self.pip_value = pip_value
        self.L0_BI = L0_BI
        self.L0_ZS = L0_ZS
        self.L0_SEG = L0_SEG
        self.L0_SEGSEG = L0_SEGSEG
        self.L0_SEGZS = L0_SEGZS
        self.L1_BI = L1_BI
        self.L1_ZS = L1_ZS
        self.L1_SEG = L1_SEG
        self.L1_SEGSEG = L1_SEGSEG
        self.L1_SEGZS = L1_SEGZS
        self.L2_BI = L2_BI
        self.L2_ZS = L2_ZS
        self.L2_SEG = L2_SEG
        self.L2_SEGSEG = L2_SEGSEG
        self.L2_SEGZS = L2_SEGZS

    def lv_factors(self, lv_chan, lv_name, bis, segs, segsegs, zss, segzss):
        factors = dict()
        for idx, bi in enumerate(bis):
            factors.update({
                f'{lv_name}_bi{idx}_amp': self.bi_amp(bi),
                f'{lv_name}_bi{idx}_rate': self.bi_rate(bi),
                f'{lv_name}_bi{idx}_slope': self.bi_slope(bi),
                f'{lv_name}_bi{idx}_klu_cnt': self.bi_klu_cnt(bi),
                f'{lv_name}_bi{idx}_klc_cnt': self.bi_klc_cnt(bi),
                f'{lv_name}_bi{idx}_begin_klu_cnt': self.bi_begin_klu_cnt(bi, lv_chan[-1][-1]),
                f'{lv_name}_bi{idx}_begin_klc_cnt': self.bi_begin_klc_cnt(bi, lv_chan[-1][-1]),
                f'{lv_name}_bi{idx}_begin_amp': self.bi_begin_amp(bi, lv_chan[-1][-1]),
                f'{lv_name}_bi{idx}_begin_rate': self.bi_begin_rate(bi, lv_chan[-1][-1]),
                f'{lv_name}_bi{idx}_begin_slope': self.bi_begin_slope(bi, lv_chan[-1][-1]),
                f'{lv_name}_bi{idx}_end_klu_cnt': self.bi_end_klu_cnt(bi, lv_chan[-1][-1]),
                f'{lv_name}_bi{idx}_end_klc_cnt': self.bi_end_klc_cnt(bi, lv_chan[-1][-1]),
                f'{lv_name}_bi{idx}_end_amp': self.bi_end_amp(bi, lv_chan[-1][-1]),
                f'{lv_name}_bi{idx}_end_rate': self.bi_end_rate(bi, lv_chan[-1][-1]),
                f'{lv_name}_bi{idx}_end_slope': self.bi_end_slope(bi, lv_chan[-1][-1]),
            })
            if idx < len(bis) - 1:
                factors.update({
                    f'{lv_name}_bi{idx}_retrace_rate': self.bi_amp(bis[idx]) / self.bi_amp(bis[idx + 1])})
            for key, value in self.bi_macd_metric(bi).items():
                factors.update({f'{lv_name}_bi{idx}_{key}': value})
        for idx, seg in enumerate(segs):
            factors.update({
                f'{lv_name}_seg{idx}_amp': self.seg_amp(seg),
                f'{lv_name}_seg{idx}_rate': self.seg_rate(seg),
                f'{lv_name}_seg{idx}_slope': self.seg_slope(seg),
                f'{lv_name}_seg{idx}_klu_cnt': self.seg_klu_cnt(seg),
                f'{lv_name}_seg{idx}_klc_cnt': self.seg_klc_cnt(seg),
                f'{lv_name}_seg{idx}_begin_klu_cnt': self.seg_begin_klu_cnt(seg, lv_chan[-1][-1]),
                f'{lv_name}_seg{idx}_begin_klc_cnt': self.seg_begin_klc_cnt(seg, lv_chan[-1][-1]),
                f'{lv_name}_seg{idx}_begin_amp': self.seg_begin_amp(seg, lv_chan[-1][-1]),
                f'{lv_name}_seg{idx}_begin_rate': self.seg_begin_rate(seg, lv_chan[-1][-1]),
                f'{lv_name}_seg{idx}_begin_slope': self.seg_begin_slope(seg, lv_chan[-1][-1]),
                f'{lv_name}_seg{idx}_end_klu_cnt': self.seg_end_klu_cnt(seg, lv_chan[-1][-1]),
                f'{lv_name}_seg{idx}_end_klc_cnt': self.seg_end_klc_cnt(seg, lv_chan[-1][-1]),
                f'{lv_name}_seg{idx}_end_amp': self.seg_end_amp(seg, lv_chan[-1][-1]),
                f'{lv_name}_seg{idx}_end_rate': self.seg_end_rate(seg, lv_chan[-1][-1]),
                f'{lv_name}_seg{idx}_end_slope': self.seg_end_slope(seg, lv_chan[-1][-1]),
            })
            if idx < len(segs) - 1:
                factors.update({
                    f'{lv_name}_seg{idx}_retrace_rate': self.seg_amp(segs[idx]) / self.seg_amp(segs[idx + 1])})
            factors.update({
                f'{lv_name}_seg{idx}_macd_slope': self.seg_macd_slope(seg),
                f'{lv_name}_seg{idx}_macd_amp': self.seg_macd_amp(seg)})
        for idx, segseg in enumerate(segsegs):
            factors.update({
                f'{lv_name}_segseg{idx}_amp': self.seg_amp(segseg),
                f'{lv_name}_segseg{idx}_rate': self.seg_rate(segseg),
                f'{lv_name}_segseg{idx}_slope': self.seg_slope(segseg),
                f'{lv_name}_segseg{idx}_klu_cnt': self.seg_klu_cnt(segseg),
                f'{lv_name}_segseg{idx}_klc_cnt': self.seg_klc_cnt(segseg),
                f'{lv_name}_segseg{idx}_begin_klu_cnt': self.seg_begin_klu_cnt(segseg, lv_chan[-1][-1]),
                f'{lv_name}_segseg{idx}_begin_klc_cnt': self.seg_begin_klc_cnt(segseg, lv_chan[-1][-1]),
                f'{lv_name}_segseg{idx}_begin_amp': self.seg_begin_amp(segseg, lv_chan[-1][-1]),
                f'{lv_name}_segseg{idx}_begin_rate': self.seg_begin_rate(segseg, lv_chan[-1][-1]),
                f'{lv_name}_segseg{idx}_begin_slope': self.seg_begin_slope(segseg, lv_chan[-1][-1]),
                f'{lv_name}_segseg{idx}_end_klu_cnt': self.seg_end_klu_cnt(segseg, lv_chan[-1][-1]),
                f'{lv_name}_segseg{idx}_end_klc_cnt': self.seg_end_klc_cnt(segseg, lv_chan[-1][-1]),
                f'{lv_name}_segseg{idx}_end_amp': self.seg_end_amp(segseg, lv_chan[-1][-1]),
                f'{lv_name}_segseg{idx}_end_rate': self.seg_end_rate(segseg, lv_chan[-1][-1]),
                f'{lv_name}_segseg{idx}_end_slope': self.seg_end_slope(segseg, lv_chan[-1][-1]),
            })
            if idx < len(segsegs) - 1:
                factors.update({
                    f'{lv_name}_segseg{idx}_retrace_rate': self.seg_amp(segsegs[idx]) / self.seg_amp(segsegs[idx + 1])})
            factors.update({
                f'{lv_name}_segseg{idx}_macd_slope': self.seg_macd_slope(segseg),
                f'{lv_name}_segseg{idx}_macd_amp': self.seg_macd_amp(segseg)})
        for idx, zs in enumerate(zss):
            factors.update({
                f'{lv_name}_zs{idx}_amp': self.zs_amp(zs),
                f'{lv_name}_zs{idx}_rate': self.zs_rate(zs),
                f'{lv_name}_zs{idx}_slope': self.zs_slope(zs),
                f'{lv_name}_zs{idx}_klu_cnt': self.zs_klu_cnt(zs),
                f'{lv_name}_zs{idx}_klc_cnt': self.zs_klc_cnt(zs),
                f'{lv_name}_zs{idx}_begin_klu_cnt': self.zs_begin_klu_cnt(zs, lv_chan[-1][-1]),
                f'{lv_name}_zs{idx}_begin_klc_cnt': self.zs_begin_klc_cnt(zs, lv_chan[-1][-1]),
                f'{lv_name}_zs{idx}_peak_amp': self.zs_peak_amp(zs),
                f'{lv_name}_zs{idx}_peak_rate': self.zs_peak_rate(zs),
                f'{lv_name}_zs{idx}_peak_slope': self.zs_peak_slope(zs),
                f'{lv_name}_zs{idx}_end_klu_cnt': self.zs_end_klu_cnt(zs, lv_chan[-1][-1]),
                f'{lv_name}_zs{idx}_end_klc_cnt': self.zs_end_klc_cnt(zs, lv_chan[-1][-1]),
                f'{lv_name}_zs{idx}_high_amp': self.zs_high_amp(zs, lv_chan[-1][-1]),
                f'{lv_name}_zs{idx}_high_rate': self.zs_high_rate(zs, lv_chan[-1][-1]),
                f'{lv_name}_zs{idx}_low_amp': self.zs_low_amp(zs, lv_chan[-1][-1]),
                f'{lv_name}_zs{idx}_low_rate': self.zs_low_rate(zs, lv_chan[-1][-1]),
                f'{lv_name}_zs{idx}_mid_amp': self.zs_mid_amp(zs, lv_chan[-1][-1]),
                f'{lv_name}_zs{idx}_mid_rate': self.zs_mid_rate(zs, lv_chan[-1][-1]),
                f'{lv_name}_zs{idx}_peak_high_amp': self.zs_peak_high_amp(zs, lv_chan[-1][-1]),
                f'{lv_name}_zs{idx}_peak_high_rate': self.zs_peak_high_rate(zs, lv_chan[-1][-1]),
                f'{lv_name}_zs{idx}_peak_low_amp': self.zs_peak_low_amp(zs, lv_chan[-1][-1]),
                f'{lv_name}_zs{idx}_peak_low_rate': self.zs_peak_low_rate(zs, lv_chan[-1][-1]),
            })
            for key, value in self.zs_divergence(zs).items():
                factors.update({f'{lv_name}_zs{idx}_{key}': value})
        for idx, segzs in enumerate(segzss):
            factors.update({
                f'{lv_name}_segzs{idx}_amp': self.zs_amp(segzs),
                f'{lv_name}_segzs{idx}_rate': self.zs_rate(segzs),
                f'{lv_name}_segzs{idx}_slope': self.zs_slope(segzs),
                f'{lv_name}_segzs{idx}_klu_cnt': self.zs_klu_cnt(segzs),
                f'{lv_name}_segzs{idx}_klc_cnt': self.zs_klc_cnt(segzs),
                f'{lv_name}_segzs{idx}_begin_klu_cnt': self.zs_begin_klu_cnt(segzs, lv_chan[-1][-1]),
                f'{lv_name}_segzs{idx}_begin_klc_cnt': self.zs_begin_klc_cnt(segzs, lv_chan[-1][-1]),
                f'{lv_name}_segzs{idx}_peak_amp': self.zs_peak_amp(segzs),
                f'{lv_name}_segzs{idx}_peak_rate': self.zs_peak_rate(segzs),
                f'{lv_name}_segzs{idx}_peak_slope': self.zs_peak_slope(segzs),
                f'{lv_name}_segzs{idx}_end_klu_cnt': self.zs_end_klu_cnt(segzs, lv_chan[-1][-1]),
                f'{lv_name}_segzs{idx}_end_klc_cnt': self.zs_end_klc_cnt(segzs, lv_chan[-1][-1]),
                f'{lv_name}_segzs{idx}_high_amp': self.zs_high_amp(segzs, lv_chan[-1][-1]),
                f'{lv_name}_segzs{idx}_high_rate': self.zs_high_rate(segzs, lv_chan[-1][-1]),
                f'{lv_name}_segzs{idx}_low_amp': self.zs_low_amp(segzs, lv_chan[-1][-1]),
                f'{lv_name}_segzs{idx}_low_rate': self.zs_low_rate(segzs, lv_chan[-1][-1]),
                f'{lv_name}_segzs{idx}_mid_amp': self.zs_mid_amp(segzs, lv_chan[-1][-1]),
                f'{lv_name}_segzs{idx}_mid_rate': self.zs_mid_rate(segzs, lv_chan[-1][-1]),
                f'{lv_name}_segzs{idx}_peak_high_amp': self.zs_peak_high_amp(segzs, lv_chan[-1][-1]),
                f'{lv_name}_segzs{idx}_peak_high_rate': self.zs_peak_high_rate(segzs, lv_chan[-1][-1]),
                f'{lv_name}_segzs{idx}_peak_low_amp': self.zs_peak_low_amp(segzs, lv_chan[-1][-1]),
                f'{lv_name}_segzs{idx}_peak_low_rate': self.zs_peak_low_rate(segzs, lv_chan[-1][-1]),
            })
            for key, value in self.segzs_divergence(segzs).items():
                factors.update({f'{lv_name}_segzs{idx}_{key}': value})
        return factors

    def get_factors(self):

        if self.L0_BI <= 0:
            lv0_bis = []
        else:
            lv0_bis = self.lv0_chan.bi_list[-min(self.L0_BI, len(self.lv0_chan.bi_list)):][::-1]
        if self.L0_SEG <= 0:
            lv0_segs = []
        else:
            lv0_segs = self.lv0_chan.seg_list[-min(self.L0_SEG, len(self.lv0_chan.seg_list)):][::-1]
        if self.L0_SEGSEG <= 0:
            lv0_segsegs = []
        else:
            lv0_segsegs = self.lv0_chan.segseg_list[-min(self.L0_SEGSEG, len(self.lv0_chan.segseg_list)):][::-1]
        if self.L0_ZS <= 0:
            lv0_zss = []
        else:
            lv0_zss = self.lv0_chan.zs_list[-min(self.L0_ZS, len(self.lv0_chan.zs_list)):][::-1]
        if self.L0_SEGZS <= 0:
            lv0_segzss = []
        else:
            lv0_segzss = self.lv0_chan.segzs_list[-min(self.L0_SEGZS, len(self.lv0_chan.segzs_list)):][::-1]

        if self.L1_BI <= 0:
            lv1_bis = []
        else:
            lv1_bis = self.lv1_chan.bi_list[-min(self.L1_BI, len(self.lv1_chan.bi_list)):][::-1]
        if self.L1_SEG <= 0:
            lv1_segs = []
        else:
            lv1_segs = self.lv1_chan.seg_list[-min(self.L1_SEG, len(self.lv1_chan.seg_list)):][::-1]
        if self.L1_SEGSEG <= 0:
            lv1_segsegs = []
        else:
            lv1_segsegs = self.lv1_chan.segseg_list[-min(self.L1_SEGSEG, len(self.lv1_chan.segseg_list)):][::-1]
        if self.L1_ZS <= 0:
            lv1_zss = []
        else:
            lv1_zss = self.lv1_chan.zs_list[-min(self.L1_ZS, len(self.lv1_chan.zs_list)):][::-1]
        if self.L1_SEGZS <= 0:
            lv1_segzss = []
        else:
            lv1_segzss = self.lv1_chan.segzs_list[-min(self.L1_SEGZS, len(self.lv1_chan.segzs_list)):][::-1]

        if self.L2_BI <= 0:
            lv2_bis = []
        else:
            lv2_bis = self.lv2_chan.bi_list[-min(self.L2_BI, len(self.lv2_chan.bi_list)):][::-1]
        if self.L2_SEG <= 0:
            lv2_segs = []
        else:
            lv2_segs = self.lv2_chan.seg_list[-min(self.L2_SEG, len(self.lv2_chan.seg_list)):][::-1]
        if self.L2_SEGSEG <= 0:
            lv2_segsegs = []
        else:
            lv2_segsegs = self.lv2_chan.segseg_list[-min(self.L2_SEGSEG, len(self.lv2_chan.segseg_list)):][::-1]
        if self.L2_ZS <= 0:
            lv2_zss = []
        else:
            lv2_zss = self.lv2_chan.zs_list[-min(self.L2_ZS, len(self.lv2_chan.zs_list)):][::-1]
        if self.L2_SEGZS <= 0:
            lv2_segzss = []
        else:
            lv2_segzss = self.lv2_chan.segzs_list[-min(self.L2_SEGZS, len(self.lv2_chan.segzs_list)):][::-1]

        factors = dict()
        for key, value in self.lv0_chan[-1][-1].indicators.items():
            factors.update({f"lv0_last_klu_{key}": value})
        for key, value in self.lv1_chan[-1][-1].indicators.items():
            factors.update({f"lv1_last_klu_{key}": value})
        for key, value in self.lv2_chan[-1][-1].indicators.items():
            factors.update({f"lv2_last_klu_{key}": value})
        lv0_factors = self.lv_factors(self.lv0_chan, "lv0", lv0_bis, lv0_segs, lv0_segsegs, lv0_zss, lv0_segzss)
        lv1_factors = self.lv_factors(self.lv1_chan, "lv1", lv1_bis, lv1_segs, lv1_segsegs, lv1_zss, lv1_segzss)
        lv2_factors = self.lv_factors(self.lv2_chan, "lv2", lv2_bis, lv2_segs, lv2_segsegs, lv2_zss, lv2_segzss)
        factors.update(lv0_factors)
        factors.update(lv1_factors)
        factors.update(lv2_factors)
        return factors

    ############################### 笔 ########################################
    def bi_amp(self, bi):
        return (bi.get_end_val() - bi.get_begin_val()) / self.pip_value

    def bi_rate(self, bi):
        return (bi.get_end_val() / bi.get_begin_val() - 1) / self.pip_value

    def bi_slope(self, bi):
        return (bi.get_end_val() - bi.get_begin_val()) / self.pip_value / bi.get_klu_cnt()

    def bi_klu_cnt(self, bi):
        return bi.get_klu_cnt()

    def bi_klc_cnt(self, bi):
        return bi.get_klc_cnt()

    def bi_begin_klu_cnt(self, bi, klu):
        return klu.idx - bi.get_begin_klu().idx + 1

    def bi_begin_klc_cnt(self, bi, klu):
        return klu.klc.idx - bi.get_begin_klu().klc.idx + 1

    def bi_begin_amp(self, bi, klu):
        return (bi.get_begin_val() - klu.close) / self.pip_value

    def bi_begin_rate(self, bi, klu):
        return (bi.get_begin_val() / klu.close - 1) / self.pip_value

    def bi_begin_slope(self, bi, klu):
        return (bi.get_begin_val() - klu.close) / self.pip_value / (klu.idx - bi.get_begin_klu().idx + 1)

    def bi_end_klu_cnt(self, bi, klu):
        return klu.idx - bi.get_end_klu().idx + 1

    def bi_end_klc_cnt(self, bi, klu):
        return klu.klc.idx - bi.get_end_klu().klc.idx + 1

    def bi_end_amp(self, bi, klu):
        return (bi.get_end_val() - klu.close) / self.pip_value

    def bi_end_rate(self, bi, klu):
        return (bi.get_end_val() / klu.close - 1) / self.pip_value

    def bi_end_slope(self, bi, klu):
        return (bi.get_end_val() - klu.close) / self.pip_value / (klu.idx - bi.get_end_klu().idx + 1)

    def bi_macd_metric(self, bi):
        factors = dict()
        for macd_algo in [
            MACD_ALGO.AREA,
            MACD_ALGO.PEAK,
            MACD_ALGO.FULL_AREA,
            MACD_ALGO.DIFF,
            MACD_ALGO.SLOPE,
            MACD_ALGO.AMP
        ]:
            macd_metric = bi.cal_macd_metric(macd_algo, is_reverse=False)
            factors[f"macd_{macd_algo.name}"] = macd_metric / self.pip_value
        return factors

    def bi_end_klu_indicators(self, bi):
        return bi.get_end_klu().indicators.items()

    ############################### 线段 ######################################
    def seg_amp(self, seg):
        return (seg.get_end_val() - seg.get_begin_val()) / self.pip_value

    def seg_rate(self, seg):
        return (seg.get_end_val() / seg.get_begin_val() - 1) / self.pip_value

    def seg_slope(self, seg):
        return (seg.get_end_val() - seg.get_begin_val()) / self.pip_value / (
                seg.get_end_klu().idx - seg.get_begin_klu().idx + 1)

    def seg_bi_cnt(self, seg):
        return seg.cal_bi_cnt()

    def seg_is_down(self, seg):
        return int(seg.is_down())

    def seg_klu_cnt(self, seg):
        return seg.get_klu_cnt()

    def seg_klc_cnt(self, seg):
        return seg.get_end_klu().klc.idx - seg.get_begin_klu().klc.idx + 1

    def seg_begin_klu_cnt(self, seg, klu):
        return klu.idx - seg.get_begin_klu().idx + 1

    def seg_begin_klc_cnt(self, seg, klu):
        return klu.klc.idx - seg.get_begin_klu().klc.idx + 1

    def seg_begin_amp(self, seg, klu):
        return (seg.get_begin_val() - klu.close) / self.pip_value

    def seg_begin_rate(self, seg, klu):
        return (seg.get_begin_val() / klu.close - 1) / self.pip_value

    def seg_begin_slope(self, seg, klu):
        return (seg.get_begin_val() - klu.close) / self.pip_value / (klu.idx - seg.get_begin_klu().idx + 1)

    def seg_end_klu_cnt(self, seg, klu):
        return klu.idx - seg.get_end_klu().idx + 1

    def seg_end_klc_cnt(self, seg, klu):
        return klu.klc.idx - seg.get_end_klu().klc.idx + 1

    def seg_end_amp(self, seg, klu):
        return (seg.get_end_val() - klu.close) / self.pip_value

    def seg_end_rate(self, seg, klu):
        return (seg.get_end_val() / klu.close - 1) / self.pip_value

    def seg_end_slope(self, seg, klu):
        return (seg.get_end_val() - klu.close) / self.pip_value / (klu.idx - seg.get_end_klu().idx + 1)

    def seg_macd_slope(self, seg):
        return seg.Cal_MACD_slope() / self.pip_value

    def seg_macd_amp(self, seg):
        return seg.Cal_MACD_amp() / self.pip_value

    def zs_klu_cnt(self, zs):
        return zs.end.idx - zs.begin.idx + 1

    def zs_klc_cnt(self, zs):
        return zs.end.klc.idx - zs.begin.klc.idx + 1

    def zs_amp(self, zs):
        return (zs.high - zs.low) / self.pip_value

    def zs_rate(self, zs):
        return (zs.high / zs.low - 1) / self.pip_value

    def zs_slope(self, zs):
        return (zs.high - zs.low) / self.pip_value / (zs.end.idx - zs.begin.idx + 1)

    def zs_peak_amp(self, zs):
        return (zs.peak_high - zs.peak_low) / self.pip_value

    def zs_peak_rate(self, zs):
        return (zs.peak_high / zs.peak_low - 1) / self.pip_value

    def zs_peak_slope(self, zs):
        return (zs.peak_high - zs.peak_low) / self.pip_value / (zs.end.idx - zs.begin.idx + 1)

    def zs_begin_klu_cnt(self, zs, klu):
        return klu.idx - zs.begin.idx + 1

    def zs_begin_klc_cnt(self, zs, klu):
        return klu.klc.idx - zs.begin.klc.idx + 1

    def zs_end_klu_cnt(self, zs, klu):
        return klu.idx - zs.end.idx + 1

    def zs_end_klc_cnt(self, zs, klu):
        return klu.klc.idx - zs.end.klc.idx + 1

    def zs_high_amp(self, zs, klu):
        return (zs.high - klu.close) / self.pip_value

    def zs_high_rate(self, zs, klu):
        return (zs.high / klu.close - 1) / self.pip_value

    def zs_low_amp(self, zs, klu):
        return (zs.low - klu.close) / self.pip_value

    def zs_low_rate(self, zs, klu):
        return (zs.low / klu.close - 1) / self.pip_value

    def zs_mid_amp(self, zs, klu):
        return (zs.mid - klu.close) / self.pip_value

    def zs_mid_rate(self, zs, klu):
        return (zs.mid / klu.close - 1) / self.pip_value

    def zs_peak_high_amp(self, zs, klu):
        return (zs.peak_high - klu.close) / self.pip_value

    def zs_peak_high_rate(self, zs, klu):
        return (zs.peak_high / klu.close - 1) / self.pip_value

    def zs_peak_low_amp(self, zs, klu):
        return (zs.peak_low - klu.close - 1) / self.pip_value

    def zs_peak_low_rate(self, zs, klu):
        return (zs.peak_low / klu.close - 1) / self.pip_value

    def zs_divergence(self, zs):
        for macd_algo in [
            MACD_ALGO.AREA,
            MACD_ALGO.PEAK,
            MACD_ALGO.FULL_AREA,
            MACD_ALGO.DIFF,
            MACD_ALGO.SLOPE,
            MACD_ALGO.AMP
        ]:
            factors = dict()
            bi_in_metric = zs.bi_in.cal_macd_metric(macd_algo, is_reverse=False)
            factors[f"bi_in_{macd_algo.name}"] = bi_in_metric / self.pip_value
            if zs.bi_out:
                bi_out_metric = zs.bi_out.cal_macd_metric(macd_algo, is_reverse=True)
                factors[f"bi_out_{macd_algo.name}"] = bi_out_metric / self.pip_value
                factors[f"divergence_{macd_algo.name}"] = bi_out_metric / (bi_in_metric + 1e-7)
        return factors

    def segzs_divergence(self, zs):
        for macd_algo in [
            MACD_ALGO.SLOPE,
            MACD_ALGO.AMP
        ]:
            factors = dict()
            bi_in_metric = zs.bi_in.cal_macd_metric(macd_algo, is_reverse=False)
            factors[f"bi_in_{macd_algo.name}"] = bi_in_metric / self.pip_value
            if zs.bi_out:
                bi_out_metric = zs.bi_out.cal_macd_metric(macd_algo, is_reverse=True)
                factors[f"bi_out_{macd_algo.name}"] = bi_out_metric / self.pip_value
                factors[f"divergence_{macd_algo.name}"] = bi_out_metric / (bi_in_metric + 1e-7)
        return factors
