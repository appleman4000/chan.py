# cython: language_level=3
# encoding:utf-8
from Chan import CChan
from Common.CEnum import MACD_ALGO


class FeatureLib:
    def __init__(self, chan_snapshot: CChan,
                 pip_value=0.0001,
                 MAX_BI=7,
                 MAX_ZS=1,
                 MAX_SEG=1,
                 MAX_SEGSEG=1,
                 MAX_SEGZS=1):
        self.lv0_chan = chan_snapshot[0]
        self.pip_value = pip_value
        self.MAX_BI = MAX_BI
        self.MAX_ZS = MAX_ZS
        self.MAX_SEG = MAX_SEG
        self.MAX_SEGSEG = MAX_SEGSEG
        self.MAX_SEGZS = MAX_SEGZS

    def get_factors(self):
        klu = self.lv0_chan[-1][-1]
        if self.MAX_BI <= 0:
            bis = []
        else:
            bis = self.lv0_chan.bi_list[-min(self.MAX_BI, len(self.lv0_chan.bi_list)):][::-1]
        if self.MAX_SEG <= 0:
            segs = []
        else:
            segs = self.lv0_chan.seg_list[-min(self.MAX_SEG, len(self.lv0_chan.seg_list)):][::-1]
        if self.MAX_SEGSEG <= 0:
            segsegs = []
        else:
            segsegs = self.lv0_chan.segseg_list[-min(self.MAX_SEGSEG, len(self.lv0_chan.segseg_list)):][::-1]
        if self.MAX_ZS <= 0:
            zss = []
        else:
            zss = self.lv0_chan.zs_list[-min(self.MAX_ZS, len(self.lv0_chan.zs_list)):][::-1]
        if self.MAX_SEGZS <= 0:
            segzss = []
        else:
            segzss = self.lv0_chan.segzs_list[-min(self.MAX_SEGZS, len(self.lv0_chan.segzs_list)):][::-1]

        factors = dict()
        for idx, bi in enumerate(bis):
            factors.update({
                f'bi{idx}_amp': self.bi_amp(bi),
                f'bi{idx}_rate': self.bi_rate(bi),
                f'bi{idx}_slope': self.bi_slope(bi),
                f'bi{idx}_klu_cnt': self.bi_klu_cnt(bi),
                f'bi{idx}_klc_cnt': self.bi_klc_cnt(bi),
                f'bi{idx}_begin_klu_cnt': self.bi_begin_klu_cnt(bi, klu),
                f'bi{idx}_begin_klc_cnt': self.bi_begin_klc_cnt(bi, klu),
                f'bi{idx}_begin_amp': self.bi_begin_amp(bi, klu),
                f'bi{idx}_begin_rate': self.bi_begin_rate(bi, klu),
                f'bi{idx}_begin_slope': self.bi_begin_slope(bi, klu),
                f'bi{idx}_end_klu_cnt': self.bi_end_klu_cnt(bi, klu),
                f'bi{idx}_end_klc_cnt': self.bi_end_klc_cnt(bi, klu),
                f'bi{idx}_end_amp': self.bi_end_amp(bi, klu),
                f'bi{idx}_end_rate': self.bi_end_rate(bi, klu),
                f'bi{idx}_end_slope': self.bi_end_slope(bi, klu),
            })
            if idx < len(bis) - 1:
                factors.update({
                    f'bi{idx}_retrace_rate': self.bi_amp(bis[idx]) / self.bi_amp(bis[idx + 1])})
            for key, value in self.bi_macd_metric(bi).items():
                factors.update({f'bi{idx}_{key}': value})
        for idx, seg in enumerate(segs):
            factors.update({
                f'seg{idx}_amp': self.seg_amp(seg),
                f'seg{idx}_rate': self.seg_rate(seg),
                f'seg{idx}_slope': self.seg_slope(seg),
                f'seg{idx}_klu_cnt': self.seg_klu_cnt(seg),
                f'seg{idx}_klc_cnt': self.seg_klc_cnt(seg),
                f'seg{idx}_begin_klu_cnt': self.seg_begin_klu_cnt(seg, klu),
                f'seg{idx}_begin_klc_cnt': self.seg_begin_klc_cnt(seg, klu),
                f'seg{idx}_begin_amp': self.seg_begin_amp(seg, klu),
                f'seg{idx}_begin_rate': self.seg_begin_rate(seg, klu),
                f'seg{idx}_begin_slope': self.seg_begin_slope(seg, klu),
                f'seg{idx}_end_klu_cnt': self.seg_end_klu_cnt(seg, klu),
                f'seg{idx}_end_klc_cnt': self.seg_end_klc_cnt(seg, klu),
                f'seg{idx}_end_amp': self.seg_end_amp(seg, klu),
                f'seg{idx}_end_rate': self.seg_end_rate(seg, klu),
                f'seg{idx}_end_slope': self.seg_end_slope(seg, klu),
            })
            if idx < len(segs) - 1:
                factors.update({
                    f'seg{idx}_retrace_rate': self.seg_amp(segs[idx]) / self.seg_amp(segs[idx + 1])})
            factors.update({
                f'seg{idx}_macd_slope': self.seg_macd_slope(seg),
                f'seg{idx}_macd_amp': self.seg_macd_amp(seg)})
        for idx, segseg in enumerate(segsegs):
            factors.update({
                f'segseg{idx}_amp': self.seg_amp(segseg),
                f'segseg{idx}_rate': self.seg_rate(segseg),
                f'segseg{idx}_slope': self.seg_slope(segseg),
                f'segseg{idx}_klu_cnt': self.seg_klu_cnt(segseg),
                f'segseg{idx}_klc_cnt': self.seg_klc_cnt(segseg),
                f'segseg{idx}_begin_klu_cnt': self.seg_begin_klu_cnt(segseg, klu),
                f'segseg{idx}_begin_klc_cnt': self.seg_begin_klc_cnt(segseg, klu),
                f'segseg{idx}_begin_amp': self.seg_begin_amp(segseg, klu),
                f'segseg{idx}_begin_rate': self.seg_begin_rate(segseg, klu),
                f'segseg{idx}_begin_slope': self.seg_begin_slope(segseg, klu),
                f'segseg{idx}_end_klu_cnt': self.seg_end_klu_cnt(segseg, klu),
                f'segseg{idx}_end_klc_cnt': self.seg_end_klc_cnt(segseg, klu),
                f'segseg{idx}_end_amp': self.seg_end_amp(segseg, klu),
                f'segseg{idx}_end_rate': self.seg_end_rate(segseg, klu),
                f'segseg{idx}_end_slope': self.seg_end_slope(segseg, klu),
            })
            if idx < len(segsegs) - 1:
                factors.update({
                    f'segseg{idx}_retrace_rate': self.seg_amp(segsegs[idx]) / self.seg_amp(segsegs[idx + 1])})
            factors.update({
                f'segseg{idx}_macd_slope': self.seg_macd_slope(segseg),
                f'segseg{idx}_macd_amp': self.seg_macd_amp(segseg)})
        for idx, zs in enumerate(zss):
            factors.update({
                f'zs{idx}_amp': self.zs_amp(zs),
                f'zs{idx}_rate': self.zs_rate(zs),
                f'zs{idx}_slope': self.zs_slope(zs),
                f'zs{idx}_klu_cnt': self.zs_klu_cnt(zs),
                f'zs{idx}_klc_cnt': self.zs_klc_cnt(zs),
                f'zs{idx}_begin_klu_cnt': self.zs_begin_klu_cnt(zs, klu),
                f'zs{idx}_begin_klc_cnt': self.zs_begin_klc_cnt(zs, klu),
                f'zs{idx}_peak_amp': self.zs_peak_amp(zs),
                f'zs{idx}_peak_rate': self.zs_peak_rate(zs),
                f'zs{idx}_peak_slope': self.zs_peak_slope(zs),
                f'zs{idx}_end_klu_cnt': self.zs_end_klu_cnt(zs, klu),
                f'zs{idx}_end_klc_cnt': self.zs_end_klc_cnt(zs, klu),
                f'zs{idx}_high_amp': self.zs_high_amp(zs, klu),
                f'zs{idx}_high_rate': self.zs_high_rate(zs, klu),
                f'zs{idx}_low_amp': self.zs_low_amp(zs, klu),
                f'zs{idx}_low_rate': self.zs_low_rate(zs, klu),
                f'zs{idx}_mid_amp': self.zs_mid_amp(zs, klu),
                f'zs{idx}_mid_rate': self.zs_mid_rate(zs, klu),
                f'zs{idx}_peak_high_amp': self.zs_peak_high_amp(zs, klu),
                f'zs{idx}_peak_high_rate': self.zs_peak_high_rate(zs, klu),
                f'zs{idx}_peak_low_amp': self.zs_peak_low_amp(zs, klu),
                f'zs{idx}_peak_low_rate': self.zs_peak_low_rate(zs, klu),
            })
            for key, value in self.zs_divergence(zs).items():
                factors.update({f'zs{idx}_{key}': value})
        for idx, segzs in enumerate(segzss):
            factors.update({
                f'segzs{idx}_amp': self.zs_amp(segzs),
                f'segzs{idx}_rate': self.zs_rate(segzs),
                f'segzs{idx}_slope': self.zs_slope(segzs),
                f'segzs{idx}_klu_cnt': self.zs_klu_cnt(segzs),
                f'segzs{idx}_klc_cnt': self.zs_klc_cnt(segzs),
                f'segzs{idx}_begin_klu_cnt': self.zs_begin_klu_cnt(segzs, klu),
                f'segzs{idx}_begin_klc_cnt': self.zs_begin_klc_cnt(segzs, klu),
                f'segzs{idx}_peak_amp': self.zs_peak_amp(segzs),
                f'segzs{idx}_peak_rate': self.zs_peak_rate(segzs),
                f'segzs{idx}_peak_slope': self.zs_peak_slope(segzs),
                f'segzs{idx}_end_klu_cnt': self.zs_end_klu_cnt(segzs, klu),
                f'segzs{idx}_end_klc_cnt': self.zs_end_klc_cnt(segzs, klu),
                f'segzs{idx}_high_amp': self.zs_high_amp(segzs, klu),
                f'segzs{idx}_high_rate': self.zs_high_rate(segzs, klu),
                f'segzs{idx}_low_amp': self.zs_low_amp(segzs, klu),
                f'segzs{idx}_low_rate': self.zs_low_rate(segzs, klu),
                f'segzs{idx}_mid_amp': self.zs_mid_amp(segzs, klu),
                f'segzs{idx}_mid_rate': self.zs_mid_rate(segzs, klu),
                f'segzs{idx}_peak_high_amp': self.zs_peak_high_amp(segzs, klu),
                f'segzs{idx}_peak_high_rate': self.zs_peak_high_rate(segzs, klu),
                f'segzs{idx}_peak_low_amp': self.zs_peak_low_amp(segzs, klu),
                f'segzs{idx}_peak_low_rate': self.zs_peak_low_rate(segzs, klu),
            })
            for key, value in self.zs_divergence(zs).items():
                factors.update({f'segzs{idx}_{key}': value})
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
