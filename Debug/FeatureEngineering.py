# encoding:utf-8
from Chan import CChan
from Common.CEnum import MACD_ALGO, BSP_TYPE

N_BI = 2
N_ZS = 2
N_SEG = 2


class FeatureFactors:
    def __init__(self, chan: CChan):
        self.chan = chan

    def open_klu_rate(self):
        klu = self.chan[0][-1][-1]
        return {
            "open_klu_amp": klu.close - klu.open,
            "open_klu_rate": klu.close / klu.open,
        }

    def bsp_type(self):
        bsp_list = self.chan.get_bsp()
        if not bsp_list:
            return {"bsp_type": None}
        bsp = bsp_list[-1]
        return {"bsp_type": list(BSP_TYPE).index(bsp.type[0])}

    # def fx(self,chan: CChan):
    #     fx = self.chan[0][-1].fx
    #     return {"fx": list(FX_TYPE).index(fx)}

    ############################### 笔 ####################################################
    # def bi_dir(self,chan: CChan):
    #     returns = dict()
    #     for i in range(1, N_BI + 1):
    #         bi = self.chan[0].bi_list[-i]
    #         returns[f"bi_dir{i}"] = list(BI_DIR).index(bi.dir)
    #     return returns

    # def bi_is_sure(self,chan: CChan):
    #     returns = dict()
    #     for i in range(1, N_BI + 1):
    #         bi = self.chan[0].bi_list[-i]
    #         returns[f"bi_is_sure{i}"] = int(bi.is_sure)
    #     return returns

    def bi_high(self):
        klu = self.chan[0][-1][-1]
        returns = dict()
        for i in range(1, N_BI + 1):
            bi = self.chan[0].bi_list[-i]
            returns[f"bi_high_amp{i}"] = bi._high() - klu.close
            returns[f"bi_high_rate{i}"] = bi._high() / klu.close
        return returns

    def bi_low(self):
        klu = self.chan[0][-1][-1]
        returns = dict()
        for i in range(1, N_BI + 1):
            bi = self.chan[0].bi_list[-i]
            returns[f"bi_low_amp{i}"] = bi._low() - klu.close
            returns[f"bi_low_rate{i}"] = bi._low() / klu.close
        return returns

    def bi_mid(self):
        klu = self.chan[0][-1][-1]
        returns = dict()
        for i in range(1, N_BI + 1):
            bi = self.chan[0].bi_list[-i]
            returns[f"bi_mid_amp{i}"] = bi._mid() - klu.close
            returns[f"bi_mid_rate{i}"] = bi._mid() / klu.close
        return returns

    def bi_begin(self):
        klu = self.chan[0][-1][-1]
        returns = dict()
        for i in range(1, N_BI + 1):
            bi = self.chan[0].bi_list[-i]
            returns[f"bi_begin_amp{i}"] = bi.get_begin_val() - klu.close
            returns[f"bi_begin_rate{i}"] = bi.get_begin_val() / klu.close
            returns[f"bi_begin_slope{i}"] = (bi.get_begin_val() - klu.close) / (klu.idx - bi.get_begin_klu().idx + 1)
        return returns

    def bi_end(self):
        klu = self.chan[0][-1][-1]
        returns = dict()
        for i in range(1, N_BI + 1):
            bi = self.chan[0].bi_list[-i]
            returns[f"bi_end_amp{i}"] = bi.get_end_val() - klu.close
            returns[f"bi_end_rate{i}"] = bi.get_end_val() / klu.close
            returns[f"bi_end_slope{i}"] = (bi.get_end_val() - klu.close) / (klu.idx - bi.get_end_klu().idx + 1)
        return returns

    def bi_amp_rate_slope(self):
        returns = dict()
        for i in range(1, N_BI + 1):
            bi = self.chan[0].bi_list[-i]
            returns[f"bi_amp{i}"] = bi.get_end_val() - bi.get_begin_val()
            returns[f"bi_rate{i}"] = bi.get_end_val() / bi.get_begin_val()
            returns[f"bi_slope{i}"] = (bi.get_end_val() - bi.get_begin_val()) / bi.get_klu_cnt()
        return returns

    def bi_klu_cnt(self):
        returns = dict()
        for i in range(1, N_BI + 1):
            bi = self.chan[0].bi_list[-i]
            returns[f"bi_klu_cnt{i}"] = bi.get_klu_cnt()
        return returns

    def bi_klc_cnt(self):
        returns = dict()
        for i in range(1, N_BI + 1):
            bi = self.chan[0].bi_list[-i]
            returns[f"bi_klc_cnt{i}"] = bi.get_klc_cnt()
        return returns

    def divergence_macd_metric(self):
        returns = dict()
        for macd_algo in [MACD_ALGO.AREA,
                          MACD_ALGO.PEAK,
                          MACD_ALGO.FULL_AREA,
                          MACD_ALGO.DIFF,
                          MACD_ALGO.SLOPE,
                          MACD_ALGO.AMP]:
            bi_in_metric = self.chan[0].zs_list[-1].bi_in.cal_macd_metric(macd_algo, is_reverse=False)
            bi_out_metric = self.chan[0].zs_list[-1].bi_out.cal_macd_metric(macd_algo, is_reverse=True)
            returns[f"bi_in_divergence_macd_metric{macd_algo.name}"] = bi_in_metric
            returns[f"bi_out_divergence_macd_metric{macd_algo.name}"] = bi_out_metric
            returns[f"divergence_macd_metric{macd_algo.name}"] = bi_out_metric / bi_in_metric
        return returns

    ############################### 中枢 ####################################################
    def zs_high(self):
        klu = self.chan[0][-1][-1]
        returns = dict()
        for i in range(1, N_ZS + 1):
            zs = self.chan[0].zs_list[-i]
            returns[f"zs_high_amp{i}"] = zs.high - klu.close
            returns[f"zs_high_rate{i}"] = zs.high / klu.close
        return returns

    def zs_low(self):
        klu = self.chan[0][-1][-1]
        returns = dict()
        for i in range(1, N_ZS + 1):
            zs = self.chan[0].zs_list[-i]
            returns[f"zs_low_amp{i}"] = zs.low - klu.close
            returns[f"zs_low_rate{i}"] = zs.low / klu.close
        return returns

    def zs_mid(self):
        klu = self.chan[0][-1][-1]
        returns = dict()
        for i in range(1, N_ZS + 1):
            zs = self.chan[0].zs_list[-i]
            returns[f"zs_mid_amp{i}"] = zs.mid - klu.close
            returns[f"zs_mid_rate{i}"] = zs.mid / klu.close
        return returns

    def zs_peak_high(self):
        klu = self.chan[0][-1][-1]
        returns = dict()
        for i in range(1, N_ZS + 1):
            zs = self.chan[0].zs_list[-i]
            returns[f"zs_peak_high_amp{i}"] = zs.peak_high - klu.close
            returns[f"zs_peak_high_rate{i}"] = zs.peak_high / klu.close
        return returns

    def zs_peak_low(self):
        klu = self.chan[0][-1][-1]
        returns = dict()
        for i in range(1, N_ZS + 1):
            zs = self.chan[0].zs_list[-i]
            returns[f"zs_peak_low_amp{i}"] = zs.peak_low - klu.close
            returns[f"zs_peak_low_rate{i}"] = zs.peak_low / klu.close

        return returns

    ############################### 线段 ####################################################
    def seg_amp_slope(self):
        klu = self.chan[0][-1][-1]
        returns = dict()
        for i in range(1, N_SEG + 1):
            seg = self.chan[0].seg_list[-i]
            returns[f"seg_amp{i}"] = seg.get_end_val() - seg.get_begin_val()
            returns[f"seg_rate{i}"] = seg.get_end_val() / seg.get_begin_val()
            returns[f"seg_slope{i}"] = (seg.get_end_val() - seg.get_begin_val()) / (
                    seg.get_end_klu().idx - seg.get_begin_klu().idx)
        return returns

    def seg_bi_cnt(self):
        returns = dict()
        for i in range(1, N_SEG + 1):
            seg = self.chan[0].seg_list[-i]
            returns[f"seg_bi_cnt{i}"] = seg.cal_bi_cnt()
        return returns

    def seg_low(self):
        klu = self.chan[0][-1][-1]
        returns = dict()
        for i in range(1, N_SEG + 1):
            seg = self.chan[0].seg_list[-i]
            returns[f"seg_low_amp{i}"] = seg._low() - klu.close
            returns[f"seg_low_rate{i}"] = seg._low() / klu.close
        return returns

    def seg_high(self):
        klu = self.chan[0][-1][-1]
        returns = dict()
        for i in range(1, N_SEG + 1):
            seg = self.chan[0].seg_list[-i]
            returns[f"seg_high_amp{i}"] = seg._high() - klu.close
            returns[f"seg_high_rate{i}"] = seg._high() / klu.close
        return returns

    def seg_is_down(self):
        klu = self.chan[0][-1][-1]
        returns = dict()
        for i in range(1, N_SEG + 1):
            seg = self.chan[0].seg_list[-i]
            returns[f"seg_is_down{i}"] = int(seg.is_down())
        return returns

    def seg_begin_val(self):
        klu = self.chan[0][-1][-1]
        returns = dict()
        for i in range(1, N_SEG + 1):
            seg = self.chan[0].seg_list[-i]
            returns[f"seg_begin_val_amp{i}"] = seg.get_begin_val() - klu.close
            returns[f"seg_begin_val_rate{i}"] = seg.get_begin_val() / klu.close
            returns[f"seg_begin_val_slope{i}"] = (seg.get_begin_val() - klu.close) / (
                    klu.idx - seg.get_begin_klu().idx + 1)
        return returns

    def seg_end_val(self):
        klu = self.chan[0][-1][-1]
        returns = dict()
        for i in range(1, N_SEG + 1):
            seg = self.chan[0].seg_list[-i]
            returns[f"seg_end_val_amp{i}"] = seg.get_end_val() - klu.close
            returns[f"seg_end_val_rate{i}"] = seg.get_end_val() / klu.close
            returns[f"seg_end_val_slope{i}"] = (seg.get_end_val() - klu.close) / (klu.idx - seg.get_end_klu().idx + 1)
        return returns

    def seg_klu_cnt(self):
        returns = dict()
        for i in range(1, N_SEG + 1):
            seg = self.chan[0].seg_list[-i]
            returns[f"seg_klu_cnt{i}"] = seg.get_klu_cnt()
        return returns

    def seg_macd_slope(self):
        returns = dict()
        for i in range(1, N_SEG + 1):
            seg = self.chan[0].seg_list[-i]
            returns[f"seg_macd_slope{i}"] = seg.Cal_MACD_slope()
        return returns

    def seg_macd_amp(self):
        returns = dict()
        for i in range(1, N_SEG + 1):
            seg = self.chan[0].seg_list[-i]
            returns[f"seg_macd_amp{i}"] = seg.Cal_MACD_amp()
        return returns
