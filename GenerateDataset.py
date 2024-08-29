# cython: language_level=3
import os.path
from multiprocessing import Process
from typing import Dict, TypedDict

import matplotlib
import pandas

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, BSP_TYPE
from FeatureEngineering import FeatureFactors

matplotlib.use('Agg')

config = CChanConfig({
    "trigger_step": True,  # 打开开关！
    "bi_strict": True,
    "skip_step": 0,
    "divergence_rate": float("inf"),
    "bsp2_follow_1": False,
    "bsp3_follow_1": False,
    "min_zs_cnt": 0,
    "bs1_peak": True,
    "macd_algo": "peak",
    "bs_type": '1,1p',
    "print_warning": True,
    "zs_algo": "normal",
    "cal_rsi": True,
    "cal_kdj": True,
    "cal_demark": False,
    "kl_data_check": False
})
plot_config = {
    "plot_kline": False,
    "plot_kline_combine": False,
    "plot_bi": True,
    "plot_seg": False,
    "plot_eigen": False,
    "plot_zs": True,
    "plot_macd": False,
    "plot_mean": False,
    "plot_channel": False,
    "plot_bsp": False,
    "plot_extrainfo": False,
    "plot_demark": False,
    "plot_marker": False,
    "plot_rsi": False,
    "plot_kdj": False,
}

plot_para = {
    "figure": {
        "w": 224 / 10,
        "h": 224 / 50 / 2,
        "x_range": 200,
    },
    "seg": {
        # "plot_trendline": True,
        "disp_end": False,
        "end_fontsize": 10,
        "width": 0.5
    },
    "bi": {
        "show_num": False,
        "disp_end": False,
        "end_fontsize": 10,
    },
    "zs": {
        "fontsize": 10,
        "fill": True,
        "alpha": 0.5,
        "linewidth": 1
    },
    "bsp": {
        "fontsize": 10
    },
    "segseg": {
        "end_fontsize": 10,
        "width": 0.5
    },
    "seg_bsp": {
        "fontsize": 10
    },
    "marker": {
        # "markers": {  # text, position, color
        #     '2023/06/01': ('marker here', 'up', 'red'),
        #     '2023/06/08': ('marker here', 'down')
        # },
    }
}


class T_SAMPLE_INFO(TypedDict):
    bsp_type: list[BSP_TYPE]
    feature: dict


def generate_dataset(code, source_dir, lv_list, begin_time, end_time):
    """
    本demo主要演示如何记录策略产出的买卖点的特征
    然后将这些特征作为样本，训练一个模型(以XGB为demo)
    用于预测买卖点的准确性

    请注意，demo训练预测都用的是同一份数据，这是不合理的，仅仅是为了演示
    """

    data_src = DATA_SRC.FOREX
    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.NONE,
    )

    bsp_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的bsp的特征
    os.makedirs(f"{source_dir}/{code}", exist_ok=True)
    # 跑策略，保存买卖点的特征
    for chan_snapshot in chan.step_load():

        lv_chan = chan_snapshot[0]
        bsp_list = chan.get_bsp(0)  # 获取买卖点列表
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        if last_bsp.klu.idx not in bsp_dict and last_bsp.klu.klc.idx == lv_chan[-1].idx:
            print(lv_chan[-1][-1].time.to_str())
            str_date = lv_chan[-1][-1].time.to_str().replace("/", "_").replace(":", "_").replace(" ", "_")
            bsp_dict[last_bsp.klu.idx] = {
                "bsp_type": last_bsp.type,
                "feature": last_bsp.features,
            }
            factors = FeatureFactors(chan[0]).get_factors()
            for key in factors.keys():
                bsp_dict[last_bsp.klu.idx]['feature'].add_feat(key, factors[key])

    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp(0)]
    rows = []
    for bsp_klu_idx, feature_info in bsp_dict.items():
        label = int(bsp_klu_idx in bsp_academy)
        feature = feature_info["feature"]
        bsp_type = feature_info["bsp_type"]
        row = {"label": label, "bsp_type": "-".join([t.value for t in bsp_type])}
        for feature_name, value in feature.items():
            row.update({feature_name: value})
        rows.append(row)
    df = pandas.DataFrame(rows)
    df.to_csv(f"./TMP/{code}_dataset.csv", sep=",")
    return df


if __name__ == "__main__":
    symbols = [
        # Major
        "EURUSD",
        "GBPUSD",
        "AUDUSD",
        "NZDUSD",
        "USDJPY",
        "USDCAD",
        "USDCHF",
        # Crosses
        "AUDCHF",
        "AUDJPY",
        "AUDNZD",
        "CADCHF",
        "CADJPY",
        "CHFJPY",
        "EURAUD",
        "EURCAD",
        "AUDCAD",
        "EURCHF",
        "GBPNZD",
        "GBPCAD",
        "GBPCHF",
        "GBPJPY",
    ]
    lv_list = [KL_TYPE.K_30M]
    source_dir = './PNG_15_3'

    begin_time = "2010-01-01 00:00:00"
    end_time = "2021-01-01 00:00:00"
    processes = []
    for symbol in symbols:
        process = Process(target=generate_dataset, args=(symbol, source_dir, lv_list, begin_time, end_time))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
