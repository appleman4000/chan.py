# cython: language_level=3
import csv
import json
import os.path
from typing import Dict, TypedDict

import matplotlib
from matplotlib import pyplot as plt

from BuySellPoint.BS_Point import CBS_Point
from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, BSP_TYPE
from CommonTools import chan_to_png
from Plot.PlotDriver import CPlotDriver

matplotlib.use('Agg')

config = CChanConfig({
    "trigger_step": True,  # 打开开关！
    "bi_strict": True,
    "skip_step": 0,
    "divergence_rate": 0.9,
    "bsp2_follow_1": True,
    "bsp3_follow_1": True,
    "min_zs_cnt": 1,
    "bs1_peak": False,
    "macd_algo": "peak",
    "bs_type": '1,2,3a,1p,2s,3b',
    "print_warning": True,
    "zs_algo": "normal",
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
        "w": 224 / 50,
        "h": 224 / 50 / 2,
        "x_range": 120,
    },
    "seg": {
        # "plot_trendline": True,
        "disp_end": False,
        "end_fontsize": 15,
        "width": 0.5
    },
    "bi": {
        "show_num": False,
        "disp_end": False,
        "end_fontsize": 15,
    },
    "zs": {
        "fontsize": 15,
    },
    "bsp": {
        "fontsize": 20
    },
    "segseg": {
        "end_fontsize": 15,
        "width": 0.5
    },
    "seg_bsp": {
        "fontsize": 20
    },
    "marker": {
        # "markers": {  # text, position, color
        #     '2023/06/01': ('marker here', 'up', 'red'),
        #     '2023/06/08': ('marker here', 'down')
        # },
    }
}


class T_SAMPLE_INFO(TypedDict):
    last_bsp: CBS_Point
    features: str
    file_path: str


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
    os.makedirs(source_dir, exist_ok=True)
    # 跑策略，保存买卖点的特征
    for chan_snapshot in chan.step_load():

        lv_chan = chan_snapshot[0]
        bsp_list = chan.get_bsp(0)  # 获取高级别买卖点列表
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]
        if BSP_TYPE.T1 not in last_bsp.type and BSP_TYPE.T1P not in last_bsp.type \
                and BSP_TYPE.T2 not in last_bsp.type and BSP_TYPE.T2S not in last_bsp.type \
                and BSP_TYPE.T3A not in last_bsp.type and BSP_TYPE.T3B not in last_bsp.type:  # 假如只做2类买卖点
            continue
        cur_lv_chan = chan_snapshot[0]
        if last_bsp.klu.idx not in bsp_dict and cur_lv_chan[-2].idx == last_bsp.klu.klc.idx:
            # 假如策略是：买卖点分形第二元素出现时交易
            str_date = lv_chan[-1][-1].time.to_str().replace("/", "_").replace(":", "_").replace(" ", "_")
            file_path = f"{source_dir}/{code}_{str_date}.PNG"  # 输出文件的路径

            if not os.path.exists(file_path):
                chan_to_png(chan_snapshot, plot_config, plot_para, file_path=file_path)
            bsp_dict[last_bsp.klu.idx] = {
                "last_bsp": last_bsp,
                "file_path": file_path
            }
            print(last_bsp.klu.time, last_bsp.is_buy)
    feature_meta = {}  # 特征meta
    cur_feature_idx = 0
    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp(0)]
    with open(f"./TMP/{code}_dataset.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'file_path'])  # Write the header
        for bsp_klu_idx, feature_info in bsp_dict.items():
            label = int(bsp_klu_idx in bsp_academy)  # 以买卖点识别是否准确为label
            last_bsp = feature_info["last_bsp"]
            file_path = str(feature_info["file_path"])
            features = []
            for feature_name, value in last_bsp.features.items():
                if feature_name not in feature_meta:
                    feature_meta[feature_name] = cur_feature_idx
                    cur_feature_idx += 1
                features.append((feature_meta[feature_name], value))
            features.sort(key=lambda x: x[0])
            feature_str = " ".join([f"{idx}:{value}" for idx, value in features])
            writer.writerow(
                [label, last_bsp.type[0].value, file_path, feature_str])
    with open(f"./TMP/{code}_feature.meta", "w") as fid:
        # meta保存下来，实盘预测时特征对齐用
        fid.write(json.dumps(feature_meta))


if __name__ == "__main__":
    code = "EURUSD"
    lv_list = [KL_TYPE.K_30M, KL_TYPE.K_5M]
    source_dir = './PNG'

    begin_time = "2010-01-01 00:00:00"
    end_time = "2021-04-01 00:00:00"

    generate_dataset(code, source_dir, lv_list, begin_time, end_time)
