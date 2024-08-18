# cython: language_level=3
import csv
import os.path
from typing import Dict, TypedDict

import matplotlib
from matplotlib import pyplot as plt

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, BSP_TYPE
from Plot.PlotDriver import CPlotDriver

config = CChanConfig({
    "trigger_step": True,  # 打开开关！
    "skip_step": 500,
    "divergence_rate": 0.9,
    "min_zs_cnt": 1,
    "macd_algo": "slope",
    "kl_data_check": False,
    "bs_type": "1,1p,2,2s,3a,3b",
})
plot_config = {
    "plot_kline": False,
    "plot_kline_combine": True,
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
        "h": 224 / 50,
        "x_range": 90,
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
    feature: CFeatures
    is_buy: bool
    close: float


def generate_dataset(code, kl_type, begin_time, end_time):
    """
    本demo主要演示如何记录策略产出的买卖点的特征
    然后将这些特征作为样本，训练一个模型(以XGB为demo)
    用于预测买卖点的准确性

    请注意，demo训练预测都用的是同一份数据，这是不合理的，仅仅是为了演示
    """

    data_src = DATA_SRC.FOREX
    lv_list = [kl_type]

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
    source_dir = 'png'
    os.makedirs(source_dir, exist_ok=True)
    # 跑策略，保存买卖点的特征
    for chan_snapshot in chan.step_load():

        lv_chan = chan_snapshot[0]
        bsp_list = chan.get_bsp(0)  # 获取高级别买卖点列表
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]
        if BSP_TYPE.T2 not in last_bsp.type and BSP_TYPE.T2S not in last_bsp.type:  # 假如只做2类买卖点
            continue
        cur_lv_chan = chan_snapshot[0]
        if last_bsp.klu.klc.idx != cur_lv_chan[-1].idx:
            continue
        str_date = lv_chan[-1][-1].time.to_str().replace("/", "_").replace(":", "_").replace(" ", "_")
        file_path = f"{source_dir}/{code}_{str_date}.png"  # 输出文件的路径

        if not os.path.exists(file_path):
            matplotlib.use('Agg')
            g = CPlotDriver(chan, plot_config, plot_para)
            # 移除标题
            for ax in g.figure.axes:
                ax.set_title("", loc="left")
                # 移除 x 轴和 y 轴标签
                ax.set_xlabel('')
                ax.set_ylabel('')

                # 移除 x 轴和 y 轴的刻度标签
                ax.set_xticks([])
                ax.set_yticks([])

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

            g.figure.tight_layout()
            g.figure.savefig(file_path, format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close(g.figure)

        bsp_dict[last_bsp.klu.idx] = {
            "feature": file_path,
            "is_buy": last_bsp.is_buy,
            "close": lv_chan[-1][-1].close
        }
        print(last_bsp.klu.time, last_bsp.is_buy)
    closes = []
    filepaths = []
    is_buys = []
    labels = []
    for bsp_klu_idx, feature_info in bsp_dict.items():
        closes.append(feature_info['close'])
        filepaths.append(feature_info['feature'])
        is_buys.append(feature_info['is_buy'])
    for i, price in enumerate(closes):
        label = 0
        j = 0
        while True and i + 1 + j < len(closes):
            if closes[i + 1 + j] / price - 1 >= 0.004:
                label = 1
                break
            if closes[i + 1 + j] / price - 1 <= -0.004:
                label = 0
                break
            j += 1
        labels.append(label)

    with open(f"./TMP/{code}_dataset.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'filepath'])  # Write the header
        for label, filepath in zip(labels, filepaths):
            writer.writerow([label, filepath])  # Write label and filepath to the CSV


if __name__ == "__main__":
    code = "EURUSD"
    begin_time = "2010-01-01 00:00:00"
    end_time = "2021-01-01 00:00:00"
    kl_type = KL_TYPE.K_30M
    generate_dataset(code, kl_type, begin_time, end_time)
