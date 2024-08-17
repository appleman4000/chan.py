# cython: language_level=3
import csv
import os.path
from typing import Dict, TypedDict

import matplotlib
from matplotlib import pyplot as plt

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Plot.PlotDriver import CPlotDriver

config = CChanConfig({
    "trigger_step": True,  # 打开开关！
    "skip_step": 500,
    "divergence_rate": float("inf"),
    "min_zs_cnt": 1,
    "macd_algo": "slope",
    "kl_data_check": False,
    "bi_end_is_peak": True,
    "bsp1_only_multibi_zs": True,
    "max_bs2_rate": 0.999,
    "bs1_peak": True,
    "bs_type": "1,1p,2,2s,3a,3b",
    "bsp2_follow_1": True,
    "bsp3_follow_1": True,
    "bsp3_peak": True,
    "bsp2s_follow_2": True,
    "max_bsp2s_lv": None,
    "strict_bsp3": True,
})
plot_config = {
    "plot_kline": False,
    "plot_kline_combine": False,
    "plot_bi": True,
    "plot_seg": True,
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
        "w": 224 / 100,
        "h": 224 / 100,
        "x_range": 400,
    },
    "seg": {
        # "plot_trendline": True,
        "disp_end": False,
        "end_fontsize": 15,
        "width": 1
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
        "width": 1
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


if __name__ == "__main__":
    """
    本demo主要演示如何记录策略产出的买卖点的特征
    然后将这些特征作为样本，训练一个模型(以XGB为demo)
    用于预测买卖点的准确性

    请注意，demo训练预测都用的是同一份数据，这是不合理的，仅仅是为了演示
    """
    symbols = [
        # Major
        "EURUSD",
        # "GBPUSD",
        # "AUDUSD",
        # "NZDUSD",
        # "USDJPY",
        # "USDCAD",
        # "USDCHF",
        # Crosses
        # "AUDCHF",
        # "AUDJPY",
        # "AUDNZD",
        # "CADCHF",
        # "CADJPY",
        # "CHFJPY",
        # "EURAUD",
        # "EURCAD",
        # "AUDCAD",
        # "EURCHF",
        # "GBPNZD",
        # "GBPCAD",
        # "GBPCHF",
        # "GBPJPY",
        # "USDCNH",
        # "XAUUSD",
        # "XAGUSD",
    ]
    for code in symbols:
        begin_time = "2010-01-01 00:00:00"
        end_time = "2021-01-01 00:00:00"
        data_src = DATA_SRC.FOREX
        # bottom_kl_type = KL_TYPE.K_3M
        top_kl_type = KL_TYPE.K_30M
        lv_list = [top_kl_type]

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
        source_dir = './png'
        os.makedirs(source_dir, exist_ok=True)
        # 跑策略，保存买卖点的特征
        for chan_snapshot in chan.step_load():

            top_lv_chan = chan_snapshot[0]
            top_bsp_list = chan.get_bsp(0)  # 获取高级别买卖点列表
            if not top_bsp_list:
                continue
            top_last_bsp = top_bsp_list[-1]
            top_entry_rule = top_lv_chan[-1].idx == top_last_bsp.klu.klc.idx
            if top_entry_rule and (top_last_bsp.is_buy or not top_last_bsp.is_buy):
                str_date = top_lv_chan[-1][-1].time.to_str().replace("/", "_").replace(":", "_").replace(" ", "_")
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

                        # 移除 x 轴和 y 轴的刻度线
                        ax.tick_params(axis='both', which='both', length=0)

                        # 移除网格线
                        ax.grid(False)
                    g.figure.tight_layout()
                    g.figure.savefig(file_path, format='png')
                    plt.close(g.figure)

                bsp_dict[top_last_bsp.klu.idx] = {
                    "feature": file_path,
                    "is_buy": top_last_bsp.is_buy,
                    "close": top_lv_chan[-1][-1].close
                }
                print(top_last_bsp.klu.time, top_last_bsp.is_buy)
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
                if closes[i + 1 + j] / price - 1 >= 0.003:
                    label = 1
                    break
                if closes[i + 1 + j] / price - 1 <= -0.003:
                    label = 0
                    break
                j += 1
            labels.append(label)

        with open('dataset.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['label', 'filepath'])  # Write the header
            for label, filepath in zip(labels, filepaths):
                writer.writerow([label, filepath])  # Write label and filepath to the CSV
    # 生成libsvm样本特征
    # bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp(0)]
    # feature_meta = {}  # 特征meta
    # cur_feature_idx = 0
    # plot_marker = {}
    # with open('dataset.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['label', 'filepath'])  # Write the header
    #     for bsp_klu_idx, feature_info in bsp_dict.items():
    #         label = int(bsp_klu_idx in bsp_academy)
    #         filepath = feature_info['feature']
    #         writer.writerow([label, filepath])  # Write label and filepath to the CSV
