import os.path
import shutil
from typing import Dict, TypedDict

from matplotlib import pyplot as plt

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE, BSP_TYPE
from Common.CTime import CTime
from Plot.PlotDriver import CPlotDriver

config = CChanConfig({
    "trigger_step": True,  # 打开开关！
    "bi_strict": True,
    "gap_as_kl": True,
    "min_zs_cnt": 1,
    "divergence_rate": float("inf"),
    "max_bs2_rate": 0.618,
    "macd_algo": "diff",
})
plot_config = {
    "plot_kline": True,
    "plot_kline_combine": False,
    "plot_bi": True,
    "plot_seg": True,
    "plot_eigen": False,
    "plot_zs": True,
    "plot_macd": True,
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
    "seg": {
        # "plot_trendline": True,
        "disp_end": False,
        "end_fontsize": 15
    },
    "bi": {
        "show_num": False,
        "disp_end": False,
        "end_fontsize": 15
    },
    "zs": {
        "fontsize": 15
    },
    "bsp": {
        "fontsize": 20
    },
    "segseg": {
        "end_fontsize": 15
    },
    "seg_bsp": {
        "fontsize": 20
    },
    "figure": {
        "x_range": 400,
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
    open_time: CTime


def plot(chan, plot_marker):
    plot_para["marker"] = dict(markers=plot_marker)
    plot_driver = CPlotDriver(
        chan,
        plot_config=plot_config,
        plot_para=plot_para,
    )
    plot_driver.save2img("label.png")


def stragety_feature(last_klu):
    return {
        "open_klu_rate": (last_klu.close - last_klu.open) / last_klu.open,
    }


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
        "GBPUSD",
        "AUDUSD",
        "NZDUSD",
        "USDJPY",
        "USDCAD",
        "USDCHF",
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
        lv_list = [KL_TYPE.K_1H]

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
        source_dir = './PNG/TMP'
        target_dir = './PNG/TRAIN'
        os.makedirs(source_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
        # 跑策略，保存买卖点的特征
        for step, chan_snapshot in enumerate(chan.step_load()):
            if step < 500:
                continue
            last_klu = chan_snapshot[0][-1][-1]
            bsp_list = chan_snapshot.get_bsp()
            if not bsp_list:
                continue
            last_bsp = bsp_list[-1]

            cur_lv_chan = chan_snapshot[0]
            if last_bsp.klu.idx not in bsp_dict and last_bsp.klu.time == cur_lv_chan[-1][-1].time and \
                    (BSP_TYPE.T1 in last_bsp.type or BSP_TYPE.T1P in last_bsp.type):
                str_date = last_klu.time.to_str().replace("/", "_").replace(":", "_").replace(" ", "_")
                file_path = f"{source_dir}/{code}_{str_date}.png"  # 输出文件的路径
                if not os.path.exists(file_path):
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

                bsp_dict[last_bsp.klu.idx] = {
                    "feature": file_path,
                    "is_buy": last_bsp.is_buy,
                    "open_time": last_klu.time,
                }
                print(last_bsp.klu.time, last_bsp.is_buy)

        # 生成libsvm样本特征
        bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
        feature_meta = {}  # 特征meta
        cur_feature_idx = 0
        plot_marker = {}
        os.makedirs(f"{target_dir}/{0}", exist_ok=True)
        os.makedirs(f"{target_dir}/{1}", exist_ok=True)
        for bsp_klu_idx, feature_info in bsp_dict.items():
            label = int(bsp_klu_idx in bsp_academy)  # 以买卖点识别是否准确为label
            filepath = feature_info['feature']
            shutil.copy(filepath, f"{target_dir}/{label}")
