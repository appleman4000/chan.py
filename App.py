# cython: language_level=3
import datetime
import sys

import matplotlib.pyplot as plt
import streamlit as st

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, KL_TYPE, AUTYPE
from Plot import PlotDriver
from Plot.AnimatePlotDriver import CAnimateDriver
from Plot.PlotDriver import CPlotDriver

sys.setrecursionlimit(10000)
# 时间周期映射
time_frame_mapping = {
    '1分钟': KL_TYPE.K_1M,
    '5分钟': KL_TYPE.K_5M,
    '15分钟': KL_TYPE.K_15M,
    '30分钟': KL_TYPE.K_30M,
    '1小时': KL_TYPE.K_60M,
    '1天': KL_TYPE.K_DAY
}
time_frame_type = ['1分钟', '5分钟', '15分钟', '30分钟', '1小时', '1天']


def run_chanlun(code, begin_time=None, end_time=None, market_type="外汇", time_frames=[], trigger_step=False):
    print(code)
    print(begin_time)
    print(end_time)
    print(market_type)
    print(time_frames)
    print(trigger_step)
    if market_type == "外汇/贵金属/CFD":
        data_src = DATA_SRC.FOREX
    elif market_type == "国内A股":
        data_src = DATA_SRC.AKSHARE_STOCK
    elif market_type == "国内ETF基金":
        data_src = DATA_SRC.AKSHARE_ETF
    else:
        data_src = None

    lv_list = [time_frame_mapping[time_frame] for time_frame in time_frames]
    config = CChanConfig({
        "bi_strict": True,
        "trigger_step": trigger_step,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
        "kl_data_check": False
    })

    plot_config = {
        "plot_kline": True,
        "plot_kline_combine": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_eigen": True,
        "plot_zs": True,
        "plot_macd": True,
        "plot_mean": False,
        "plot_channel": False,
        "plot_bsp": True,
        "plot_extrainfo": True,
        "plot_demark": False,
        "plot_marker": False,
        "plot_rsi": False,
        "plot_kdj": False,
    }

    plot_para = {
        "seg": {
            # "plot_trendline": True,
        },
        "bi": {
            "show_num": True,
            "disp_end": True,
        },
        "figure": {
            "x_range": 200,
        },
        "marker": {
            # "markers": {  # text, position, color
            #     '2023/06/01': ('marker here', 'up', 'red'),
            #     '2023/06/08': ('marker here', 'down')
            # },
        }
    }
    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )

    if not config.trigger_step:
        plot_driver = CPlotDriver(
            chan,
            plot_config=plot_config,
            plot_para=plot_para,
        )
        plot_driver.figure.tight_layout(pad=5)
        placeholder.pyplot(plot_driver.figure)
        plt.close(plot_driver.figure)

    else:
        CAnimateDriver(
            chan,
            plot_config=plot_config,
            plot_para=plot_para,
            placeholder=placeholder
        )


if __name__ == "__main__":
    # WRYH = mpl.font_manager.FontProperties(fname="./simhei.ttf")
    plt.rcParams.update({'font.size': 18})
    plt.rcParams["font.sans-serif"] = ["simhei"]
    plt.rcParams["axes.unicode_minus"] = False
    # 主界面布局设计
    st.set_page_config(layout="wide", page_title="股票/外汇交易分析")

    # 设置自定义样式以减小空白
    st.markdown("""
                <style>
                    .main > div {
                        padding-top: 0px;
                        padding-bottom: 0px;
                    }
                    #MainMenu {visibility: hidden;}
                    header {visibility: hidden;}
                    footer {visibility: hidden;}
                </style>
                """, unsafe_allow_html=True)
    begin_time = None
    end_time = None
    # 右栏布局
    with st.container():
        st.markdown("<h1 style='font-size: 18px;text-align: center;'>缠论蜡烛图</h1>", unsafe_allow_html=True)
        # 绘制实时蜡烛图表
        placeholder = st.empty()
    # 左栏布局
    with st.sidebar:
        st.markdown("<h1 style='font-size: 18px;text-align: center;'>股票/基金/外汇交易缠论分析</h1>",
                    unsafe_allow_html=True)
        mode = st.radio("模式", ["在线分析", "历史复盘"], index=0)
        if mode == "历史复盘":
            cols = st.columns(2)
            with cols[0]:
                # 获取当前日期
                today = datetime.datetime.now().date()
                # 计算一个月前的日期
                one_month_ago = today - datetime.timedelta(weeks=4 * 6)
                begin_time = st.date_input('时间段', value=one_month_ago)
            with cols[1]:
                end_time = st.date_input('结束日期', label_visibility="hidden", value=today)
        market_type = st.radio("市场", ["国内A股", "国内ETF基金", "外汇/贵金属/CFD"], index=0)
        if market_type == "国内A股":
            code = st.text_input("股票代码", value="601818")
        elif market_type == "国内ETF基金":
            code = st.text_input("基金代码", value="513100")
        elif market_type == "外汇/贵金属/CFD":
            code = st.text_input("交易品种", value="USDCNH")
        time_frames = st.multiselect('时间周期', time_frame_type, default=['1天'])
        query = st.button("开始分析", use_container_width=True)
        if query:
            st.balloons()
            with st.spinner('正在计算缠论，请耐心稍候...'):
                PlotDriver.figure = None
                PlotDriver.axes = None
                PlotDriver.axes_origin = None
                if mode == "在线分析":
                    # 获取当前日期
                    end_time = datetime.datetime.now()
                    # 计算200天前的日期
                    begin_time = end_time - datetime.timedelta(days=200)
                    begin_time = begin_time.strftime("%Y-%m-%d %H:%M:%S")
                    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
                time_frames = sorted(time_frames, key=lambda i: time_frame_type[::-1].index(i))
                run_chanlun(code=code, begin_time=begin_time, end_time=end_time, market_type=market_type,
                            time_frames=time_frames, trigger_step=mode == "历史复盘")
