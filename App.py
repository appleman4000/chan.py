# cython: language_level=3
# encoding:utf-8
import datetime
import sys

import matplotlib.pyplot as plt
import streamlit as st

from Chan import CChan
from Common.CEnum import DATA_SRC, AUTYPE
from CommonTools import period_name, period_seconds
from GenerateDataset import config, plot_config, plot_para
from Plot.AnimatePlotDriver import CAnimateDriver
from Plot.PlotDriver import CPlotDriver

sys.setrecursionlimit(10000)


def run_chanlun(code, begin_time=None, end_time=None, market_type="外汇", lv_list=[], trigger_step=False):
    print(code)
    print(begin_time)
    print(end_time)
    print(market_type)
    print(lv_list)
    print(trigger_step)
    if market_type == "外汇/贵金属/CFD":
        data_src = DATA_SRC.FOREX
    elif market_type == "国内A股":
        data_src = DATA_SRC.AKSHARE_STOCK
    elif market_type == "国内ETF基金":
        data_src = DATA_SRC.AKSHARE_ETF
    else:
        data_src = None
    plot_config["plot_kline"] = True
    plot_config["plot_kline_combine"] = False
    plot_para["figure"] = {
        "w": 224,
        "h": 224,
        "x_range": 200,
    }
    plot_para["seg"] = {
        # "plot_trendline": True,
        "disp_end": True,
        "end_fontsize": 15,
        "width": 0.5
    }
    plot_para["bi"] = {
        "show_num": False,
        "disp_end": True,
        "end_fontsize": 15,
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
        mode = st.radio("模式", ["在线分析"], index=0)
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
            code = st.text_input("交易品种", value="EURUSD")
        lv_names = st.multiselect('时间周期', period_name.values(), default=['1天'])
        lv_list = [key for key in period_name.keys() if period_name[key] in lv_names]
        lv_list = sorted(lv_list, key=lambda i: lv_list[::-1].index(i))
        query = st.button("开始分析", use_container_width=True)
        if query:
            st.balloons()
            with st.spinner('正在计算缠论，请耐心稍候...'):
                if mode == "在线分析":
                    # 获取当前日期
                    end_time = datetime.datetime.now()
                    # 计算200天前的日期
                    seconds = period_seconds[lv_list[0]]
                    begin_time = end_time - datetime.timedelta(seconds=seconds * 2000)
                    begin_time = begin_time.strftime("%Y-%m-%d %H:%M:%S")
                    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")

                run_chanlun(code=code, begin_time=begin_time, end_time=end_time, market_type=market_type,
                            lv_list=lv_list, trigger_step=mode == "历史复盘")
