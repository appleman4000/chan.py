# cython: language_level=3
# encoding:utf-8
from matplotlib import pyplot as plt

from Chan import CChan
from .PlotDriver import CPlotDriver


class CAnimateDriver:
    def __init__(self, chan: CChan, plot_config=None, plot_para=None, placeholder=None):

        # plt.ion()
        if plot_config is None:
            plot_config = {}
        if plot_para is None:
            plot_para = {}

        for _ in chan.step_load():
            g = CPlotDriver(chan, plot_config, plot_para)
            # plt.pause(0.01)  # 暂停0.1秒，控制更新频率
            placeholder.pyplot(g.figure)
            plt.close(g.figure)
