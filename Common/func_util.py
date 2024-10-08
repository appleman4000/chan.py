# cython: language_level=3
# encoding:utf-8
from .CEnum import BI_DIR, KL_TYPE


def kltype_lt_day(_type):
    return _type in [KL_TYPE.K_1M, KL_TYPE.K_5M, KL_TYPE.K_15M, KL_TYPE.K_30M, KL_TYPE.K_1H, KL_TYPE.K_4H]


def kltype_lte_day(_type):
    return _type in [KL_TYPE.K_1M, KL_TYPE.K_5M, KL_TYPE.K_15M, KL_TYPE.K_30M, KL_TYPE.K_1H, KL_TYPE.K_4H,
                     KL_TYPE.K_DAY]


def check_kltype_order(type_list: list):
    _dict = {
        KL_TYPE.K_1M: 1,
        KL_TYPE.K_2M: 2,
        KL_TYPE.K_3M: 3,
        KL_TYPE.K_4M: 4,
        KL_TYPE.K_5M: 5,
        KL_TYPE.K_6M: 6,
        KL_TYPE.K_10M: 7,
        KL_TYPE.K_12M: 8,
        KL_TYPE.K_15M: 9,
        KL_TYPE.K_20M: 10,
        KL_TYPE.K_30M: 11,
        KL_TYPE.K_1H: 12,
        KL_TYPE.K_2H: 13,
        KL_TYPE.K_3H: 14,
        KL_TYPE.K_4H: 15,
        KL_TYPE.K_6H: 16,
        KL_TYPE.K_8H: 17,
        KL_TYPE.K_12H: 18,
        KL_TYPE.K_DAY: 19,
        KL_TYPE.K_WEEK: 20,
        KL_TYPE.K_MON: 21,
        KL_TYPE.K_QUARTER: 22,
        KL_TYPE.K_YEAR: 23
    }
    last_lv = float("inf")
    for kl_type in type_list:
        cur_lv = _dict[kl_type]
        assert cur_lv < last_lv, "lv_list的顺序必须从大级别到小级别"
        last_lv = cur_lv


def revert_bi_dir(dir):
    return BI_DIR.DOWN if dir == BI_DIR.UP else BI_DIR.UP


def has_overlap(l1, h1, l2, h2, equal=False):
    return h2 >= l1 and h1 >= l2 if equal else h2 > l1 and h1 > l2


def str2float(s):
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_inf(v):
    if type(v) == float:
        if v == float("inf"):
            v = 'float("inf")'
        if v == float("-inf"):
            v = 'float("-inf")'
    return v
