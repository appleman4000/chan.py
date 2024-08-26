# cython: language_level=3
# encoding:utf-8
import typing
from collections.abc import MutableSequence

MAX_LEN = 500


class LimitedList(MutableSequence):
    def __init__(self, maxlen=MAX_LEN, initial_list=None):
        self.maxlen = maxlen
        self._list = list(initial_list) if initial_list is not None else []

    def append(self, item):
        if len(self._list) >= self.maxlen:
            self._list.pop(0)
        self._list.append(item)

    def extend(self, iterable):
        for item in iterable:
            self.append(item)

    def insert(self, index, item):
        if len(self._list) >= self.maxlen:
            self._list.pop(0)
        self._list.insert(index, item)

    def pop(self, index=-1):
        return self._list.pop(index)

    def remove(self, item):
        self._list.remove(item)

    def __getitem__(self, index):
        return self._list[index]

    def __setitem__(self, index, value):
        self._list[index] = value

    def __delitem__(self, index):
        del self._list[index]

    def __len__(self):
        return len(self._list)

    def __repr__(self):
        return repr(self._list)

    def __iter__(self):
        return iter(self._list)

    def clear(self):
        self._list.clear()

    def count(self, item):
        return self._list.count(item)

    def index(self, item, start=0, end=None):
        return self._list.index(item, start, end)

    def copy(self):
        return LimitedList(self.maxlen, self._list.copy())


# Hot replace typing.List with LimitedList
typing.List = LimitedList
