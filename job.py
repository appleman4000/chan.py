# cython: language_level=3
# encoding:utf-8
import time
import subprocess
from datetime import datetime, timedelta


def get_next_open_and_close_times():
    # 此函数应返回下一个开市和休市时间
    # 这里使用占位符时间作为示例
    now = datetime.now()
    next_open = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0)
    next_close = (now + timedelta(days=1)).replace(hour=17, minute=0, second=0)
    return next_open, next_close


def main():
    while True:
        next_open, next_close = get_next_open_and_close_times()
        now = datetime.now()

        # 计算距离开市前10分钟的时间
        open_wait_time = (next_open - timedelta(minutes=10) - now).total_seconds()
        # 计算距离休市10分钟的时间
        close_wait_time = (next_close + timedelta(minutes=10) - now).total_seconds()

        if open_wait_time > 0:
            print(f"等待开市前10分钟: {open_wait_time} 秒")
            time.sleep(open_wait_time)
            print("开市前10分钟，启动主程序")
            subprocess.Popen(['python', 'monitor.py'])

        if close_wait_time > 0:
            print(f"等待休市10分钟后关闭程序: {close_wait_time} 秒")
            time.sleep(close_wait_time)
            print("休市10分钟，关闭主程序")
            # 找到并杀死特定的 Python 进程（main_program.py）
            tasklist = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], capture_output=True,
                                      text=True).stdout
            for line in tasklist.splitlines()[1:]:
                if 'monitor.py' in line:
                    pid = line.split(',')[1].strip('"')
                    subprocess.run(['taskkill', '/PID', pid, '/F'])

        # 每天循环检查
        time.sleep(60)


if __name__ == "__main__":
    main()
