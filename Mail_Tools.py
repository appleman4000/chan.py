# cython: language_level=3
# smtplib 用于邮件的发信动作
import io
import smtplib
# 构建邮件头
from email.header import Header
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
# email 用于构建邮件内容
from email.mime.text import MIMEText
from email.utils import formataddr
from threading import Thread

import matplotlib

from Plot.PlotDriver import CPlotDriver


def asynchronous(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()

    return wrapper


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


@asynchronous
def send_email(to_emails, subject, message, chan):
    try:
        # 发信方的信息：发信邮箱，QQ 邮箱授权码
        from_addr = 'appleman4000@qq.com'
        password = 'unfzwuwkwdwqcace'
        # 收信方邮箱

        # 发信服务器
        smtp_server = 'smtp.qq.com'
        smtpobj = smtplib.SMTP_SSL(smtp_server)
        # 建立连接--qq邮箱服务和端口号（可百度查询）
        smtpobj.connect(smtp_server, 465)
        # 登录--发送者账号和口令
        smtpobj.login(from_addr, password)
        matplotlib.use('Agg')  # 设置 matplotlib 后端为 Agg
        g = CPlotDriver(chan, plot_config, plot_para)
        buf = io.BytesIO()
        g.figure.savefig(buf, format='png')
        buf.seek(0)
        # 发送邮件
        for to_email in to_emails:
            to_addr = to_email
            msg = MIMEMultipart('related')
            # 邮件头信息
            msg['From'] = formataddr((Header("程恩", 'utf-8').encode(), from_addr))
            msg['To'] = formataddr((Header(to_email, 'utf-8').encode(), to_email))
            msg['Subject'] = Header(subject, 'utf-8')  # 邮件主题
            # 构建邮件正文
            html = f"""
                        <html>
                          <body>
                            <p>{message}<br>
                               <img src="cid:image1"><br>
                            </p>
                          </body>
                        </html>
                        """
            msg.attach(MIMEText(html, 'html', 'utf-8'))
            msg_image = MIMEImage(buf.getvalue())
            msg_image.add_header('Content-ID', '<image1>')
            msg.attach(msg_image)
            # 发送邮件
            try:
                smtpobj.sendmail(from_addr, to_addr, msg.as_string())
                print(f"邮件成功发送到: {to_email}")
            except Exception as e:
                print(f"发送到 {to_email} 时发生错误: {e}")

    except smtplib.SMTPException as e:
        print(e)
        print("无法发送邮件")
    finally:
        # 关闭服务器
        smtpobj.quit()
