# smtplib 用于邮件的发信动作
import smtplib
# 构建邮件头
from email.header import Header
# email 用于构建邮件内容
from email.mime.text import MIMEText
from threading import Thread


def asynchronous(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()

    return wrapper


@asynchronous
def send_email(to_emails, message):
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
        # 发送邮件
        for to_email in to_emails:
            to_addr = to_email
            # 邮箱正文内容，第一个参数为内容，第二个参数为格式(plain 为纯文本)，第三个参数为编码
            msg = MIMEText('使用python发送邮件', 'plain', 'utf-8')
            # 邮件头信息
            msg['From'] = Header('appleman4000@qq.com')  # 发送者
            msg['To'] = Header(to_email)  # 接收者
            subject = '外汇-' + message
            msg['Subject'] = Header(subject, 'utf-8')  # 邮件主题
            # 发送邮件
            try:
                smtpobj.sendmail(from_addr, to_addr, msg.as_string())
                print(f"邮件成功发送到: {to_email}")
            except Exception as e:
                print(f"发送到 {to_email} 时发生错误: {e}")

        print("邮件发送成功")
    except smtplib.SMTPException as e:
        print(e)
        print("无法发送邮件")
    finally:
        # 关闭服务器
        smtpobj.quit()
