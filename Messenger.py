# cython: language_level=3
# encoding:utf-8
# smtplib 用于邮件的发信动作
import io
import json
import os
import smtplib
# 构建邮件头
from email.header import Header
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
# email 用于构建邮件内容
from email.mime.text import MIMEText
from threading import Thread

import lark_oapi as lark
import matplotlib
import matplotlib.pyplot as plt
import requests
from PIL import Image
from lark_oapi.api.im.v1 import CreateImageRequest, CreateImageRequestBody, CreateImageResponse
from pywchat import Sender

from GenerateDataset import plot_config, plot_para
from Plot.PlotDriver import CPlotDriver


def asynchronous(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()

    return wrapper


def send_feishu_message(app_id, app_secret, webhook_url, subject, message, image_file):
    def send_card_message(webhook_url, subject, message, image_key):
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "msg_type": "interactive",
            "card": {
                "elements": [
                    {
                        "tag": "img",
                        "img_key": image_key,
                        "alt": {
                            "tag": "plain_text",
                            "content": "Image description"
                        }
                    }
                ],
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"{subject}\n{message}"
                    }
                }
            }
        }
        response = requests.post(webhook_url, headers=headers, data=json.dumps(payload))
        return response.json()

    def upload_image(client, image_file):
        # 构造请求对象
        # file = open(image_path, "rb")
        request: CreateImageRequest = CreateImageRequest.builder() \
            .request_body(CreateImageRequestBody.builder()
                          .image_type("message")
                          .image(image_file)
                          .build()) \
            .build()

        # 发起请求
        response: CreateImageResponse = client.im.v1.image.create(request)
        return response.data.image_key

    # 创建client
    client = lark.Client.builder() \
        .app_id(app_id) \
        .app_secret(app_secret) \
        .log_level(lark.LogLevel.ERROR) \
        .build()

    image_key = upload_image(client, image_file)
    send_card_message(webhook_url, subject=subject, message=message, image_key=image_key)


def send_wchat_message(subject, message, image_file):
    corpid = "ww3998521650a4cb79"
    corpsecret = "oUPLyn6uDRnFVJ1RqeMSzISrrN38LMtWWa2x6cwI1Gs"
    agentid = "1000002"
    app = Sender(corpid, corpsecret, agentid)
    bytes_data = image_file.getvalue()
    # 将字节数据写入到文件中
    file_path = './upload_file.png'  # 输出文件的路径
    with open(file_path, 'wb') as f:
        f.write(bytes_data)
    image_url = app.upload_image(file_path, enable=False)
    print(image_url)
    os.remove(file_path)
    app.send_graphic(subject, message, image_url, todept="BornToFly Limited")


def combine_images_vertically(image_bytes_list):
    # 打开所有图像
    images = [Image.open(io.BytesIO(image_bytes)) for image_bytes in image_bytes_list]

    # 获取每张图像的尺寸
    width, height = images[0].size

    # 计算合成图像的总高度
    total_height = height * len(images)

    # 创建一个新的图像，宽度和每张图像一样，高度为所有图像高度之和
    combined_image = Image.new('RGBA', (width, total_height))

    # 将每张图像粘贴到新图像上
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += height

    # 将合成后的图像保存到 io.BytesIO 对象中
    buffer = io.BytesIO()
    combined_image.save(buffer, format='PNG')
    buffer.seek(0)  # 将指针移动到流的开头
    return buffer


def send_mail(to_emails, subject, message, chans):
    try:
        # 发信方的信息：发信邮箱，QQ 邮箱授权码
        from_addr = 'appleman4000@qq.com'
        password = 'unfzwuwkwdwqcace'
        # 发信服务器
        smtp_server = 'smtp.qq.com'
        smtpobj = smtplib.SMTP_SSL(smtp_server)
        # 建立连接--qq邮箱服务和端口号（可百度查询）
        smtpobj.connect(smtp_server, 465)
        # 登录--发送者账号和口令
        smtpobj.login(from_addr, password)
        matplotlib.use('Agg')  # 设置 matplotlib 后端为 Agg
        image_bytes_list = []
        for chan in chans:
            g = CPlotDriver(chan, plot_config, plot_para)
            buf = io.BytesIO()

            g.figure.savefig(buf, format='png')
            plt.close(g.figure)
            buf.seek(0)
            image_bytes_list.append(buf.getvalue())
        buf = combine_images_vertically(image_bytes_list)
        # 发送邮件
        msg = MIMEMultipart('related')
        # 邮件头信息
        msg['From'] = from_addr
        msg['To'] = ', '.join(to_emails)
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
            smtpobj.sendmail(from_addr, to_emails, msg.as_string())
            print(f"邮件成功发送到: {', '.join(to_emails)}")
        except Exception as e:
            print(f"发送到 {', '.join(to_emails)} 时发生错误: {e}")

    except smtplib.SMTPException as e:
        print(e)
        print("无法发送邮件")
    finally:
        # 关闭服务器
        smtpobj.quit()


# @asynchronous
def send_message(app_id, app_secret, webhook_url, subject, message, chans):
    plot_config["plot_kline"] = True
    plot_config["plot_kline_combine"] = False
    plot_para.update({"figure": {
        "w": 224 / 50,
        "h": 224 / 50,
        "x_range": 200,
    }})
    matplotlib.use('Agg')  # 设置 matplotlib 后端为 Agg
    image_bytes_list = []
    for chan in chans:
        g = CPlotDriver(chan, plot_config, plot_para)
        for ax in g.figure.axes:
            ax.set_title("", loc="left")
            # 移除 x 轴和 y 轴标签
            ax.set_xlabel('')
            ax.set_ylabel('')

            # 移除 x 轴和 y 轴的刻度标签
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # 移除 x 轴和 y 轴的刻度线
            ax.tick_params(axis='both', which='both', length=0)

            # 移除网格线
            ax.grid(False)
        g.figure.tight_layout()
        buf = io.BytesIO()
        g.figure.savefig(buf, format='png')
        plt.close(g.figure)
        buf.seek(0)
        image_bytes_list.append(buf.getvalue())
    if len(chans) > 0:
        buf = combine_images_vertically(image_bytes_list)
    # 发送邮件
    try:
        send_feishu_message(app_id, app_secret, webhook_url, subject, message, buf)
        print("飞书成功发送")
    except Exception as e:
        print(f"发送到飞书时发生错误: {e}")
