import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

# SMTP服务器配置
SMTP_SERVER = "smtp.qq.com"  # SMTP服务器地址
SMTP_PORT = 465              # 使用SSL加密时的SMTP服务器端口
SMTP_USERNAME = "3321734090@qq.com"  # SMTP服务器登录用户名
SMTP_PASSWORD = "iwikmfyrlksychhi"   # SMTP服务器登录密码（QQ邮箱使用授权码）

def send_email(subject, body, to_addr, from_addr=SMTP_USERNAME):
    # 创建邮件对象
    msg = MIMEMultipart()
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = Header(subject, 'utf-8')
    
    # 添加邮件正文
    body_content = MIMEText(body, 'plain', 'utf-8')
    msg.attach(body_content)
    
    try:
        # 使用SSL加密方式连接到SMTP服务器
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)  # 登录SMTP服务器
            server.sendmail(from_addr, to_addr, msg.as_string())  # 发送邮件
            print("邮件发送成功！")
    except Exception as e:
        print(f"邮件发送失败：{e}")

# 测试邮件发送功能
if __name__ == "__main__":
    to_address = "21251134@bjtu.edu.cn"  # 收件人地址
    send_email("测试邮件", "这是一封测试邮件。", to_address)
