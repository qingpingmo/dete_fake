import imaplib
import email
from email.header import decode_header

# IMAP服务器配置
IMAP_SERVER = "imap.qq.com"
IMAP_PORT = 993
IMAP_USERNAME = "3321734090@qq.com"
IMAP_PASSWORD = "iwikmfyrlksychhi"

def fetch_emails():
    # 连接到IMAP服务器
    mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
    # 登录
    mail.login(IMAP_USERNAME, IMAP_PASSWORD)
    # 选择"收件箱"文件夹
    mail.select("inbox")
    
    # 搜索所有邮件
    result, data = mail.search(None, "ALL")
    mail_ids = data[0]
    id_list = mail_ids.split()
    
    for i in id_list:
        result, data = mail.fetch(i, '(RFC822)')
        
        for response_part in data:
            if isinstance(response_part, tuple):
                # 解析邮件内容
                msg = email.message_from_bytes(response_part[1])
                subject, encoding = decode_header(msg["subject"])[0]
                if isinstance(subject, bytes):
                    # 如果subject是字节类型的，使用其指定的编码进行解码
                    subject = subject.decode(encoding) if encoding else subject.decode('utf-8', errors='ignore')
                email_from = msg["from"]
                print(f"From: {email_from}\nSubject: {subject}\n")

if __name__ == "__main__":
    fetch_emails()
