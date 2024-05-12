import socket
import threading

class GBNClientReceiver:
    def __init__(self, server_address, window_size):
        self.server_address = server_address  # 服务器地址
        self.window_size = window_size        # 滑动窗口大小
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建UDP套接字
        self.sock.bind(server_address)       # 绑定到指定地址
        self.window = {}                      # 窗口中存放的数据包 {序列号: 数据}
        self.index = 0
        self.message = ""  # 新增属性用于存储完整信息

    def receive_data(self):
        # 接收数据的主要逻辑
        while True:
            try:
                # 接收数据包
                data, client_address = self.sock.recvfrom(1024)
                sequence_number, payload = data.split(b':', 1)  # 解析数据包，得到序列号和数据
                sequence_number = int(sequence_number)
                
                # 检查序列号，如果是期望的序列号，则处理并打印当前的完整信息
                if sequence_number == self.index:
                    # 将ASCII码转换为字符并添加到完整信息中
                    character = chr(int(payload.decode()))
                    self.message += character
                    
                    # 更新窗口并发送确认消息
                    self.window[sequence_number] = payload
                    self.sock.sendto(str(sequence_number).encode(), client_address)
                    self.index = self.index + 1 

                print(f"Packet information {sequence_number}: {character} and index:{self.index - 1}")
                print("Complete message so far: ", self.message)  # 打印目前为止的完整信息

            except Exception as e:
                print("Error:", e)

    def close(self):
        # 关闭接收方的方法
        self.sock.close()  # 关闭套接字

# 用法示例
if __name__ == "__main__":
    print("sunyue's receiver")
    server_address = ('localhost', 1223)  # 服务器地址
    window_size = 1                        # 滑动窗口大小
    
    receiver = GBNClientReceiver(server_address, window_size)  # 创建接收方对象
    receiver.receive_data()  # 接收数据
