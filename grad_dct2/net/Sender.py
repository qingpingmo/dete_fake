import socket
import threading
import time
import random

# Configuration for simulating packet loss
PACKET_DROP_CHANCE = 0.3

class DataTransmissionUnit:
    def __init__(self, destination, capacity, deadline, payload):
        self.destination = destination
        self.capacity = capacity
        self.deadline = deadline
        self.payload = payload
        self.current_id = 0
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.connection.settimeout(deadline)
        self.pending = {}
        self.sync = threading.Lock()
        self.process = threading.Thread(target=self.dispatch)
        self.process.start()
        self.complete = False
        self.active = True
        self.delay_timer = None

    def dispatch(self):
        while self.current_id < len(self.payload):
            self.sync.acquire()

            for packet_id in range(self.current_id, min(self.current_id + self.capacity, len(self.payload))):
                if packet_id not in self.pending:
                    if PACKET_DROP_CHANCE > 0 and random.random() < PACKET_DROP_CHANCE:
                        self.pending[packet_id] = self.payload[packet_id]
                        print(f"Packet {packet_id} dropped")
                    else:
                        packet_data = f"{packet_id}:{self.payload[packet_id]}".encode()
                        self.connection.sendto(packet_data, self.destination)
                        self.pending[packet_id] = self.payload[packet_id]
                        print(f"Dispatched packet {packet_id}: {self.payload[packet_id]}")
                        self.initiate_timer(packet_id)
            
            self.sync.release()

            confirmation_received = False
            while not confirmation_received:
                try:
                    ack_msg, _ = self.connection.recvfrom(1024)
                    ack_id = int(ack_msg.decode().split(':')[0])
                    if ack_id in self.pending:
                        del self.pending[ack_id]
                        self.current_id += 1
                    confirmation_received = True
                except socket.timeout:
                    print("Timeout, retrying...")
                    for pkt_id in self.pending:
                        self.retry(pkt_id)

        print("Payload fully sent.")
        self.complete = True

    def initiate_timer(self, pkt_id):
        if self.delay_timer is not None:
            self.delay_timer.cancel()
        self.delay_timer = threading.Timer(self.deadline, self.retry, [pkt_id])
        self.delay_timer.start()

    def retry(self, pkt_id):
        if pkt_id in self.pending and self.active:
            print(f"Retrying packet {pkt_id}: {self.payload[pkt_id]}")
            retry_data = f"{pkt_id}:{self.payload[pkt_id]}".encode()
            self.connection.sendto(retry_data, self.destination)
            self.initiate_timer(pkt_id)

    def shutdown(self):
        while not self.complete:
            print("Finalizing...")
            time.sleep(1)
        self.active = False
        if self.delay_timer is not None:
            self.delay_timer.cancel()
        self.process.join()
        self.connection.close()

if __name__ == "__main__":
    print("sunyue 21281050 Message Transmission System")
    target = ('localhost', 1223)
    max_load = 4
    delay = 1
    
    message = input("Input your message here: ")
    byte_message = message.encode('utf-8')
    
    trans_unit = DataTransmissionUnit(target, max_load, delay, byte_message)
    trans_unit.shutdown()
