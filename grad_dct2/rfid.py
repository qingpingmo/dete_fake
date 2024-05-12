# by 王俊童 21251134

def hex_to_binary(hex_data):
    
    binary_data = bin(int(hex_data, 16))[2:]
    
    padding_length = 4 - (len(binary_data) % 4) if len(binary_data) % 4 != 0 else 0
    return '0' * padding_length + binary_data

def encode_data(binary_data, direction):
    
    if direction == "PCD->PICC":
        # PCD->PICC 使用改进的弥勒码
        encoding_map = {'1': '1101', '0': '0111'}  
        sof = '0111'  # Z信号
        eof = '1111'  # Y信号
    elif direction == "PICC->PCD":
        # PICC->PCD 使用曼彻斯特编码
        encoding_map = {'1': '1100', '0': '0011'}  
        sof = '1100'  # D信号
        eof = '0000'  # F信号
    else:
        raise ValueError("Invalid direction. Use 'PCD->PICC' or 'PICC->PCD'.")

    
    encoded_data = ''.join([encoding_map[bit] for bit in binary_data])

    
    encoded_frame = sof + ',' + encoded_data + ',' + eof
    return encoded_frame

def generate_waveform(encoded_frame):
    
    waveform = ""
    for group in encoded_frame.split(','):
        for bit in group:
            if bit == '1':
                waveform += "===="  # 高电平表示
            else:
                waveform += "----"  # 低电平表示
        waveform += ","  
    return waveform[:-1]  

def encode_hex_to_waveform(hex_data, direction):
    
    binary_data = hex_to_binary(hex_data)
    encoded_frame = encode_data(binary_data, direction)
    waveform = generate_waveform(encoded_frame)
    return waveform


pcd2picc_hex_data = "1A3F"
picc2pcd_hex_data="2048"
direction_pcd_to_picc = "PCD->PICC"
direction_picc_to_pcd = "PICC->PCD"

waveform_pcd_to_picc = encode_hex_to_waveform(pcd2picc_hex_data, direction_pcd_to_picc)
waveform_picc_to_pcd = encode_hex_to_waveform(picc2pcd_hex_data, direction_picc_to_pcd)

print("PCD->PICC 方向的方波波形图：\n", waveform_pcd_to_picc)
print("\nPICC->PCD 方向的方波波形图：\n", waveform_picc_to_pcd)
