import struct

x = 12345.0

# 单精度 (32-bit)
single_hex = f"0x{struct.unpack('>I', struct.pack('>f', x))[0]:08X}"
print("single precision hex:", single_hex)   # 期望输出: 0x4640E400

# 双精度 (64-bit)
double_hex = f"0x{struct.unpack('>Q', struct.pack('>d', x))[0]:016X}"
print("double precision hex:", double_hex)   # 期望输出: 0x40C81C8000000000