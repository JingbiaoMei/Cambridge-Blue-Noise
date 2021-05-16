import numpy as np
import binascii
from bitarray import bitarray

N = 1024
prefix = 32

channel_IR = np.genfromtxt('channel.csv',delimiter=',')
rx_signal = np.genfromtxt('file1.csv',delimiter=',')

num_blocks = len(rx_signal)/(N+prefix) # 950 for file 1
ofdm_frames = np.split(rx_signal, num_blocks) # lenth 1056

channel_fft = np.fft.fft(channel_IR,n=N)

data = ''
for i in range(len(ofdm_frames)):
    frame_prefix = ofdm_frames[i][prefix:] # remove cp
    frame_dft = np.fft.fft(frame_prefix, n=N) # 
    for i in range(1, 512): # only useful info bits 1-511
        element = frame_dft[i] / channel_fft[i]
        if np.real(element) >= 0:
            if np.imag(element) >= 0:
                data += '00'
            else:
                data += '10'
        else:
            if np.imag(element) >= 0:
                data += '01'
            else:
                data += '11'

data_bytes = bitarray(data)

#print(len(data_bytes.tobytes()))

with open("decoded.tiff", 'wb') as f:
    f.write(data_bytes.tobytes()[29:])











