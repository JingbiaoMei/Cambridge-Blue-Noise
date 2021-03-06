import matplotlib.pyplot as plt
from transmit import *
from LDPC import *


# OFDM system setup:
test = OFDM
N, prefix_no, fs, repeat, gap_second = 2048, 256, 44100, 10, 0

LDPC_=True

chirp_high= 16000

min_bin = 40

max_bin = 575

inputLenIndicator_len=32
rate='1/2'
r=0.5
z=81
len_protection='input_repeat_then_LDPC'
repeat_times=5

# Pass the parameters
test.__init__(test, N, prefix_no, fs, repeat, gap_second, chirp_high, min_bin, max_bin, inputLenIndicator_len=inputLenIndicator_len, rate=rate,r=r,z=z,len_protection=len_protection,repeat_times=repeat_times)

# File for transfer
testfile = 'test_files/shakespeare_small.txt'
out_filename=None

# print the default seed for checking
print("The default seed is set to:", test.seed)

# Generate the random symbols with the specific seed
random_symbols = test.generate_random_symbols_seeds(test)
known_frame, known_frames = test.generate_known_OFDM(test, random_symbols) 

chirp, inv_chirp = test.define_chirp(test)


data_frames, underfill, carrier_indices, pilot_values, bits_tran = test.data_add_random_pilots(test, testfile,LDPC=LDPC_)
data_frames_len = len(data_frames)
# Combine to get the overall tx signals
tx_signal = test.tx_waveform_data_pilot(test, known_frames, chirp, data_frames)
# plt.plot(tx_signal)
# sf.write('transmit.wav', tx_signal, test.fs)

# simulated channel:
print("about to do convolution")
rx_signal = test.ideal_channel_response(test,tx_signal)
# rx_signal = awgn(tx_signal, 0.0000000001)
# rx_signal = tx_signal
# print(rx_signal[100:110])
# rx_signal[100:110]=[-0.03]*10


#Record and load the record file
# rx_signal = test.real_channel_response_file(test, 'transmit.wav')#"sound_files/Rec19.m4a")
# plt.plot(rx_signal)
print("about to do test.matched_filter_double")
convolution, peak_index1, peak_index2 = test.matched_filter_double(test, rx_signal, inv_chirp)
# plt.plot(rx_signal, label='channel op')
# plt.plot(convolution/100, label='convolution')
# plt.show()


print("about to do test.process_transmission_pilot")
split_frame, avg_frame, start_refined = test.process_transmission_pilot(test, rx_signal, peak_index1, offset=20)

print("about to do test.estimate_channel_response_pilot")
channel_freq_response, channel_imp_response = test.estimate_channel_response_pilot(test, avg_frame, known_frame)
# plt.plot(channel_imp_response)


print("about to do test.fine_tuning_pilot")
best_offset, best_score, bits_rec, best_imp_response = test.fine_tuning_pilot(test, rx_signal, peak_index1, known_frame, carrier_indices, pilot_values, underfill, data_frames_len=data_frames_len, find_range=30, offset=0, filename=out_filename,LDPC=LDPC_)

# best_offset, best_score, bits_rec, best_imp_response = test.fine_tuning_pilot(test, rx_signal, peak_index1, known_frame, carrier_indices, pilot_values, underfill, data_frames_len=data_frames_len, find_range=30, offset=0, filename="test1.png",LDPC=False)

# bits_rec= LDPC_decode(bits_rec,N=10,rate=rate,r=r,z=z,len_protection=len_protection,repeat_times=repeat_times)

print("Best Offset:", best_offset)
print("Best Impulse Score:", best_score)
if type(bits_tran)!=type(bits_rec):
    print("type(bits_tran): ",type(bits_tran))
    print("type(bits_rec): ",type(bits_rec))


# note: bits_tran == file_to_bitstr(testfile)
print("BER:", "%.3f" % (error_rate(bits_tran, bits_rec[0:len(bits_tran)]) * 100), "%")

a=1

