import matplotlib.pyplot as plt
from transmit import *


# OFDM system setup:
test = OFDM
N, prefix_no, fs, repeat, gap_second = 2048, 256, 44100, 10, 0

chirp_high= 16000

min_bin = 40

max_bin = 575

inputLenIndicator_len=24
rate='1/2'
r=0.5
z=27
len_protection='input_repeat_then_LDPC'
repeat_times=3

# Pass the parameters
test.__init__(test, N, prefix_no, fs, repeat, gap_second, chirp_high, min_bin, max_bin, inputLenIndicator_len=inputLenIndicator_len, rate=rate,r=r,z=z,len_protection=len_protection,repeat_times=repeat_times)

# File for transfer
# testfile = "test_files/test.jpg"
testfile = 'test_files/test_small.png'

# print the default seed for checking
print("The default seed is set to:", test.seed)

# Generate the random symbols with the specific seed
random_symbols = test.generate_random_symbols_seeds(test)
known_frame, known_frames = test.generate_known_OFDM(test, random_symbols) 

chirp, inv_chirp = test.define_chirp(test)


data_frames, underfill, carrier_indices, pilot_values, bits_tran = test.data_add_random_pilots(test, testfile,LDPC=True)
data_frames_len = len(data_frames)
# Combine to get the overall tx signals
tx_signal = test.tx_waveform_data_pilot(test, known_frames, chirp, data_frames)
# plt.plot(tx_signal)
sf.write('transmit.wav', tx_signal, test.fs)


#Record and load the record file
rx_signal = test.real_channel_response_file(test, 'transmit.wav')#"sound_files/Rec19.m4a")
# plt.plot(rx_signal)

convolution, peak_index1, peak_index2 = test.matched_filter_double(test, rx_signal, inv_chirp)
# plt.plot(rx_signal, label='channel op')
# plt.plot(convolution/100, label='convolution')
# plt.show()

split_frame, avg_frame, start_refined = test.process_transmission_pilot(test, rx_signal, peak_index1, offset=20)
channel_freq_response, channel_imp_response = test.estimate_channel_response_pilot(test, avg_frame, known_frame)
# plt.plot(channel_imp_response)


best_offset, best_score, bits_rec, best_imp_response = test.fine_tuning_pilot(test, rx_signal, peak_index1, known_frame, carrier_indices, pilot_values, underfill, data_frames_len=data_frames_len, find_range=30, offset=0, filename="test1.png",LDPC=True)




