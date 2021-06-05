import numpy as np
from QAM_EncoderDecoder import *
from LDPC import *
from scipy.io.wavfile import write, read
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from IPython.display import Audio
from scipy import interpolate, signal
from ldpc_jossy.py import ldpc
from util import *
import time

fs = 44100
N = 2048 # DFT length
prefix_no = 256 # Cyclic prefix length
min_bin=50
max_bin=700
gap_second = 1
gap = gap_second * fs

def random_symbols_from_binary(length, seed):
    """Returns a 'length'-long sequence of random constellation symbols"""

    rng = np.random.default_rng(seed)
    random_binary_sequence = rng.integers(low=0, high=2, size=2*length) # x2 as converting from binary to QPSK
    random_binary_values = np.split(random_binary_sequence, length)

    mapping = {
        '0 0':  1+1j,
        '0 1': -1+1j,
        '1 1': -1-1j,
        '1 0':  1-1j
    }

    random_symbols = np.array([mapping[str(r)[1:-1]] for r in random_binary_values])
    
    return random_symbols



def define_chirp(sec=1):
    """returns standard log chirp waveform and its time-reverse"""
    
    k = 50
    w1 = 100
    w2 = 10000
    
    t = np.linspace(0, sec, int(fs*sec))
    
    ch = np.sin(2*np.pi*sec*w1*(((w2/w1)**(t/sec)-1)/(np.log(w2/w1))))*(1-np.e**(-k*t))*(1-np.e**(k*(t-sec))) # analytical form
    
    ch /= 5 # scale so roughly same 'height' as OFDM blocks
    
    inv_ch = np.flip(ch)
    
    return ch, inv_ch

def encode_bitstr2symbols_via_LDPC(bits, filetype):
    
    LDPC_encoded_bits=LDPC_encode(bits, file_type=filetype)
    bits=LDPC_encoded_bits

    symbols=[]
    for i in range(0,len(bits),2):
        bit1=bits[i]
        bit2=bits[i+1]
        a=1
        if bit1=="0":
            if bit2=="0":
                symbol=np.complex128(complex(1,1))
            else:
                symbol=np.complex128(complex(-1,1))
        else:
            if bit2=="0":
                symbol=np.complex128(complex(1,-1))
            else:
                symbol=np.complex128(complex(-1,-1))
        symbols.append(symbol)

    return symbols

def create_single_OFDM_frame(seed):
    """Uses pseudo-random symbols to generate single known OFDM frame"""
    
    rand_symbols = random_symbols_from_binary(N//2-1, seed) # need 1023 symbols
    frame = np.zeros(N//2, dtype=complex)
    frame[1:] = rand_symbols
    frame = np.append(frame, np.append(0, np.conj(frame)[:0:-1]))
    frame = np.fft.ifft(frame)
    frame = np.real(frame)
    frame = np.append(frame[N-prefix_no:N], frame)
    
    return frame


def create_known_OFDM_frames( seed=2021):
    """Generates 5x(N//2-1) random symbols to create 5 random
    OFDM frames that are repeated twice, for channel estimation"""
    
    print("Creating known ofdm frame")
    known_rand_symbols = random_symbols_from_binary(5*(N//2-1), seed) # need 5x1023 symbols
    split_rand_symbols = np.split(known_rand_symbols, 5) # split for 5 frames
    known_frames = []
    for i in range(5):
        frame = np.zeros(N//2, dtype=complex)
        frame[1:] = split_rand_symbols[i]
        frame = np.append(frame, np.append(0,np.conj(frame)[:0:-1]))
        frame = np.fft.ifft(frame)
        frame = np.real(frame)
        frame = np.append(frame[N-prefix_no:N], frame)
        
        known_frames = np.append(known_frames, frame)
        
    known_frames = np.tile(known_frames, 2) # doing 2x5 ofdm frames
    
    return known_frames


def data_add_known_pilots(filename, min_bin, max_bin, pilot_params, ldpc_on=True):
    # important to use different seed for frequency filler
    frequency_filler = random_symbols_from_binary(N//2-1, 2023) # just filler symbols for unused freq. range excl. pilots
    
    # pilot_params = [start, stop, step]    
    pilot_value = 1+1j
    pilot_carriers = np.arange(pilot_params[0], pilot_params[1], pilot_params[2])
    
    all_carriers = np.arange(N//2)
    usable_carriers = np.arange(min_bin, max_bin)
    data_carriers = [x for x in usable_carriers if x not in pilot_carriers]
    
    filetype = filename.split(".")[-1]
    data_bits = file_to_bitstr(filename) # uses QAM_Encoder_decoder file
    if ldpc_on:
        data_symbols= encode_bitstr2symbols_via_LDPC(data_bits, filetype) # all other variables known to encoder
    else:
        data_symbols= encode_bitstr2symbols(data_bits)

    carriers_required = int(np.ceil(len(data_symbols)/len(data_carriers)))
    
    OFDM_frames = []
    for i in range(0, carriers_required):

        frame = np.zeros(N//2, dtype=complex)
        frame[1:N//2-1] = frequency_filler[1:N//2-1]
        frame[pilot_carriers] = pilot_value        
        
        data_to_add = data_symbols[i*len(data_carriers):(i+1)*len(data_carriers)] # if not enough data symbols for last frame, frequency filler will remain
        frame[data_carriers[:len(data_to_add)]] = data_to_add

        frame = np.append(frame, np.append(0, np.conj(frame)[:0:-1]))
        OFDM_frame = np.real(np.fft.ifft(frame, N))
        OFDM_frame = np.append(OFDM_frame[N-prefix_no:N], OFDM_frame)

        OFDM_frames.append(OFDM_frame)
        
    return OFDM_frames, [pilot_carriers, data_carriers], data_bits


def create_tx_waveform(filename):
    """Creates waveform as described by the standard:
    1 sec chirp | Random OFDM block seed 2020 | 10 Known OFDM blocks seed 2021
    | Repeated payload consisting of *1 Random OFDM block seed 2022 and 10 data + pilot tone blocks* | 1 sec chirp
    Returns created waveform, the inverse chirp, num repeats, known frame"""
    
    ch, inv_ch = define_chirp(1) # need chirp for start & end
    random_frame_filler = create_single_OFDM_frame(2020) # single random frame used as a break between chirp & known OFDM
    
    known_frames = create_known_OFDM_frames(2021) # returns 2x repeat of 5 pseudo-random OFDM frames
    # / 4 to scale appropiately
    preamble_frame = create_single_OFDM_frame(2022) # preamble frame in front of 10 data frames
    
    min_bin = 50
    max_bin = 700
    pilot_params = [1, 1018, 8]
    
    data_frames, carrier_indices, data_tran = data_add_known_pilots(filename, min_bin, max_bin, pilot_params)
    
    payloads_required = int(np.ceil(len(data_frames)/10))
    print(np.shape(data_frames), payloads_required)
    payload_frames = []
    
    for i in range(payloads_required):
        payload = np.concatenate((preamble_frame, data_frames[(i)*10:(i+1)*10]), axis=None) 
        payload_frames = np.concatenate((payload_frames, payload), axis=None) # add preamble in front of 10 data frames
        
    print(len(payload_frames))
    gap = 1*fs # just to pad start & end of transmission
    tx_waveform = np.concatenate((np.zeros(gap), ch, random_frame_filler, known_frames, payload_frames, inv_ch, np.zeros(gap)), axis=None)
    
    filename_upload = 'sound_files/transmit.wav'
    sf.write(filename_upload, tx_waveform, fs)
    
    return tx_waveform, ch, inv_ch, known_frames, carrier_indices, data_tran

def create_standard():
    
    ch, inv_ch = define_chirp(1) # need chirp for start & end
    
    known_frames = create_known_OFDM_frames(2021) # returns 2x repeat of 5 pseudo-random OFDM frames
    
    pilot_params = [1, 1018, 8]
    pilot_carriers = np.arange(pilot_params[0], pilot_params[1], pilot_params[2])
    
    usable_carriers = np.arange(min_bin, max_bin)
    data_carriers = [x for x in usable_carriers if x not in pilot_carriers]
    carrier_indices = [pilot_carriers, data_carriers]
    
    return ch, inv_ch, known_frames, carrier_indices

def ideal_channel_response(signal):
    """Returns channel output for tx signal"""
    
    channel = np.genfromtxt('channel.csv',delimiter=',')
    channel_op = np.convolve(signal, channel)
    
    return channel_op

def real_channel_response(signal):
    """Records and returns rx signal after writing to file"""
    
    wait_time = np.ceil(len(signal)/fs) + 1
    print("Recording for ", wait_time, " seconds")
    

    recording = sd.rec(int(wait_time * fs), samplerate=fs, channels=1)
    sd.wait()

    sf.write('sound_files/recorded.wav', recording, fs)

    print("Finished")
    recording = recording[:, 0]
    
    return recording 


def real_channel_response_file(rec_file):
    from pydub import AudioSegment
    recording = AudioSegment.from_file(rec_file)
    recording = recording.get_array_of_samples()
    channel_op = np.array(recording)
    return channel_op

def matched_filter(signal, match1, match2):
    """Returns convolution of signal with matched filter and its peak index"""
    convolution1 = np.convolve(signal[:len(signal)//2], match1)
    convolution2 = np.convolve(signal[len(signal)//2:], match2)
    peak_index1 = np.argmax(convolution1)
    peak_index2 = np.argmax(convolution2) + len(signal)//2
    return convolution1, convolution2, peak_index1, peak_index2


def channel_estimate(signal, start, known_frames, offset=0):
    """Given a signal and the index of the chirp end, calculates the channel frequency
    and impulse response"""
    
    start += N+prefix_no - offset # account for extra filler OFDM frame of length 2304
    length = (N+prefix_no) * 10 # as using 10 ofdm frames
    trimmed_frames = signal[start:start+length]
    split_frames = np.split(trimmed_frames, 10)
    
    split_known_frames = np.split(known_frames, 10)
    
    channel_freq_response = np.zeros(N)
    
    for i in range(10):
        known_frame = split_known_frames[i][prefix_no:]
        rx_frame = split_frames[i][prefix_no:]
        known_dft = np.fft.fft(known_frame)
        rx_dft = np.fft.fft(rx_frame, N)
        
        # try using N rather than N//2 then don't append? should give same result
        single_freq_response = np.zeros(N//2, dtype=complex)
        np.divide(rx_dft[:N//2], known_dft[:N//2], out=single_freq_response, where=known_dft[:N//2] != 0) # should catch divide by zero errors
        single_freq_response = np.append(single_freq_response, np.append(0,np.conj(single_freq_response)[:0:-1]))
        
        channel_freq_response = np.add(channel_freq_response, single_freq_response)
    
    channel_freq_response /= 10 # average out
    
    channel_imp_response = np.fft.ifft(channel_freq_response, N)
    channel_imp_response = np.real(channel_imp_response)
    return channel_freq_response, channel_imp_response, split_frames


def correct_phase_decode_data(all_frames, carrier_indices, channel_fft, filename=None, pilot_value=1+1j):
    
    pilot_indices = carrier_indices[0][10:70][::3]
    data_indices = np.array(carrier_indices[1])
    
    pilot_symbols = []
    data_symbols = []
    
    preamble_points = np.arange(0, len(all_frames), 11) # remove preamble frame before each set of 10 data blocks
    data_frames = np.delete(all_frames, preamble_points, axis=0)
    
    bits = ""
    for i in range(len(data_frames)):
        
        frame_no_cp = data_frames[i][prefix_no:]
        frame_dft = np.fft.fft(frame_no_cp)

        pilots = frame_dft[pilot_indices]
        data = frame_dft[data_indices]
        
        pilots_demod = pilots / channel_fft[pilot_indices]
        pilots_phase_change = np.angle(pilots_demod / pilot_value) # divide by each known pilot symbol and get phase change
        
        phase_adjustment = np.polyfit(pilot_indices, np.unwrap(pilots_phase_change), 1)[0] # take gradient, intercept should be zero
        
        pilots *=  np.exp(-1j*phase_adjustment*pilot_indices)
        data *=  np.exp(-1j*phase_adjustment*data_indices)

        bits+=decode_symbols_2_bitstring(data, channel_fft[data_indices])
        
        pilot_symbols.append(pilots)
        data_symbols.append(data)
        
    bitstr_to_file(bits, filename)
    
    return data_symbols, pilot_symbols, bits

def correct_phase_decode_data_ldpc(all_frames, carrier_indices, channel_fft, filename=None, pilot_value=1+1j):
    
    pilot_indices = carrier_indices[0][10:70][::3]
    data_indices = np.array(carrier_indices[1])
    
    pilot_symbols = []
    data_symbols = []
    
    preamble_points = np.arange(0, len(all_frames), 11) # remove preamble frame before each set of 10 data blocks
    data_frames = np.delete(all_frames, preamble_points, axis=0)
    
    ys=np.array([])
    cks=np.array([])
    
    bits = ""
    for i in range(len(data_frames)):
        
        frame_no_cp = data_frames[i][prefix_no:]
        frame_dft = np.fft.fft(frame_no_cp)

        pilots = frame_dft[pilot_indices]
        data = frame_dft[data_indices]
        
        pilots_demod = pilots / channel_fft[pilot_indices]
        pilots_phase_change = np.angle(pilots_demod / pilot_value) # divide by each known pilot symbol and get phase change
        
        phase_adjustment = np.polyfit(pilot_indices, np.unwrap(pilots_phase_change), 1)[0] # take gradient, intercept should be zero
        
        pilots *=  np.exp(-1j*phase_adjustment*pilot_indices)
        data *=  np.exp(-1j*phase_adjustment*data_indices)

        #bits+=decode_symbols_2_bitstring(data, channel_fft[data_indices])
        
        ys=np.concatenate((ys,data))
        cks=np.concatenate((cks,channel_fft[data_indices]))
        
        pilot_symbols.append(pilots)
        data_symbols.append(data)
        
    assert len(ys) == len(cks)
    bits, file_type = LDPC_decode_with_niceCKs(ys,N, cks=cks)

    if filename:
        bitstr_to_file(bits, filename)
    else:
        bitstr_to_file(bits, "Decode/decode"+ time.strftime("%H_%M", time.localtime())+file_type)
    
    return data_symbols, pilot_symbols, bits



def fine_tuning(rx_signal, peak_start, peak_end, known_frames, inverse_chirp, carrier_indices, find_range=10, offset=20, filename=None, LDPC_on=True):
        
    score_list = []
    offset_list = []

    # Could use bianry search for higher efficiency,
    # But use linear method for now
    # since performance is not important here
    for i in range(- int(find_range/2), find_range, 1):
        
        # Compute the new offset in this round of for loop
        new_offset = i + offset
        # Append the current offset value into the list 
        offset_list.append(new_offset)

        # Compute the channel responses
        freq_response, imp_response, _ = channel_estimate(rx_signal, peak_start, known_frames, new_offset)

        # Compute the score of the impulse 
        score = impulse_score(imp_response)
        # Append the score into the list 
        score_list.append(score)
        print("index and score are:", i, score)

    # <----- Use the best score calculated to do the computation again ----->
    # Find the best score
    best_score_index = np.argmax(score_list)
    # Record the best score
    best_score = np.max(score_list)
    # Find the best offset value 
    best_offset = offset_list[best_score_index]
    
    print("Found best Offset:", best_offset, "Found bets score:", best_score )

    # Refine the starting point with the best offset
    #start_refined = start - best_offset
    

    # Redo the channel measurements with the best val
    best_freq_response, best_imp_response, _ = channel_estimate(rx_signal, peak_start, known_frames, best_offset)

    
    #expected_ch_gap = len(tx_signal) - len(inverse_chirp) - 2*fs
    #actual_ch_gap = peak_end - peak_start

    #sampling_ratio = actual_ch_gap / expected_ch_gap
    #actual_sf = 44100*sampling_ratio

    data_begin = peak_start + 11*(N+prefix_no) - best_offset # as 1 random & 10 known ofdm frames after chirp
    data_end = peak_end - len(inverse_chirp) - best_offset # use end chirp to determine signal length for now, will need a fix later

    # Since LDPC is likely to get the length right, just pad data_end to be exact one OFDM block length
    if (data_end - data_begin) % (N+prefix_no) != 0:

        # Since ending chirp is 44100 samples long, this should not go to the end and being cut off 
        data_end = data_end - (data_end - data_begin) % (N+prefix_no) + (N+prefix_no)
        
    rx_data_full = rx_signal[data_begin:data_end]
    rx_data_frames = np.split(rx_data_full, len(rx_data_full)/(N+prefix_no))

    if LDPC_on:
        _, _, bits_rec = correct_phase_decode_data_ldpc(rx_data_frames, carrier_indices, freq_response)

    else:
        _, _, bits_rec = correct_phase_decode_data(rx_data_frames, carrier_indices, best_freq_response)
    
    return bits_rec, best_imp_response

def test(rx_signal, peak_start, peak_end, known_frames, inverse_chirp, carrier_indices, offset, LDPC_on=True):
    
    # Redo the channel measurements with the best val
    freq_response, imp_response, _ = channel_estimate(rx_signal, peak_start, known_frames, offset)

    data_begin = peak_start + 11*(N+prefix_no) - offset # as 1 random & 10 known ofdm frames after chirp
    data_end = peak_end - len(inverse_chirp) - offset # use end chirp to determine signal length for now, will need a fix later

    # Since LDPC is likely to get the length right, just pad data_end to be exact one OFDM block length
    if (data_end - data_begin) % (N+prefix_no) != 0:
        # Since ending chirp is 44100 samples long, this should not go to the end and being cut off 
        data_end = data_end - (data_end - data_begin) % (N+prefix_no) + (N+prefix_no)
        
    rx_data_full = rx_signal[data_begin:data_end]
    rx_data_frames = np.split(rx_data_full, len(rx_data_full)/(N+prefix_no))

    if LDPC_on:
        _, _, bits_rec = correct_phase_decode_data_ldpc(rx_data_frames, carrier_indices, freq_response)

    else:
        _, _, bits_rec = correct_phase_decode_data(rx_data_frames, carrier_indices, freq_response)
    
    return bits_rec, imp_response