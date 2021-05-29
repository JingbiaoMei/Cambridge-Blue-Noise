import numpy as np
from QAM_EncoderDecoder import * 
from scipy.io.wavfile import write, read
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from IPython.display import Audio
from scipy import interpolate
import os



# Chirp 
def define_chirp():
        """returns standard log chirp waveform and its time-reverse"""
        
        sec = 1
        k = 100
        w1 = 60
        w2 = 6000
        
        t = np.linspace(0, sec, int(fs*sec))
        
        chirp = np.sin(2*np.pi * w1 * sec * (np.exp( t * (np.log(w2 / w1) / sec)) -1) /np.log(w2 / w1) )
        chirp *= (1-np.exp(-k*t))*(1-np.exp(-k*(sec-t))) / 5
        
        inv_chirp = np.flip(chirp)
        
        return chirp, inv_chirp


# OFDM 

class OFDM():
    
    def __init__(self, N, reptition, prefix_no, fs):
        self.N = N
        self.prefix_no = prefix_no
        self.repetition = reptition
        self.fs = fs

    # Random symbol for channel estimation 
    def generate_random_symbols(self): 
        random_symbol = np.array([81, 41, 51, 46, 19, 27, 84, 53,  0, 86, 54, 17, 33, 32,  8, 24, 19,
            38, 58, 28, 74, 10, 39, 24, 39, 22, 59, 58, 70, 74, 82, 64,  4, 77,
            98, 50, 26, 36, 21, 32, 56, 27, 92, 42, 63, 91, 67, 76, 65, 40, 17,
            49, 66, 42, 87, 20, 50, 89, 48, 47, 40, 29, 57, 40, 92, 73,  8, 26,
            12, 76, 24, 82, 43, 14, 40, 19, 56, 97, 78, 43, 96, 43, 89,  6, 11,
            98, 58, 25, 91, 16, 40, 77, 52,  9, 34, 45, 36, 69, 12, 29, 38, 45,
            88, 14, 20, 49,  1, 61, 48, 36, 10, 44, 44,  5,  7, 34, 26, 72,  7,
            63, 68, 27, 12, 71, 39, 54, 96,  1, 70, 67, 76, 30, 77, 73, 28, 88,
            31, 17, 86, 62,  1, 12, 35, 74,  3, 87, 73, 26, 83, 73,  6,  3, 32,
            37, 39, 53, 90, 88, 60, 89, 93, 91,  4, 53,  5,  4,  4, 58, 35, 63,
            27, 77, 51, 87, 24, 31, 16,  4, 87, 98, 52, 90, 68, 37, 75, 56, 34,
            30, 50, 26, 20, 96, 51, 94, 60, 55, 14, 74,  4, 73, 13, 45, 67,  8,
            61, 12, 93,  6, 87, 14, 90, 64, 33, 29, 68, 13, 60, 18,  9, 60,  3,
            15,  6, 48, 34, 44, 63, 25, 39, 18,  5, 56, 38, 46,  6, 64, 36, 29,
            90, 47, 23, 29, 97, 19,  5, 47, 30, 63, 98, 99, 20, 91, 69, 24, 35,
            59])

        bin_strings=''
        for byte in random_symbol:
            binary_string = '{0:08b}'.format(byte)
            bin_strings+=binary_string

        full_symbols = encode_bitstr2symbols(bin_strings)
        symbols = full_symbols[:self.N//2-1] # fit the symbols in the 1st half of the OFDM frame  

        return symbols

    def known_OFDM_frame(self, symbols):
        """Takes N//2 symbols and returns single real OFDM frame"""
        
        block = [0]
        block[1:] = symbols
        while (len(block) <= self.N//2):
            block.append(0)
        for j in range(self.N//2 - 1,0,-1):
            block.append(np.conj(block[j]))

        frame = np.fft.ifft(block, n=self.N)
        frame = np.real(frame)
        
        return frame
    
    #### Pilot symbols TOdo
    def OFDM_symbol(self, QPSK_payload):
        """Assigns pilot values and payload values to OFDM symbol, take reverse complex conjugate and append to end to make signal passband"""
        
        symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
        symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
        symbol[dataCarriers] = QPSK_payload  # allocate the data subcarriers
        symbol = np.append(symbol, np.append(0,np.conj(symbol)[:0:-1]))
        
        return symbol
        

    def tx_waveform(self, frame, chirp, repeats, pilots=False, pilot_frame = []):
        """Returns chirp/s with repeated OFDM frame, and length of known frames"""
        
        gap = int(1 * self.fs)
        frames = np.tile(frame, repeats)
        if not pilots:
            waveform = np.concatenate((np.zeros(gap), chirp, frames, np.zeros(gap)), axis=None)
        else:
            waveform = np.concatenate((np.zeros(gap), chirp, frames, np.zeros(gap), chirp, np.zeros(gap), pilot_frame, np.zeros(gap)), axis=None)
        
        return waveform, repeats

    def tx_waveform_data(self, frame, chirp, repeats, filename):
        
        gap = int(1 * self.fs)
        frames = np.tile(frame, repeats)

        bits_tran = file_to_bitstr(filename)
        symbols_tran = encode_bitstr2symbols(bits_tran)
        data_tran = symbol_to_OFDMframes(symbols_tran, self.N, self.prefix_no)
        data_tran = np.real(data_tran)
        data_length = data_tran.shape[0]*data_tran.shape[1]
        waveform = np.concatenate((np.zeros(gap), chirp, frames, data_tran, np.zeros(gap)), axis=None)
        
        return waveform, repeats, data_length



    def ideal_channel_response(self, signal):
        """Returns channel output for tx signal"""
        
        channel = np.genfromtxt('channel.csv',delimiter=',')
        channel_op = np.convolve(signal, channel)
        
        return channel_op

    def real_channel_response(self, signal):
        """Records and returns rx signal after writing to file"""
        
        print("Recording")
        wait_time = np.ceil(len(signal)/self.fs) + 1

        recording = sd.rec(int(wait_time * self.fs), samplerate=self.fs, channels=1)
        sd.wait()

        sf.write('sound_files/sync_long_rec.wav', recording, self.fs)

        print("Finished")
        recording = recording[:, 0]
        
        return recording  


    def matched_filter(self, signal, match):
        """Returns convolution of signal with matched filter and its peak index"""
        
        convolution = np.convolve(signal, match)
        peak_index = np.argmax(convolution[0:len(convolution//2)])
        
        return convolution, peak_index



    def process_transmission(self, signal, start, repeats, offset=0):
        """Returns trimmed and averaged known OFDM symbol"""
        
        start += offset
        length = repeats * self.N
        trimmed_frames = signal[start:start+length]
        split_frames = np.split(trimmed_frames, repeats)
        average_frame = np.zeros(self.N)
        for frame in split_frames:
            average_frame = np.add(average_frame, frame)
        average_frame /= (repeats)
        
        return average_frame



    def estimate_channel_response(self, frame, known_frame):
        """Returns time and frequency channel impulse response from known OFDM symbols"""
        
        known_symbols = np.fft.fft(known_frame)
        
        OFDM_frame = np.fft.fft(frame, self.N)
        channel_freq_response = OFDM_frame / known_symbols
        channel_freq_response[self.N // 2] = 0 # avoid NaN error. error when not all bins filled, needs a fix
        
        channel_imp_response = np.fft.ifft(channel_freq_response, self.N)
        channel_imp_response = np.real(channel_imp_response)
        
        return channel_freq_response, channel_imp_response
            

    










def bitrate(file, audio, fs):
    '''
    file: file path
    audio: audio numpy array
    fs: sampling frequency
    '''
    size_byte = os.stat(file).st_size
    audio_length = len(audio) / fs
    rate = size_byte * 8 / audio_length / 1024
    print('bitrate of the system is:', str.format('{0:.2f}', rate), 'Kbits/s')







def offset_correction(impulse, offset):
