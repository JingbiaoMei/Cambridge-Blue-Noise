import numpy as np
from QAM_EncoderDecoder import *
from scipy.io.wavfile import write, read
import sounddevice as sd
import soundfile as sf
#from IPython.display import Audio
#from scipy import interpolate
import os
from pydub import AudioSegment


# Util functions

def error_rate(bits_tran, bits_rec):
    length = len(bits_tran)
    a1 = np.fromstring(bits_tran, 'u1') - ord('0')
    a2 = np.fromstring(bits_rec, 'u1') - ord('0')
    a2 = a2[0:length]
    return (length - np.sum(a1 == a2)) / length


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
    return rate


def impulse_score(impulse):
    # The input impulse should be normalised (Minus its initial mean)
    initial_no = 80
    last_no = 80
    impulse = np.real(impulse)
    impulse = impulse - np.average(impulse)
    score = np.average(np.abs(impulse[0:initial_no])) / \
        np.average(np.abs(impulse[:-last_no]))
    return score


# OFDM

class OFDM():

    # Initialize parameters
    def __init__(self, N, prefix_no, fs, repeat, gap_second, seed = 2021):
        # DFT length
        self.N = N

        # Cyclic prefix length
        self.prefix_no = prefix_no

        # Sampling frequency
        self.fs = fs

        # Repetition of Known OFDM symbols for channel measurements
        self.repeat = repeat

        # Gap in between different signals
        self.gap = gap_second * self.fs

        # Random seeds standardised across teams
        self.seed = seed


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

        bin_strings = ''
        for byte in random_symbol:
            binary_string = '{0:08b}'.format(byte)
            bin_strings += binary_string

        full_symbols = encode_bitstr2symbols(bin_strings)
        # fit the symbols in the 1st half of the OFDM frame
        symbols = full_symbols[:self.N//2-1]

        return symbols

    def generate_random_symbols_seeds(self):
        rng = np.random.default_rng(self.seed)
        length_random_sequence = self.N//2 -1
        random_sequence = rng.integers(low=0, high=4, size=length_random_sequence)

        mapping = {
            0:  1+1j,
            1: -1+1j,
            2: -1-1j,
            3:  1-1j
        }

        random_symbols = [mapping[r] for r in random_sequence]
        return random_symbols


    def known_OFDM_frame(self, symbols):
        """Takes N//2 symbols and returns single real OFDM frame"""

        # change info bins to just 256 corresponding to ~ 11kHz
        # so here now 256 info bins - same as num symbols -> so one OFDM symbol
        info_bins = int(self.N//2)-1
        OFDM_frames = []

        # start from 3rd bin (~82 Hz onward) too??

        OFDM_frames = []
        # for each OFDM block

        for i in range(0, len(symbols), info_bins):
            # frequency bins 0 and 512(int(N/2)) contains value 0
            OFDM_block = [0]
            # start from 3rd bin? depends on bandwidth, N etc.
            OFDM_block[1:] = symbols[i:i+info_bins]

            # add 0s to the end when data is not an integer factor of 512
            # change from info_bins as thats now reduced - crucial for conjugate symmetry
            while len(OFDM_block) <= ((self.N/2)):
                OFDM_block.append(0)
            # merge lines above/below?
            # OFDM_block.append(0)#frequency bins 0 and 512(int(N/2)) contains value 0

            # reverse conjugate
            for j in range(len(OFDM_block)-2, 0, -1):  # count up or down
                OFDM_block.append(np.conj(OFDM_block[j]))

            # ----iDFT----
            OFDM_frame = np.fft.ifft(OFDM_block, n=self.N)

            # ----add cyclic prefix----
            #cyclic_prefix = OFDM_frame[self.N-self.prefix_no:]

            #OFDM_frame = np.append(cyclic_prefix, OFDM_frame, axis=0)
            OFDM_frames.append(OFDM_frame)

        # only to get rid of +0j parts after iFFT
        OFDM_frames = np.real(OFDM_frames) * 10
        frame = OFDM_frames[0]
        return frame

    # Chirp
    def define_chirp(self):
        """returns standard log chirp waveform and its time-reverse"""

        sec = 1
        k = 100
        w1 = 60
        w2 = 6000

        t = np.linspace(0, sec, int(self.fs*sec))

        chirp = np.sin(2*np.pi * w1 * sec * (np.exp(t *
                                                    (np.log(w2 / w1) / sec)) - 1) / np.log(w2 / w1))
        chirp *= (1-np.exp(-k*t))*(1-np.exp(-k*(sec-t))) / 5

        inv_chirp = np.flip(chirp)

        return chirp, inv_chirp


    def tx_waveform(self, frame, chirp,  pilots=False, pilot_frame=[]):
        """Returns chirp/s with repeated OFDM frame, and length of known frames"""

        frames = np.tile(frame, self.repeat)
        if not pilots:
            waveform = np.concatenate(
                (np.zeros(self.gap), chirp, frames, np.zeros(self.gap)), axis=None)
        else:
            waveform = np.concatenate((np.zeros(self.gap), chirp, frames, np.zeros(
                self.gap), chirp, np.zeros(self.gap), pilot_frame, np.zeros(self.gap)), axis=None)

        return waveform

    def tx_waveform_data(self, frame, chirp, filename):

        frames = np.tile(frame, self.repeat)

        bits_tran = file_to_bitstr(filename)
        symbols_tran = encode_bitstr2symbols(bits_tran)
        data_tran = symbol_to_OFDMframes(symbols_tran, self.N, self.prefix_no)
        data_tran = np.real(data_tran)
        data_length = data_tran.shape[0]*data_tran.shape[1]
        waveform = np.concatenate(
            (np.zeros(self.gap), chirp, frames, data_tran, np.zeros(self.gap)), axis=None)

        return waveform, data_length, bits_tran

    def tx_waveform_data_pilot(self, frame, chirp, filename):

        frames = np.tile(frame, self.repeat)

        bits_tran = file_to_bitstr(filename)
        symbols_tran = encode_bitstr2symbols(bits_tran)
        data_tran = symbol_to_OFDMframes(symbols_tran, self.N, self.prefix_no)
        data_tran = np.real(data_tran)
        data_length = data_tran.shape[0]*data_tran.shape[1]
        waveform = np.concatenate(
            (np.zeros(self.gap), chirp, frames, data_tran, np.zeros(self.gap)), axis=None)

        return waveform, data_length, bits_tran

    

    def ideal_channel_response(self, signal):
        """Returns channel output for tx signal"""

        channel = np.genfromtxt('channel.csv', delimiter=',')
        channel_op = np.convolve(signal, channel)

        return channel_op

    def real_channel_response(self, signal, filename='sound_files/sync_long_rec.wav'):
        """Records and returns rx signal after writing to file"""

        print("Recording")
        wait_time = np.ceil(len(signal)/self.fs) + 1

        recording = sd.rec(int(wait_time * self.fs),
                           samplerate=self.fs, channels=1)
        sd.wait()

        sf.write(filename, recording, self.fs)

        print("Finished")
        recording = recording[:, 0]

        return recording

    def real_channel_response_file(self, rec_file):

        recording = AudioSegment.from_file(rec_file)
        recording = recording.get_array_of_samples()
        channel_op = np.array(recording)
        return channel_op

    def matched_filter(self, signal, match):
        """Returns convolution of signal with matched filter and its peak index"""

        convolution = np.convolve(signal, match)
        peak_index = np.argmax(convolution[0:len(convolution//2)])

        return convolution, peak_index

    def matched_filter_double(self, signal, match):
        """Returns convolution of signal with matched filter and its peak index"""
    
        convolution = np.convolve(signal, match)
        peak_index1 = np.argmax(np.abs(convolution[:len(convolution)//2])) # check 1st half of signal
        peak_index2 = np.argmax(np.abs(convolution[len(convolution)//2:])) + len(convolution)//2 # check 2nd half of signal
    
        return convolution, peak_index1, peak_index2

    def find_start(self, peak):
        ''' 
        Returns the starting point of the known OFDM symbols by maximum point 
        Requires Fine Tuning afterwards
        '''

        start = peak + self.gap
        return start

    def process_transmission(self, signal, start, offset=0):
        """Returns trimmed and averaged known OFDM symbol"""

        start -= offset
        length = self.repeat * self.N
        trimmed_frames = signal[start:start+length]
        split_frames = np.split(trimmed_frames, self.repeat)
        average_frame = np.zeros(self.N)
        for frame in split_frames:
            average_frame = np.add(average_frame, frame)
        average_frame /= (self.repeat)

        # Keep a record for the refined starting point with the offset
        start_refined = start

        return average_frame, start_refined


    def process_transmission_pilot(self, signal, start, offset=0):
    
        start -= offset
        # Here with cyclic prefix
        length = self.repeat * (self.N + self.prefix_no)
        trimmed_frames = signal[start:start+length]
        split_frames = np.split(trimmed_frames, length/(self.N+self.prefix_no))
        
        average_frame = np.zeros(self.N+self.prefix_no)
        for frame in split_frames:     
            average_frame = np.add(average_frame, frame)
        average_frame /= (length/(self.N+self.prefix_no))
        
        # Keep a record for the refined starting point with the offset
        start_refined = start


        return split_frames, average_frame, start_refined

    def estimate_channel_response(self, frame, known_frame):
        """Returns time and frequency channel impulse response from known OFDM symbols"""

        known_symbols = np.fft.fft(known_frame)

        OFDM_frame = np.fft.fft(frame, self.N)
        channel_freq_response = OFDM_frame / known_symbols
        # avoid NaN error. error when not all bins filled, needs a fix
        channel_freq_response[self.N // 2] = 0

        channel_imp_response = np.fft.ifft(channel_freq_response, self.N)
        channel_imp_response = np.real(channel_imp_response)

        return channel_freq_response, channel_imp_response



    def estimate_channel_response_pilot(self, frame, known_frame):
        """Returns time and frequency channel impulse response from known OFDM symbols"""
        
        known_frame = known_frame[self.prefix_no:]
        frame = frame[self.prefix_no:] 
        
        known_symbols = np.fft.fft(known_frame, self.N)
        OFDM_symbol = np.fft.fft(frame, self.N)
        
        channel_freq_response = OFDM_symbol / known_symbols
        channel_freq_response[self.N//2] = 0 # avoid NaN error. error when not all bins filled, needs a fix
        channel_freq_response[0] = 0
        channel_imp_response = np.fft.ifft(channel_freq_response, self.N)
        channel_imp_response = np.real(channel_imp_response)
        
        return channel_freq_response, channel_imp_response

    def retrieve_info(self, signal, start_refined):
        '''
        Return parallel inforamtion OFDM frames
        '''

        # Length of the known OFDM symbols for channel estimation
        length = self.repeat * self.N
        start_info = start_refined + length

        information = signal[start_info:-1]

        # Zero padding
        information = np.concatenate((information, np.zeros(
            self.N+self.prefix_no - len(information) % (self.N+self.prefix_no))), axis=None)

        # Serial to Parrallel
        information = information.reshape((-1, self.N+self.prefix_no))

        return information

    # This function can not be used in actual transmission, since bits_tran is unknown in an
    # actual transmission system

    def sync_error(self, signal, start, offset, bits_tran, known_frame, fileout='decode.txt'):

        # Get the average received frames by taking average of the
        # repetition of the transmitted knwon OFDM symbols
        avg_frame, start_refined = self.process_transmission(
            self, signal, start, offset)

        # Compute the channel responses
        freq_response, imp_response = self.estimate_channel_response(
            self, avg_frame, known_frame)

        information = self.retrieve_info(self, signal, start_refined)

        score = impulse_score(imp_response)
        bits_rec = OFDMframes_to_bitstring(
            information, self.N, self.prefix_no, freq_response)

        if fileout:
            bitstr_to_file(bits_rec, fileout)

        error = error_rate(bits_tran, bits_rec)
        return error, score

    def fine_tune(self, signal, start, known_frame, find_range, offset=20):
        score_list = []
        offset_list = []

        # Could use bianry search for higher efficiency,
        # But use linear method for now
        # since performance is not important here
        for i in range(- int(find_range/2), find_range, 1):
            
            new_offset = i + offset
            offset_list.append(new_offset)

            avg_frame, _ = self.process_transmission(
                self, signal, start, new_offset)

            # Compute the channel responses
            _, imp_response = self.estimate_channel_response(
                self, avg_frame, known_frame)

            score = impulse_score(imp_response)
            score_list.append(score)
            #print("index and score are:", i, score)

        # Use the best score calculated to do the computation again
        best_score_index = np.argmax(score_list)
        best_score = np.max(score_list)
        best_offset = offset_list[best_score_index]
        start_refined = start - best_offset

        avg_frame, _ = self.process_transmission(
                self, signal, start, best_offset)

        # Compute the channel responses
        freq_response, best_imp_response = self.estimate_channel_response(
                self, avg_frame, known_frame)

        information = self.retrieve_info(self, signal, start_refined)
        bits_rec = OFDMframes_to_bitstring(
            information, self.N, self.prefix_no, freq_response)

        return best_offset, best_score, bits_rec, best_imp_response


    # <-------- Pilot Symbols --------->
    def data_to_OFDM(self, filename):
    
        bits_tran = file_to_bitstr(filename)
        symbols_tran = encode_bitstr2symbols(bits_tran)
        data_tran = symbol_to_OFDMframes(symbols_tran, self.N, self.prefix_no)
        data_tran = np.real(data_tran)
        
        return data_tran

    def data_add_pilots(self,  filename):
        
    
        rng = np.random.default_rng(self.seed)
        length_random_sequence = self.N//2 # need 511 extra symbols
        random_sequence = rng.integers(low=0, high=4, size=length_random_sequence)

        mapping = {
            0:  1+1j,
            1: -1+1j,
            2: -1-1j,
            3:  1-1j}

        frequency_filler = [mapping[r] for r in random_sequence]

        
        
        K = self.N//4 # so 512 info bins, might reduce in future
        P = K // 16
        pilotValue = 1 + 1j
        
        allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
        pilotCarriers = allCarriers[1::K//P] # Pilots are every (K//P)th carrier.
        dataCarriers = np.delete(allCarriers, pilotCarriers)
        dataCarriers = np.delete(dataCarriers, [0])
        
        data_bits = file_to_bitstr(filename)
        data_symbols= encode_bitstr2symbols(data_bits)

        carriers_required = int(np.ceil(len(data_symbols)/len(dataCarriers)))
        excess = int(len(dataCarriers) * carriers_required) - len(data_symbols)
        data_symbols = np.append(data_symbols, frequency_filler[:excess])
        print(carriers_required, excess, len(data_symbols), len(dataCarriers))

        OFDM_frames = []
        for i in range(0, carriers_required*len(dataCarriers), len(dataCarriers)):
            
            OFDM_symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
            OFDM_symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
            OFDM_symbol[dataCarriers] = data_symbols[i:i+len(dataCarriers)]  # allocate the data subcarriers  
            OFDM_symbol = np.append(OFDM_symbol, frequency_filler[:self.N//2-K])
            OFDM_symbol = np.append(OFDM_symbol, np.append(0,np.conj(OFDM_symbol)[:0:-1]))
            OFDM_symbol = np.fft.ifft(OFDM_symbol)
            OFDM_symbol = np.append(OFDM_symbol[self.N-self.prefix_no:self.N], OFDM_symbol)   
            OFDM_symbol = np.real(OFDM_symbol)
            OFDM_frames.append(OFDM_symbol)   
            
        return OFDM_frames, [allCarriers, pilotCarriers, dataCarriers], data_bits

    def data_remove_pilots(self, all_frames, carrier_indices, channel_fft, filename=None):
        
        pilot_indices = carrier_indices[1]
        data_indices = carrier_indices[2]
        
        pilot_symbols = []
        data_symbols = []
        bits = ""
        for i in range(len(all_frames)):
            
            frame_no_cp = all_frames[i][self.prefix_no:]
            frame_dft = np.fft.fft(frame_no_cp)

            pilots = frame_dft[pilot_indices]
            data = frame_dft[data_indices]
            
            bits+=decode_symbols_2_bitstring(data, channel_fft[data_indices])
            
            pilot_symbols.append(pilots)
            data_symbols.append(data)
        
        if filename:
            bitstr_to_file(bits, filename)
        
        return data_symbols, pilot_symbols, bits

    def data_remove_pilots_correct_phase(self, all_frames, carrier_indices, channel_fft, filename):
    
        pilot_indices = carrier_indices[1]
        data_indices = carrier_indices[2]
        
        pilot_symbols = []
        data_symbols = []
        bits = ""
        for i in range(len(all_frames)):
            
            frame_no_cp = all_frames[i][self.prefix_no:]
            frame_dft = np.fft.fft(frame_no_cp)

            pilots = frame_dft[pilot_indices]
            data = frame_dft[data_indices]
            
            pilots_phase = np.unwrap(np.angle(pilots / channel_fft[carrier_indices[1]])) # very basic linear correction of phase
            phase_adjustment = np.polyfit(carrier_indices[1], pilots_phase, 1)[0]
            
            pilots *=  np.exp(-1j*phase_adjustment*carrier_indices[1])
            data *=  np.exp(-1j*phase_adjustment*carrier_indices[2])
            
            bits+=decode_symbols_2_bitstring(data, channel_fft[data_indices])
            
            pilot_symbols.append(pilots)
            data_symbols.append(data)
        
        bitstr_to_file(bits, filename)
        
        return data_symbols, pilot_symbols, bits