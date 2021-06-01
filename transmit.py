import numpy as np
from QAM_EncoderDecoder import *
from scipy.io.wavfile import write, read
import sounddevice as sd
import soundfile as sf
#from IPython.display import Audio
#from scipy import interpolate
import os
from pydub import AudioSegment
from LDPCQAMCombined import *


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
    initial_no = 50
    last_no = 100
    impulse = np.real(impulse)
    impulse = impulse - np.average(impulse)
    score = np.average(np.abs(impulse[0:initial_no])) / \
        np.average(np.abs(impulse[:-last_no]))
    return score


# OFDM

class OFDM():

    # Initialize parameters
    def __init__(self, N, prefix_no, fs, repeat, gap_second, chirp_high, min_bin=20, max_bin=575, seed = 2021,inputLenIndicator_len=24,rate='1/2',r=0.5,z=27,len_protection='input_repeat_then_LDPC',repeat_times=3):
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

        # Chirp high frequency
        self.chirp_high = chirp_high

        # Minimum frequency bin of the OFDM
        self.min_bin = min_bin

        # Maximum frequency bin of the OFDM
        self.max_bin = max_bin


        # parameters for LDPC
        self.rate = rate
        self.r = r
        self.z = z
        
        # number of bits used to express file_length
        self.inputLenIndicator_len = inputLenIndicator_len 

        # method to protect file_length information
        self.len_protection = len_protection
        self.repeat_times = repeat_times
        # e.g. if inputLenIndicator_len is 24, and repeat_times is 3, this means 24x3 bits before LDPC encoding will be used to hold the file_length information


    # Random symbol for channel estimation
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

    def generate_known_OFDM(self, symbols):
        known_frame = symbol_to_OFDMframes(symbols, self.N, self.prefix_no)[0]
        known_frames = np.tile(known_frame, self.repeat)
        return known_frame, known_frames


    # Chirp
    def define_chirp(self):
        """returns standard log chirp waveform and its time-reverse"""

        sec = 1
        k = 50
        w1 = 100
        w2 = self.chirp_high

        t = np.linspace(0, sec, int(self.fs*sec))

        chirp = np.sin(2*np.pi * w1 * sec * (np.exp(t *
                                                    (np.log(w2 / w1) / sec)) - 1) / np.log(w2 / w1))
        chirp *= (1-np.exp(-k*t))*(1-np.exp(-k*(sec-t))) / 5

        inv_chirp = np.flip(chirp)

        return chirp, inv_chirp

    def tx_waveform_data(self, frame, chirp, filename,LDPC=False):

        frames = np.tile(frame, self.repeat)

        bits_tran = file_to_bitstr(filename)

        if LDPC:
            symbols_tran = encode_bitstr2symbols_via_LDPC(bits_tran,inputLenIndicator_len=self.inputLenIndicator_len, N=self.N,rate=self.rate,r=self.r,z=self.z,len_protection=self.len_protection,repeat_times=self.repeat_times,test=False)
        else:
            symbols_tran = encode_bitstr2symbols(bits_tran)
        
        data_tran = symbol_to_OFDMframes(symbols_tran, self.N, self.prefix_no)
        data_tran = np.real(data_tran)
        data_length = data_tran.shape[0]*data_tran.shape[1]
        waveform = np.concatenate(
            (np.zeros(self.gap), chirp, frames, data_tran, np.zeros(self.gap)), axis=None)

        return waveform, data_length, bits_tran

    def tx_waveform_data_pilot(self, known_frames, chirp, data_frames):
        """
        Return the tx signal for the pilot version
        """

        gap = self.gap
        tx_signal = np.concatenate((np.zeros(gap), chirp, known_frames, np.zeros(gap), data_frames, np.zeros(gap), chirp, np.zeros(gap)), axis=None)
        tx_signal = np.real(tx_signal)

        return tx_signal

    
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

    def sync_error(self, signal, start, offset, bits_tran, known_frame, fileout='decode.txt',LDPC=False):

        # Get the average received frames by taking average of the
        # repetition of the transmitted knwon OFDM symbols
        avg_frame, start_refined = self.process_transmission(
            self, signal, start, offset)

        # Compute the channel responses
        freq_response, imp_response = self.estimate_channel_response(
            self, avg_frame, known_frame)

        information = self.retrieve_info(self, signal, start_refined)

        score = impulse_score(imp_response)
        if LDPC:
            bits_rec = OFDMframes_to_bitstring_via_LDPC(
                information, self.N, self.prefix_no, freq_response,inputLenIndicator_len=self.inputLenIndicator_len, N=self.N,rate=self.rate,r=self.r,z=self.z,len_protection=self.len_protection,repeat_times=self.repeat_times)
        else:
            bits_rec = OFDMframes_to_bitstring(
                information, self.N, self.prefix_no, freq_response)

        if fileout:
            bitstr_to_file(bits_rec, fileout)

        error = error_rate(bits_tran, bits_rec)
        return error, score


    

    def fine_tune(self, signal, start, known_frame, find_range, offset=20,LDPC=False):
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

        if LDPC:
            bits_rec = OFDMframes_to_bitstring_via_LDPC(
                information, self.N, self.prefix_no, freq_response,inputLenIndicator_len=self.inputLenIndicator_len, N=self.N,rate=self.rate,r=self.r,z=self.z,len_protection=self.len_protection,repeat_times=self.repeat_times)
        else:
            bits_rec = OFDMframes_to_bitstring(
                information, self.N, self.prefix_no, freq_response)
        

        return best_offset, best_score, bits_rec, best_imp_response


    # <-------- Pilot Symbols --------->
    def data_to_OFDM(self, filename,LDPC=False):
    
        bits_tran = file_to_bitstr(filename)
        if LDPC:
            symbols_tran = encode_bitstr2symbols_via_LDPC(bits_tran,inputLenIndicator_len=self.inputLenIndicator_len, N=self.N,rate=self.rate,r=self.r,z=self.z,len_protection=self.len_protection,repeat_times=self.repeat_times,test=False)
        else:
            symbols_tran = encode_bitstr2symbols(bits_tran)
        data_tran = symbol_to_OFDMframes(symbols_tran, self.N, self.prefix_no)
        data_tran = np.real(data_tran)
        
        return data_tran

    def data_add_pilots(self,  filename,LDPC=False):
        
    
        rng = np.random.default_rng(self.seed)
        length_random_sequence = self.N//2 - 1 # need 511 extra symbols
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
        if LDPC:
            data_symbols= encode_bitstr2symbols_via_LDPC(data_bits,inputLenIndicator_len=self.inputLenIndicator_len, N=self.N,rate=self.rate,r=self.r,z=self.z,len_protection=self.len_protection,repeat_times=self.repeat_times,test=False)
        else:
            data_symbols= encode_bitstr2symbols(data_bits)


        carriers_required = int(np.ceil(len(data_symbols)/len(dataCarriers)))
        excess = int(len(dataCarriers) * carriers_required) - len(data_symbols)
        data_symbols = np.append(data_symbols, frequency_filler[:excess])

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
        
        OFDM_frames = np.ravel(OFDM_frames)
        return OFDM_frames, [allCarriers, pilotCarriers, dataCarriers], data_bits


    def data_add_random_pilots(self, filename,LDPC=False):

        frequency_filler = self.generate_random_symbols_seeds(self)
        
        all_carriers = np.arange(self.min_bin, self.max_bin) # min_bin, ..., max_bin-1
        pilot_ratio = 16
        pilot_carriers = all_carriers[0:-1:pilot_ratio]
        data_carriers = np.delete(all_carriers, pilot_carriers- self.min_bin)
        
        data_bits = file_to_bitstr(filename)

        if LDPC:
            data_symbols= encode_bitstr2symbols_via_LDPC(data_bits,inputLenIndicator_len=self.inputLenIndicator_len, N=self.N,rate=self.rate,r=self.r,z=self.z,len_protection=self.len_protection,repeat_times=self.repeat_times,test=False)
        else:
            data_symbols= encode_bitstr2symbols(data_bits)
        
        carriers_required = int(np.ceil(len(data_symbols)/len(data_carriers)))
        
        OFDM_frames = []
        for i in range(0, carriers_required):

            frame = np.zeros(self.N//2, dtype=complex)
            frame[1:self.N//2-1] = frequency_filler[1:self.N//2-1]
            data_to_add = data_symbols[i*len(data_carriers):(i+1)*len(data_carriers)]
            frame[data_carriers[:len(data_to_add)]] = data_to_add

            frame = np.append(frame, np.append(0, np.conj(frame)[:0:-1]))
            OFDM_frame = np.real(np.fft.ifft(frame, self.N))
            OFDM_frame = np.append(OFDM_frame[self.N-self.prefix_no:self.N], OFDM_frame)

            OFDM_frames.append(OFDM_frame)

        OFDM_frames = np.ravel(OFDM_frames)

        underfill = len(data_to_add) # for last frame only

        pilot_values = [frequency_filler[x] for x in pilot_carriers]
        

        return OFDM_frames, underfill, [all_carriers, pilot_carriers, data_carriers], pilot_values, data_bits

    def data_remove_pilots_correct_phase(self, all_frames, carrier_indices, channel_fft, filename=None):
    
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
        
        if filename:
            bitstr_to_file(bits, filename)
        
        return data_symbols, pilot_symbols, bits


    def data_remove_random_pilots_correct_phase(self, all_frames, carrier_indices, channel_fft,  pilot_values, underfill=0, filename= None):
    
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
            
            pilots_demod = pilots / channel_fft[pilot_indices]
            pilots_phase_change = np.angle(pilots_demod / pilot_values) # divide by each known pilot symbol and get phase change
            
            phase_adjustment = np.polyfit(pilot_indices, np.unwrap(pilots_phase_change), 1)[0] # take gradient, intercept should be zero
            
            pilots *=  np.exp(-1j*phase_adjustment*carrier_indices[1])
            data *=  np.exp(-1j*phase_adjustment*carrier_indices[2])
            
            if (i==len(all_frames)-1) and (underfill != 0):
                print("Last frame")
                bits+=decode_symbols_2_bitstring(data[:underfill], channel_fft[data_indices][:underfill])
            else:
                bits+=decode_symbols_2_bitstring(data, channel_fft[data_indices])
            
            pilot_symbols.append(pilots)
            data_symbols.append(data)
            
        if filename:
            bitstr_to_file(bits, filename)
        
        return data_symbols, pilot_symbols, bits



    def data_remove_random_pilots_correct_phase_LDPC(self, all_frames, carrier_indices, channel_fft,  pilot_values, underfill=0, filename= None):
    
        pilot_indices = carrier_indices[1]
        data_indices = carrier_indices[2]
        
        pilot_symbols = []
        data_symbols = []
        
        # bits = ""

        ys=np.array([])
        cks=np.array([])

        for i in range(len(all_frames)):
            
            frame_no_cp = all_frames[i][self.prefix_no:]
            frame_dft = np.fft.fft(frame_no_cp)

            pilots = frame_dft[pilot_indices]
            data = frame_dft[data_indices]
            
            pilots_demod = pilots / channel_fft[pilot_indices]
            pilots_phase_change = np.angle(pilots_demod / pilot_values) # divide by each known pilot symbol and get phase change
            
            phase_adjustment = np.polyfit(pilot_indices, np.unwrap(pilots_phase_change), 1)[0] # take gradient, intercept should be zero
            
            pilots *=  np.exp(-1j*phase_adjustment*carrier_indices[1])
            data *=  np.exp(-1j*phase_adjustment*carrier_indices[2])
            
            if (i==len(all_frames)-1) and (underfill != 0):
                print("Last frame")
                ys=np.concatenate((ys,data[:underfill]))
                cks=np.concatenate((cks,channel_fft[data_indices][:underfill]))
                # bits+=LDPC_decode_with_niceCKs(data[:underfill],self.N,rate=self.rate,r=self.r,z=self.z,inputLenIndicator_len=self.inputLenIndicator_len, cks=ck,len_protection=self.len_protection,repeat_times=self.repeat_times)
    
                # bits+=decode_symbols_2_bitstring(data[:underfill], channel_fft[data_indices][:underfill])
            else:
                ys=np.concatenate((ys,data))
                cks=np.concatenate((cks,channel_fft[data_indices]))
                # ys.append(data)
                # cks.append(channel_fft[data_indices])
                # bits+=LDPC_decode_with_niceCKs(data,self.N,rate=self.rate,r=self.r,z=self.z,inputLenIndicator_len=self.inputLenIndicator_len, cks=ck,len_protection=self.len_protection,repeat_times=self.repeat_times)
    
                # bits+=decode_symbols_2_bitstring(data, channel_fft[data_indices])
            


            pilot_symbols.append(pilots)
            data_symbols.append(data)
            
        assert len(ys)==len(cks)
        bits=LDPC_decode_with_niceCKs(ys,self.N,rate=self.rate,r=self.r,z=self.z,inputLenIndicator_len=self.inputLenIndicator_len, cks=cks,len_protection=self.len_protection,repeat_times=self.repeat_times)

        if filename:
            bitstr_to_file(bits, filename)
        
        return data_symbols, pilot_symbols, bits



    def sync_error_pilot(self, signal, start, offset, bits_tran, known_frame, fileout='decode.txt',LDPC=False):

        # Get the average received frames by taking average of the
        # repetition of the transmitted knwon OFDM symbols
        avg_frame, start_refined = self.process_transmission(
            self, signal, start, offset)

        # Compute the channel responses
        freq_response, imp_response = self.estimate_channel_response(
            self, avg_frame, known_frame)

        information = self.retrieve_info(self, signal, start_refined)

        score = impulse_score(imp_response)

        if LDPC:
            bits_rec = OFDMframes_to_bitstring_via_LDPC(
                information, self.N, self.prefix_no, freq_response,inputLenIndicator_len=self.inputLenIndicator_len, N=self.N,rate=self.rate,r=self.r,z=self.z,len_protection=self.len_protection,repeat_times=self.repeat_times)
        else:
            bits_rec = OFDMframes_to_bitstring(
                information, self.N, self.prefix_no, freq_response)

        if fileout:
            bitstr_to_file(bits_rec, fileout)

        error = error_rate(bits_tran, bits_rec)
        return error, score



    def fine_tuning_pilot(self, signal, start, known_frame, carrier_indices, pilot_values, underfill, data_frames_len=None, find_range=10, offset=20, filename=None, LDPC=False):
        
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

            # Compute the averaged known frame after transmission
            _, avg_frame, _ = self.process_transmission_pilot(self, signal, start, new_offset)

            # Compute the channel responses
            _, imp_response = self.estimate_channel_response_pilot(self, avg_frame, known_frame)

            # Compute the score of the impulse 
            score = impulse_score(imp_response)
            # Append the score into the list 
            score_list.append(score)
            #print("index and score are:", i, score)

        # <----- Use the best score calculated to do the computation again ----->
        # Find the best score
        best_score_index = np.argmax(score_list)
        # Record the best score
        best_score = np.max(score_list)
        # Find the best offset value 
        best_offset = offset_list[best_score_index]
        # Refine the starting point with the best offset
        start_refined = start - best_offset
        
        # Redo the channel measurements with the best val
        _, avg_frame, _ = self.process_transmission_pilot(self, signal, start, best_offset)

        # Compute the channel responses
        best_freq_response, best_imp_response = self.estimate_channel_response_pilot(
                self, avg_frame, known_frame)
        
        data_begin = start_refined + len(known_frame)* self.repeat + self.gap # include gap

        if data_frames_len:
            rx_data_full = signal[data_begin:data_begin + data_frames_len]
        else:
            rx_data_full = signal[data_begin:-1]
        rx_data_frames = np.split(rx_data_full, data_frames_len / (self.N + self.prefix_no))

        if LDPC:
            _, _, bits_rec = self.data_remove_random_pilots_correct_phase_LDPC(self, rx_data_frames, carrier_indices, best_freq_response, pilot_values, underfill)

        else:
            _, _, bits_rec = self.data_remove_random_pilots_correct_phase(self, rx_data_frames, carrier_indices, best_freq_response, pilot_values, underfill)
        
        if filename:
            bitstr_to_file(bits_rec, filename)

        return best_offset, best_score, bits_rec, best_imp_response
