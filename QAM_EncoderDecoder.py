from os import name
from bitarray import bitarray
import numpy as np
from util import *

# -------------Encoder----------------
def file_to_bitstr(filename):
    """[convert the binary content in the file to a string of 0s and 1s]

    Args:
        filename ([string])

    Returns:
        [string]: [string of 0s and 1s]
    """
    with open(filename,'rb') as f:
        file_bytes=f.read()
        
    bin_strings=''
    for byte in file_bytes:
        binary_string = '{0:08b}'.format(byte)
        bin_strings+=binary_string
    return bin_strings



def encode_bitstr2symbols(bits):
    print("about to do encode_bitstr2symbols encoding")

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
    print("encode_bitstr2symbols encoding finished")
    return symbols



#-------------OFDM Encoder----------------

def symbol_to_OFDMframes(symbols,N,prefix_no):
    """returns 2d array of iDFTs
    """
    print("about to do symbol_to_OFDMframes encoding")

    info_bins=int(N/2)-1 #e.g. 511
    OFDM_frames=[]

    # for each OFDM block
    for i in range (0,len(symbols),info_bins):
        OFDM_block=[0] #frequency bins 0 and 512(int(N/2)) contains value 0
        OFDM_block[1:]=symbols[i:i+info_bins]

        #add 0s to the end when data is not an integer factor of 512
        while len(OFDM_block)<=info_bins:
            OFDM_block.append(0)

        OFDM_block.append(0)#frequency bins 0 and 512(int(N/2)) contains value 0

        # reverse conjugate
        for j in range(len(OFDM_block)-2,0,-1): # count up or down
            OFDM_block.append(np.conj(OFDM_block[j]))
                
        #----iDFT----
        OFDM_frame=np.fft.ifft(OFDM_block, n=N)
        

        # ----add cyclic prefix----
        cyclic_prefix = OFDM_frame[N-prefix_no:N]
        
        OFDM_frame = np.append(cyclic_prefix, OFDM_frame, axis=0)        
        OFDM_frames.append(OFDM_frame)

    print("symbol_to_OFDMframes encoding finished")
    return OFDM_frames



#-------------OFDM Decoder----------------
def OFDMframes_to_bitstring(OFDM_frames,N,prefix_no,channel_fft=False):
    print("about to do OFDMframes_to_bitstring decoding")
    bits=""
    for i in range(len(OFDM_frames)):
        frame_prefix = OFDM_frames[i][prefix_no:] # remove cp
        frame_dft = np.fft.fft(frame_prefix, n=N) 
        if channel_fft:
            bits+=decode_symbols_2_bitstring(frame_dft[1:int(N/2)],channel_fft[1:int(N/2)])
        else:
            bits+=decode_symbols_2_bitstring(frame_dft[1:int(N/2)])
        # print("frame_dft[1:int(N/2)]", int(N/2))

    print("OFDMframes_to_bitstring decoding finished")
    return bits

def OFDMframes_to_y_float(OFDM_frames,N,prefix_no,):
    """inputs:
        OFDM_frames: 2d np.array
    returns 1d array [[OFDMframe0_0_img, OFDMframe0_0_real, OFDMframe0_1_img,...],[OFDMframe1_0_img,...],...]
    """
    print("about to do OFDMframes_to_y_float decoding")
    ys=[]
    for i in range(len(OFDM_frames)):
        frame_prefix = OFDM_frames[i][prefix_no:] # remove cp
        frame_dft = np.fft.fft(frame_prefix, n=N) 
        ys+=separate_real_img(frame_dft[1:int(N/2)])
        # decode_symbols_2_bitstring(frame_dft[1:int(N/2)])
        # print("frame_dft[1:int(N/2)]", int(N/2))

    print("OFDMframes_to_y_float decoding finished")
    return ys

def OFDMframes_to_constellation(OFDM_frames,N,prefix_no,channel_fft=False):
    #print("about to do OFDMframes_to_bitstring decoding")
    bits=""
    for i in range(len(OFDM_frames)):
        frame_prefix = OFDM_frames[i][prefix_no:] # remove cp
        frame_dft = np.fft.fft(frame_prefix, n=N) 
    #print("OFDMframes_to_bitstring decoding finished")
    return frame_dft



# -------------Decoder----------------
def decode_symbols_2_bitstring(symbols,channel_fft=False):
    data = ''
    for i in range(len(symbols)):
        if channel_fft:
            element=symbols[i]/ channel_fft[i]
        else:
            element=symbols[i]
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
    return data


def bitstr_to_file(bin_strings,filename,cut=0):
    """
    Args:
        bin_strings ([str]): 
        filename ([str]): 
        cut (int, optional): [the length of information that you want to cut out at the start]. Defaults to 0.
    """

    data_bytes = bitarray(bin_strings)

    with open(filename, 'wb') as f:
        f.write(data_bytes.tobytes()[cut:])
    print("bitstr written to ",filename)


