from QAM_EncoderDecoder import *
# from LDPC import *
from ldpc_jossy.py import ldpc
from LDPC import *
from util import *


def divide_codebits(input_bits):
    avaliable_ks=[]
    for r in [['1/2',1/2],['2/3',2/3],['3/4',3/4], ['5/6',5/6]]:
        for z in [27,54,81]:
            avaliable_ks.append([r[1]*z*24,r[0],z])
    dist=divide_bin_to_sizes(len(bits),avaliable_ks)

    output_bits_r_z=[]
    for key in dist:
        n,r,z=dist[key]
        # n=int(n)
        # n=int(n*key)
        print(n)
        for i in range(n):
            cooresponding_bits=input_bits[:int(key)]
            input_bits=input_bits[int(key):]
            output_bits_r_z.append([cooresponding_bits,r,z])
    return output_bits_r_z


if __name__=='__main__':

    filename='input.png'
    bits=file_to_bitstr(filename)
    print(len(bits))


    # ------ LDPC encode ------
    LDPC_encoded_bits=LDPC_encode(bits)
    print(LDPC_encoded_bits)
    
    # ------ QPSK encode ------
    symbols=encode_bitstr2symbols(LDPC_encoded_bits)
    print(np.shape(symbols))

    #---OFDM encoding and decoding---
    N = 1024
    prefix_no = 32
    OFDM_frames=symbol_to_OFDMframes(symbols,N,prefix_no)
    print(np.shape(OFDM_frames))
    bin_strings=OFDMframes_to_bitstring(OFDM_frames,N,prefix_no)
    print(len(bin_strings)/2)
    # bin_strings=decode_symbols_2_bitstring(symbols)

    #LDPC decoding
    LDPC_decoded_bits= LDPC_coder.encode(bin_strings, dectype='sumprod2', corr_factor=0.7)


    bitstr_to_file(LDPC_decoded_bits,'OFDM_output.png')