from QAM_EncoderDecoder import *
# from LDPC import *
from ldpc_jossy.py import ldpc
from LDPC import *
from util import *
from sys import stdout


# def divide_codebits(input_bits):
#     avaliable_ks=[]
#     for r in [['1/2',1/2],['2/3',2/3],['3/4',3/4], ['5/6',5/6]]:
#         for z in [27,54,81]:
#             avaliable_ks.append([r[1]*z*24,r[0],z])
#     dist=divide_bin_to_sizes(len(bits),avaliable_ks)

#     output_bits_r_z=[]
#     for key in dist:
#         n,r,z=dist[key]
#         # n=int(n)
#         # n=int(n*key)
#         print(n)
#         for i in range(n):
#             cooresponding_bits=input_bits[:int(key)]
#             input_bits=input_bits[int(key):]
#             output_bits_r_z.append([cooresponding_bits,r,z])
#     return output_bits_r_z


if __name__=='__main__':

    filename='input_small.png'
    bits=file_to_bitstr(filename) #[:2000]
    print('input file length:',len(bits))


    # ------ LDPC encode ------
    LDPC_encoded_bits=LDPC_encode(bits)
    print('\nlen(LDPC_encoded_bits)',len(LDPC_encoded_bits))
    
    # ------ QPSK encode ------
    symbols=encode_bitstr2symbols(LDPC_encoded_bits)
    print(np.shape(symbols))

    #--- OFDM encoding ---
    N = 1024
    prefix_no = 32
    OFDM_frames=symbol_to_OFDMframes(symbols,N,prefix_no)
    print(np.shape(OFDM_frames))
    # bin_strings=OFDMframes_to_bitstring(OFDM_frames,N,prefix_no)

    # --- Channel ---
    data_transmitted=OFDM_frames
    print("len(data_transmitted): ",len(data_transmitted))
    # example of data_transmitted values: 0.063, -0.069, 0.04, 0.004, 0.023, 0.0064
    var=0.001

    data_received=[]
    for OFDM_block in OFDM_frames:
        data_received.append(awgn(OFDM_block,var))
    print("len(data_received): ",len(data_received))

    #--- change into 1d array of ys---
    ys_=OFDMframes_to_y_float(data_received,N,prefix_no)

    #-- LDPC decoding --

    LDPC_decoded_bits= LDPC_decode(ys_,var,N) #LDPC_coder.encode(bin_strings, dectype='sumprod2', corr_factor=0.7)
    # len(LDPC_decoded_bits)=3296 compared to len(original bits)=2000.
    # len(LDPC_encoded_bits)=4056

    print('\noutput file length:',len(LDPC_decoded_bits))
    bitstr_to_file(LDPC_decoded_bits,'OFDM_output.png')

    print("hi")