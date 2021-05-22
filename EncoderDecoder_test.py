from QAM_EncoderDecoder import *
# from LDPC import *
from ldpc_jossy.py import ldpc
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
    bits_r_zs=divide_codebits(bits)

    LDPC_coded=[]
    
    for bits_r_z in bits_r_zs:
        bits=bitstr_to_np_array(bits_r_z[0])
        r=bits_r_z[1]
        z=bits_r_z[2]
        # print("bits:",bits)
        # print('r:',type(r))
        # print('z:',z)
        if r!=0:
            print('about to make coder with rate={0} and z={1}, K=r*z*24'.format(r,z))
            LDPC_coder = ldpc.code(standard = '802.11n', rate = r, z=z, ptype='A')
            print('about to encode, bits length=',len(bits))
            
            print("about to encode")
            print('K:',LDPC_coder.K)
            print('rate:',r)
            print('z:',z)
            print('standard:','802.11n')
            print('ptype:','A')
            print('u:',bits)
            coded=LDPC_coder.encode(bits)
            a=1
            
        else: # use Hanning code
            coded=bits+bits+bits
            
        LDPC_coded.append(coded)
        print("coded :",coded)

    # # LDPC encoding
    # r=1/2
    # z=27*2
    # LDPC_coder = ldpc.code(standard = '802.11n', rate = '1/2', z=z, ptype='A') # parameters are the default ones
    
    # avaliable_ks=[]
    # for r in [1/2,2/3,3/4, 5/6]:
    #     for z in [27,54,81]:
    #         avaliable_ks.append(r*z*24)
    # print(avaliable_ks)

    # d
    # print(LDPC_coder.K/z/r)
    # print(len(bits)/LDPC_coder.K)
    # print(r*z*24==LDPC_coder.K)
    
    # LDPC_encoded_bits= LDPC_coder.encode(bits)


    # symbols=encode_bitstr2symbols(LDPC_encoded_bits)
    # print(np.shape(symbols))

    # #---OFDM encoding and decoding---
    # N = 1024
    # prefix_no = 32
    # OFDM_frames=symbol_to_OFDMframes(symbols,N,prefix_no)
    # print(np.shape(OFDM_frames))
    # bin_strings=OFDMframes_to_bitstring(OFDM_frames,N,prefix_no)
    # print(len(bin_strings)/2)
    # # bin_strings=decode_symbols_2_bitstring(symbols)

    # #LDPC decoding
    # LDPC_decoded_bits= LDPC_coder.encode(bin_strings, dectype='sumprod2', corr_factor=0.7)


    # bitstr_to_file(LDPC_decoded_bits,'OFDM_output.png')