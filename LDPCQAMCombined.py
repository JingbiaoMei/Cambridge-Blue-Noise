from QAM_EncoderDecoder import *
from LDPC import *

def encode_bitstr2symbols_via_LDPC(bits,inputLenIndicator_len=24, inputGuard_len=8,N=2048,rate='1/2',r=0.5,z=27,len_protection='input_repeat_then_LDPC',repeat_times=3,test=False):
    #print("about to do encode_bitstr2symbols encoding")
    LDPC_encoded_bits=LDPC_encode(bits,inputLenIndicator_len=inputLenIndicator_len, inputGuard_len=inputGuard_len,N=N,rate=rate,r=r,z=z,len_protection=len_protection,repeat_times=repeat_times,test=test)
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
    #print("encode_bitstr2symbols encoding finished")
    return symbols

def OFDMframes_to_bitstring_via_LDPC(OFDM_frames,N,prefix_no,channel_fft,rate='1/2',r=0.5,z=27,inputLenIndicator_len=24, inputGuard_len=8,len_protection='input_repeat_then_LDPC',OnlyTestLen=False,FileLengthKnown=0,repeat_times=3):
    print("inside OFDMframes_to_bitstring_via_LDPC")
    cks=channel_fft
    ys_=OFDMframes_to_y_float(OFDM_frames,N,prefix_no)

    #-- LDPC decoding --

    # cks=[var**0.5]*int(N/2)

    LDPC_decoded= LDPC_decode(ys_,N,rate=rate,r=r,z=z,inputLenIndicator_len=inputLenIndicator_len, inputGuard_len=inputGuard_len,cks=cks,len_protection=len_protection,OnlyTestLen=OnlyTestLen,FileLengthKnown=FileLengthKnown,repeat_times=repeat_times)

    return LDPC_decoded
    #print("about to do OFDMframes_to_bitstring decoding")
    bits=""
    for i in range(len(OFDM_frames)):
        frame_prefix = OFDM_frames[i][prefix_no:] # remove cp
        frame_dft = np.fft.fft(frame_prefix, n=N) 
        # For array use .any()
        if channel_fft.any():
            bits+=decode_symbols_2_bitstring(frame_dft[1:int(N/2)],channel_fft[1:int(N/2)])
        else:
            bits+=decode_symbols_2_bitstring(frame_dft[1:int(N/2)])
        # print("frame_dft[1:int(N/2)]", int(N/2))

    #print("OFDMframes_to_bitstring decoding finished")
    return bits

