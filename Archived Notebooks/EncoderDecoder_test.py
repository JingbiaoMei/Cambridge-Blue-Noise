import numpy as np
from QAM_EncoderDecoder import *
from ldpc_jossy.py import ldpc
from LDPC import *
from util import *
from sys import stdout



def do_it(N=2048,prefix_no=32,snr=1.5,ReturnError=False,len_protection='input_repeat_then_LDPC',inputLenIndicator_len=32,inputGuard_len=8,rate='1/2',r=0.5,z=81,filename='input.png',OnlyTestLen=True,repeat_times=5,cut=2000):

    print("\n----------------- New Trial ----------------- \nlen_protection is: ",len_protection)

    
    bits=file_to_bitstr(filename)[:cut]
    print('input file length:',len(bits))


    # ------ LDPC encode ------
    LDPC_encoded_bits=LDPC_encode(bits,inputLenIndicator_len=inputLenIndicator_len,inputGuard_len=inputGuard_len,N=N,rate=rate,r=r,z=z,len_protection=len_protection,repeat_times=repeat_times)#,test=True)
    print('\nlen(LDPC_encoded_bits)',len(LDPC_encoded_bits))
    
    # ------ QPSK encode ------
    symbols=encode_bitstr2symbols(LDPC_encoded_bits)
    print(np.shape(symbols))

    #--- OFDM encoding ---
    
    OFDM_frames=symbol_to_OFDMframes(symbols,N,prefix_no)
    print(np.shape(OFDM_frames))
    # bin_strings=OFDMframes_to_bitstring(OFDM_frames,N,prefix_no)

    # --- Channel ---
    data_transmitted=OFDM_frames
    print("len(data_transmitted): ",len(data_transmitted))
    input_P=abs(np.var(data_transmitted)+np.mean(data_transmitted)**2)
    print("input average power: ",input_P)
    
    
    snr=1.5

    print("\nsnr:", snr)

    var=input_P/snr
    # var=0
    print("var:", var)

    data_received=[]
    for OFDM_block in OFDM_frames:
        data_received.append(awgn(OFDM_block,var))
    print("len(data_received): ",len(data_received))
    
    
    #--- change into 1d array of ys---
    ys_=OFDMframes_to_y_float(data_received,N,prefix_no)

    #-- LDPC decoding --

    cks=np.array([var**0.5]*len(ys_))

    print('len(ys_):', len(ys_)) 
    assert len(ys_)>=len(data_transmitted)
    LDPC_decoded_bits, file_type= LDPC_decode_with_niceCKs(ys_,N,cks=cks,ysReal=True) #LDPC_coder.encode(bin_strings, dectype='sumprod2', corr_factor=0.7)
    # len(LDPC_decoded_bits)=3296 compared to len(original bits)=2000.
    # len(LDPC_encoded_bits)=4056

    print('\noutput file length:',len(LDPC_decoded_bits))


    print('\ninput file length:',len(bits))
    print('\decoded file length:',len(LDPC_decoded_bits))
    assert len(LDPC_decoded_bits)==len(bits)

    biterrors = np.sum(bitstr_to_np_array(LDPC_decoded_bits) != bitstr_to_np_array(bits) )
    print("\nbiterrors:",biterrors)
    errorratio=np.log10(biterrors/len(bits))
    print("error ratio = 10^",errorratio)
    print("\n")

    if ReturnError:
        return errorratio

    bitstr_to_file(LDPC_decoded_bits,'OFDM_output.png')
    
    print("finished one.")

if __name__=='__main__':
    filename='test_files/shakespeare_small.txt'
    a=do_it(filename=filename)