from os import error
from QAM_EncoderDecoder import *
# from LDPC import *
from ldpc_jossy.py import ldpc
# from ldpc import *
from util import *

# TODO: when noise variance gets large, the file length decoded can get wrong (and decoder crashes)


def divide_codebits(input__bits,decode=False,N=1024,rate='1/2',r=0.5,z=27):

    """
     Args:
         input__bits: array of numbers. can be 1s and 0s, and also decimals (the y received)
         decode: a bool
         N: needed for decoding. This is the OFDM block length
         rate: a str ('1/2' or '2/3' or '3/4' or '5/6')
         r: a float of rate
         z: an int (27 or 54 or 81)
         inputLenIndicator_len: the length in front of info bit that indicates the length og info bits

     Returns:
         output_bits_frange: array: [[[bits0],[]], [[bits1],[]], ...]
         
         -- note 'bits' can also be ys
    """
    input_bits=input__bits

    
    if decode:
        k=z*24
    else:
        k=r*z*24
    
    output_bits_frange=[]
    
    if decode:
        while len(input_bits)>0:
            output_bits_frange.append([input_bits[:int(k)],[ ]])
            
            input_bits=input_bits[int(k):]
            
    else:
        this_block_remain = k - len(input_bits)
        while this_block_remain <= 0:
            output_bits_frange.append([input_bits[:int(k)],[]])
            input_bits=input_bits[int(k):]
            this_block_remain = k - len(input_bits)
        input_bits=input_bits+'0'*int(this_block_remain) #padding
        assert len(input_bits)==k
        output_bits_frange.append([input_bits,rate,z])
    
    return output_bits_frange


def LDPC_encode(bits,inputLenIndicator_len=24, inputGuard_len=8,N=1024,rate='1/2',r=0.5,z=27,len_protection='input_repeat_then_LDPC',repeat_times=3,test=False):
    """
    bits: array of numbers. can be 1s and 0s, and also decimals (the y received)     
    len_protection (default:'no'): str. choices: 'no', 'input_repeat_then_LDPC', 'input_repeat_then_LDPC', 'guardBits'
    returns [LDPCstr_coded, list of rzs]
    """
    input_bit_length=len(bits)
    input_bit_length_bin=deci_to_binstr(input_bit_length,inputLenIndicator_len)
    
    


    if len_protection =='no':
        inputGuard_len=0
        add=''
        for i in input_bit_length_bin:
            if i=='0':
                add+='0'
            elif i=='1':
                add+='1'
            else:
                raise ValueError
        bits_with_indicator_and_guard=add+bits
    elif len_protection =='guardBits':
        add=''
        for i in input_bit_length_bin:
            if i=='0':
                add+='0'
            elif i=='1':
                add+='1'
            else:
                raise ValueError
        bits_with_indicator_and_guard=add+'0'*inputGuard_len+bits
    elif len_protection =='input_repeat_then_LDPC':
        inputGuard_len=0
        inputLenIndicator_len*=repeat_times
        add=''
        for i in input_bit_length_bin:
            if i=='0':
                add+='0'*repeat_times
            elif i=='1':
                add+='1'*repeat_times
            else:
                raise ValueError
        bits_with_indicator_and_guard=add+bits
    # elif len_protection =='input_repeat9_then_LDPC':
    #     inputGuard_len=0
    #     inputLenIndicator_len*=9
    #     add=''
    #     for i in input_bit_length_bin:
    #         if i=='0':
    #             add+='000000000'
    #         elif i=='1':
    #             add+='111111111'
    #         else:
    #             raise ValueError
    #     bits_with_indicator_and_guard=add+bits
    else:
        raise ValueError("input for len_protection incorrect")



    assert len(bits_with_indicator_and_guard)==input_bit_length+inputLenIndicator_len+inputGuard_len
    bits_r_zs=divide_codebits(bits_with_indicator_and_guard,decode=False,N=N,rate=rate,r=r,z=z)

    # LDPC_coded=[]
    LDPCstr_coded ='' 
    rzs=[]
    for i in range(len(bits_r_zs)):
        bits_r_z=bits_r_zs[i]
        print("\r",end="")
        print("encoding {0}th LDPC block, {1} in total".format(i,len(bits_r_zs)),end="")
        sys.stdout.flush()
        bits=bitstr_to_np_array(bits_r_z[0])
        # r=bits_r_z[1]
        # z=bits_r_z[2]
        if r!=0:
            
            LDPC_coder = ldpc.code(standard = '802.11n', rate = rate, z=z, ptype='A')

            coded=LDPC_coder.encode(bits)
            str_coded = ''
            for i in coded:
                str_coded+=str(i)
            coded=str_coded
            
        else: # use 
            raise ValueError("should not come here")
            coded=str(bits[0])*3
            # print("coded:",coded)
        LDPCstr_coded+=coded
        rzs.append([r,z])
        # LDPC_coded.append([coded,r,z])
        if test:
            return LDPCstr_coded

    a=LDPCstr_coded[:int(inputLenIndicator_len)]
    return LDPCstr_coded



def llr(ys,ck):
    # TODO: channel estimate coeff
    """returns llr of ys.
    Var is the noise variance of the awgn channel
    """
    return (2.0**0.5)/(ck*np.conj(ck))*ys

def LDPC_decode(ys_,N,rate='1/2',r=0.5,z=27,inputLenIndicator_len=24, inputGuard_len=8,cks=[],len_protection='input_repeat_then_LDPC',OnlyTestLen=False,FileLengthKnown=0,repeat_times=3):
    """[summary]

    Args:
        ys : ys received from OFDM decoding. Array of floats
        var : var of awgn channel noise
        N : block length of OFDM
        len_protection (default:'no'): str. choices: 'no', 'input_repeat3_then_LDPC', 'input_repeat9_then_LDPC', 'guardBits'

    Returns:
        LDPCstr_decoded
    """
    ys=ys_
    ys_franges=divide_codebits(ys,decode=True,N=N,rate=rate,r=r,z=z)

    encoded_block_length_k=z*24

    # LDPC_coded=[]
    LDPCstr_decoded ='' 
    decoded_length_count=0

    N_count=1 #frequency bins 0 and 512(int(N/2)) contains value 0
    N_upperbound=int(N/2)-1

    assert len(cks)>=N_upperbound

    for i in range(len(ys_franges)):
        ys_frange=ys_franges[i]
        print("\r",end="")
        print("decoding {0}th LDPC block, {1} in total".format(i,len(ys_franges)),end="")
        sys.stdout.flush()
        ys=np.array(ys_frange[0])

        LDPC_coder = ldpc.code(standard = '802.11n', rate = rate, z=z, ptype='A')
        
        llrs=[]

                
        for j in range (0,len(ys),2):
            ys1=ys[j]
            ys2=ys[j+1]
            
            llrs.append(llr(ys1,cks[N_count]))
            llrs.append(llr(ys2,cks[N_count]))

            N_count+=1
            if N_count>N_upperbound:
                N_count-=N_upperbound
        
        # llrs = llr(ys,var) 
        llrs=np.array(llrs)
        more_indicator_len=0

        if i==0:
            if len_protection=='guardBits':
                more_indicator_len=inputGuard_len

                # TODO: how can we make sure which llrs are certain?
                # we are certain about these llrs (certain that these codes are 0) (due to zero padding in inputGuard_len)
                # llrs[inputLenIndicator_len:inputLenIndicator_len+inputGuard_len]=[positive_infnity]*inputGuard_len
                
                (app,nit)= LDPC_coder.decode(llrs)
                transmitted=(app<0.0) # transmitted is np.array of Trues and Falses # this is the LDPC encoded bits before awgn transmission
                decoded=transmitted[:int(len(transmitted)/2)]
                str_decoded = ''
                for i in decoded:
                    str_decoded+=str(int(i))
                decoded=str_decoded[int(inputLenIndicator_len):]
                len_=str_decoded[:int(inputLenIndicator_len)]
                total_length= binstr_to_deci(len_ )/r

            elif len_protection=='no':
                
                (app,nit)= LDPC_coder.decode(llrs)
                transmitted=(app<0.0) # transmitted is np.array of Trues and Falses # this is the LDPC encoded bits before awgn transmission
                decoded=transmitted[:int(len(transmitted)/2)]
                str_decoded = ''
                for i in decoded:
                    str_decoded+=str(int(i))
                decoded=str_decoded[int(inputLenIndicator_len):]
                total_length= binstr_to_deci( str_decoded[:int(inputLenIndicator_len)])/r


            elif len_protection=='input_repeat_then_LDPC':
                inputLenIndicator_len*=repeat_times
                (app,nit)= LDPC_coder.decode(llrs)
                transmitted=(app<0.0) # transmitted is np.array of Trues and Falses # this is the LDPC encoded bits before awgn transmission
                decoded=transmitted[:int(len(transmitted)/2)]
                str_decoded = ''
                for i in decoded:
                    str_decoded+=str(int(i))
                decoded=str_decoded[int(inputLenIndicator_len):]
                length_bin = str_decoded[:int(inputLenIndicator_len)]
                length_bin = repetitive_decode_str2str(length_bin,repeat_times)
                total_length= binstr_to_deci(length_bin)/r


            # elif len_protection=='input_repeat9_then_LDPC':
            #     inputLenIndicator_len*=9
            #     (app,nit)= LDPC_coder.decode(llrs)
            #     transmitted=(app<0.0) # transmitted is np.array of Trues and Falses # this is the LDPC encoded bits before awgn transmission
            #     decoded=transmitted[:int(len(transmitted)/2)]
            #     str_decoded = ''
            #     for i in decoded:
            #         str_decoded+=str(int(i))
            #     decoded=str_decoded[int(inputLenIndicator_len):]
            #     length_bin = str_decoded[:int(inputLenIndicator_len)]
            #     length_bin = repetitive_decode_str2str(length_bin,9)
            #     total_length= binstr_to_deci(length_bin)/r

            else:
                raise ValueError("param len_protection wrong")



            
            total_length=int(total_length)
            print("\ntotal_length: ",total_length)
            if OnlyTestLen:
                return total_length/2==FileLengthKnown
            decoded_length_count+=int(encoded_block_length_k - inputLenIndicator_len/r - more_indicator_len/r)
 

        elif decoded_length_count+encoded_block_length_k<total_length:

            if i==len(ys_franges)-1:
                raise ValueError("last block not detected")

             # TODO: check
            (app,nit)= LDPC_coder.decode(llrs)

            transmitted=(app<0.0) # transmitted is np.array of Trues and Falses
            # this is the LDPC encoded bits before awgn transmission


            decoded=transmitted[:int(len(transmitted)/2)]
            str_decoded = ''
            for i in decoded:
                str_decoded+=str(int(i))
            decoded=str_decoded

            decoded_length_count+=encoded_block_length_k

        else: #last block that contain information, doesn't have to be last OFDM block (OFDM has paddings as well)
            # if i!=len(ys_franges)-1:
            #     raise ValueError("not last block")
             # TODO: check
            # llrs=llrs[:total_length-decoded_length_count]
            # padding=np.array([positive_infnity]*(encoded_block_length_k-(total_length-decoded_length_count)))
            # llrs=np.concatenate([llrs,padding])

            assert len(llrs)==encoded_block_length_k
        
            (app,nit)= LDPC_coder.decode(llrs)

            transmitted=(app<0.0) # transmitted is np.array of Trues and Falses
            # this is the LDPC encoded bits before awgn transmission


            decoded=transmitted[:int(len(transmitted)/2)]
            str_decoded = ''
            for i in decoded:
                str_decoded+=str(int(i))
            decoded=str_decoded

            decoded_length_count+=encoded_block_length_k

            LDPCstr_decoded+=decoded


            return LDPCstr_decoded[:int(total_length*r)]
            
        LDPCstr_decoded+=decoded
    
    raise ValueError("should not execute to this line")
    return LDPCstr_decoded[:int(total_length*r)]
    # return LDPCstr_decoded
