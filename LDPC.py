from os import error
from QAM_EncoderDecoder import *
# from LDPC import *
from ldpc_jossy.py import ldpc
# from ldpc import *
from util import *

# TODO: when noise variance gets large, the operation for removing zero padding gets buggy

zero_padding_thresh=0.2

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
         output_bits_r_z: array: [[[bits0],rate0,z0], [[bits1],rate1,z1], ...] 
         
         -- note 'bits' can also be ys
    """
    input_bits=input__bits

    
    if decode:
        k=z*24
    else:
        k=r*z*24
    
    output_bits_r_z=[]
    
    if decode:
        while len(input_bits)>0:
            output_bits_r_z.append([input_bits[:int(k)],rate,z])
            input_bits=input_bits[int(k):]
            
    else:
        this_block_remain = k - len(input_bits)
        while this_block_remain <= 0:
            output_bits_r_z.append([input_bits[:int(k)],rate,z])
            input_bits=input_bits[int(k):]
            this_block_remain = k - len(input_bits)
        input_bits=input_bits+'0'*int(this_block_remain)
        assert len(input_bits)==k
        output_bits_r_z.append([input_bits,rate,z])
    
    return output_bits_r_z

    # """
    # Args:
    #     input_bits: array of numbers. can be 1s and 0s, and also decimals (the y received)

    # Returns:
    #     output_bits_r_z: array: [[[bits0],r0,z0], [[bits1],r1,z1], ...] 
    #     -- note 'bits' can also be ys
    # """
    # avaliable_ks=[]
    # for r in [['1/2',1/2],['2/3',2/3],['3/4',3/4], ['5/6',5/6]]:
    #     for z in [27,54,81]:
    #         avaliable_ks.append([r[1]*z*24,r[1],r[0],z])
    #         # if decode:
    #         #     avaliable_ks.append([z*24,r[1],r[0],z]) #TODO
    #         # else:
    #         #     avaliable_ks.append([r[1]*z*24,r[1],r[0],z])
    
    # dist=divide_bin_to_sizes(len(input_bits),avaliable_ks)

    # output_bits_r_z=[]
    # for key in dist:
        
    #     n,r,r_str,z=dist[key]
    #     # if key!=1:
    #     #     print("hi")
    #     # n=int(n)
    #     # n=int(n*key)
    #     # print(n)
    #     for i in range(n):
    #         if decode:
    #             if key==648:
    #                 print('hi')
    #             if r!=0:
    #                 encoded_codelength=key/r
    #                 cooresponding_bits=input_bits[:int(encoded_codelength)]
    #             else:
    #                 cooresponding_bits=input_bits[:int(key)]
    #         else:
    #             cooresponding_bits=input_bits[:int(key)]
    #         input_bits=input_bits[int(key):]
    #         output_bits_r_z.append([cooresponding_bits,r_str,z])
    # return output_bits_r_z


def LDPC_encode(bits,inputLenIndicator_len=24,inputGuard_len=8,N=1024,rate='1/2',r=0.5,z=27):
    """
    bits: array of numbers. can be 1s and 0s, and also decimals (the y received)     
    returns [LDPCstr_coded, list of rzs]
    """
    input_bit_length=len(bits)
    input_bit_length_bin=deci_to_binstr(input_bit_length,inputLenIndicator_len)
    
    add=''
    for i in input_bit_length_bin:
        if i=='0':
            add+='0'
        elif i=='1':
            add+='1'
        else:
            raise ValueError
    bits_with_indicator=add+bits
    bits_with_indicator_and_guard=bits_with_indicator+'0'*inputGuard_len
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
        r=bits_r_z[1]
        z=bits_r_z[2]
        if r!=0:
            
            LDPC_coder = ldpc.code(standard = '802.11n', rate = r, z=z, ptype='A')

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

    return LDPCstr_coded



def llr(ys,var):
    # TODO: channel estimate coeff
    """returns llr of ys.
    Var is the noise variance of the awgn channel
    """
    return (2.0**0.5)/var*ys

def LDPC_decode(ys_,var,N,rate='1/2',r=0.5,z=27,inputLenIndicator_len=24,inputGuard_len=8):
    """[summary]

    Args:
        ys : ys received from OFDM decoding. Array of floats
        var : var of awgn channel noise
        N : block length of OFDM

    Returns:
        LDPCstr_decoded
    """
    ys=ys_
    ys_r_zs=divide_codebits(ys,True,N)

    encoded_block_length_k=z*24

    # LDPC_coded=[]
    LDPCstr_decoded ='' 
    decoded_length_count=0
    for i in range(len(ys_r_zs)):
        ys_r_z=ys_r_zs[i]
        print("\r",end="")
        print("decoding {0}th LDPC block, {1} in total".format(i,len(ys_r_zs)),end="")
        sys.stdout.flush()
        ys=np.array(ys_r_z[0])
        if i==0:
            LDPC_coder = ldpc.code(standard = '802.11n', rate = rate, z=z, ptype='A')
            llrs = llr(ys,var) 

            # TODO: infinity after matrix multi with G matrix
            # Ask Jossy?

            # llrs[inputLenIndicator_len:inputLenIndicator_len+inputGuard_len] = np.array([positive_infnity]*(inputGuard_len))
            # llrs[inputLenIndicator_len:inputGuard_len] = 
            (app,nit)= LDPC_coder.decode(llrs)

            transmitted=(app<0.0) # transmitted is np.array of Trues and Falses
            # this is the LDPC encoded bits before awgn transmission


            decoded=transmitted[:int(len(transmitted)/2)]
            str_decoded = ''
            for i in decoded:
                str_decoded+=str(int(i))
            decoded=str_decoded[int(inputLenIndicator_len):]
            total_length= binstr_to_deci( str_decoded[:int(inputLenIndicator_len)])/r
            
            total_length=int(total_length)
            print("\ntotal_length: ",total_length)
            decoded_length_count+=encoded_block_length_k - inputLenIndicator_len

        elif decoded_length_count+encoded_block_length_k<total_length:

            if i==len(ys_r_zs)-1:
                raise ValueError("last block not detected")

            LDPC_coder = ldpc.code(standard = '802.11n', rate = rate, z=z, ptype='A')

            llrs = llr(ys,var) # TODO: check
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
            # if i!=len(ys_r_zs)-1:
            #     raise ValueError("not last block")
            LDPC_coder = ldpc.code(standard = '802.11n', rate = rate, z=z, ptype='A')
            llrs = llr(ys,var) # TODO: check
            llrs=llrs[:total_length-decoded_length_count]
            padding=np.array([positive_infnity]*(encoded_block_length_k-(total_length-decoded_length_count)))
            llrs=np.concatenate([llrs,padding])

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
    # return LDPCstr_decoded
