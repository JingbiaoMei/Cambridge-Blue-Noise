from os import error
from QAM_EncoderDecoder import *
# from LDPC import *
from ldpc_jossy.py import ldpc
# from ldpc import *
from util import *


def divide_codebits(input__bits,decode=False,N=1024,rate='1/2',r=0.5,z=27):

    """
     Args:
         input__bits: array of numbers. can be 1s and 0s, and also decimals (the y received)
         decode: a bool
         N: needed for decoding. This is the OFDM block length
         rate: a str ('1/2' or '2/3' or '3/4' or '5/6')
         r: a float of rate
         z: an int (27 or 54 or 81)

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
        for j in range(len(input_bits)-1,-1,-1):
            if abs(input_bits[j])>0.00001: #deal with zero padding
                last_bit=j
                print("last_bit:",last_bit)
                break
        
        while len(input_bits)>=k and len(input_bits)>=last_bit+1:
            output_bits_r_z.append([input_bits[:int(k)],rate,z])
            input_bits=input_bits[int(k):]
            last_bit-=k
        for i in range(0,last_bit,3):
            ap=[input_bits[i],input_bits[i+1],input_bits[i+2]]
            output_bits_r_z.append([ap,0,0])
        
    else:
        while len(input_bits)>=k:
            output_bits_r_z.append([input_bits[:int(k)],rate,z])
            input_bits=input_bits[int(k):]
        for i in input_bits:
            output_bits_r_z.append([i,0,0])
    
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


def LDPC_encode(bits):
    """
    bits: array of numbers. can be 1s and 0s, and also decimals (the y received)     
    returns [LDPCstr_coded, list of rzs]
    """
    bits_r_zs=divide_codebits(bits)

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
            # print('about to encode, bits length=',len(bits))
            
            # print("about to encode")
            # print('K:',LDPC_coder.K)
            # print('rate:',r)
            # print('z:',z)
            # print('standard:','802.11n')
            # print('ptype:','A')
            # print('u:',bits)
            coded=LDPC_coder.encode(bits)
            str_coded = ''
            for i in coded:
                str_coded+=str(i)
            coded=str_coded
            
        else: # use Hanning code
            coded=str(bits[0])*3
            # print("coded:",coded)
        LDPCstr_coded+=coded
        rzs.append([r,z])
        # LDPC_coded.append([coded,r,z])

    return LDPCstr_coded



def llr(ys,var):
    """returns llr of ys.
    Var is the noise variance of the awgn channel
    """
    return (2.0**0.5)/var*ys

def LDPC_decode(ys,var,N):
    """[summary]

    Args:
        ys : ys received from OFDM decoding. Array of floats
        var : var of awgn channel noise
        N : block length of OFDM

    Returns:
        LDPCstr_decoded
    """
    ys_r_zs=divide_codebits(ys,True,N)

    # LDPC_coded=[]
    LDPCstr_decoded ='' 
    for i in range(len(ys_r_zs)):
        ys_r_z=ys_r_zs[i]
        print("\r",end="")
        print("decoding {0}th LDPC block, {1} in total".format(i,len(ys_r_zs)),end="")
        sys.stdout.flush()
        ys=np.array(ys_r_z[0])
        r=ys_r_z[1]
        z=ys_r_z[2]
        if r!=0:
            LDPC_coder = ldpc.code(standard = '802.11n', rate = r, z=z, ptype='A')
            # print("about to decode"); print('K:',LDPC_coder.K)
            # print('rate:',r); print('z:',z)
            # print('standard:','802.11n'); print('ptype:','A')
            # # print('ys:',ys); 
            # print('len(ys):',len(ys))
            llrs = llr(ys,var) # TODO: check
            (app,nit)= LDPC_coder.decode(llrs)

            transmitted=(app<0.0) # transmitted is np.array of Trues and Falses
            # this is the LDPC encoded bits before awgn transmission


            decoded=transmitted[:int(len(transmitted)/2)]
            str_decoded = ''
            for i in decoded:
                str_decoded+=str(int(i))
            decoded=str_decoded
            
        else: # use Hanning code
            # decoded = 
            zeros=0
            ones=0
            for i in ys:
                if i>0.5:
                    ones+=1
                else:
                    zeros+=1
            if ones>=2:
                decoded='1'
            elif zeros>=2:
                decoded='0'
            else:
                raise ValueError
        LDPCstr_decoded+=decoded

    return LDPCstr_decoded
