from QAM_EncoderDecoder import *
# from LDPC import *
from ldpc_jossy.py import ldpc
# from ldpc import *
from util import *


def divide_codebits(input_bits,decode=False):
    """
    Args:
        input_bits: array of numbers. can be 1s and 0s, and also decimals (the y received)

    Returns:
        output_bits_r_z: array: [[[bits0],r0,z0], [[bits1],r1,z1], ...] 
        -- note 'bits' can also be ys
    """
    avaliable_ks=[]
    for r in [['1/2',1/2],['2/3',2/3],['3/4',3/4], ['5/6',5/6]]:
        for z in [27,54,81]:
            avaliable_ks.append([r[1]*z*24,r[1],r[0],z])
            # if decode:
            #     avaliable_ks.append([z*24,r[1],r[0],z]) #TODO
            # else:
            #     avaliable_ks.append([r[1]*z*24,r[1],r[0],z])
    
    dist=divide_bin_to_sizes(len(input_bits),avaliable_ks)

    output_bits_r_z=[]
    for key in dist:
        
        n,r,r_str,z=dist[key]
        # if key!=1:
        #     print("hi")
        # n=int(n)
        # n=int(n*key)
        # print(n)
        for i in range(n):
            if decode:
                if key==648:
                    print('hi')
                if r!=0:
                    encoded_codelength=key/r
                    cooresponding_bits=input_bits[:int(encoded_codelength)]
                else:
                    cooresponding_bits=input_bits[:int(key)]
            else:
                cooresponding_bits=input_bits[:int(key)]
            input_bits=input_bits[int(key):]
            output_bits_r_z.append([cooresponding_bits,r_str,z])
    return output_bits_r_z


def LDPC_encode(bits):
    """returns [LDPCstr_coded, list of rzs]
    """
    bits_r_zs=divide_codebits(bits)

    # LDPC_coded=[]
    LDPCstr_coded ='' 
    rzs=[]
    for i in range(len(bits_r_zs)):
        bits_r_z=bits_r_zs[i]
        print("{0}th LDPC block, {1} in total".format(i,len(bits_r_zs)))
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
            print('u:',bits)
            coded=LDPC_coder.encode(bits)
            str_coded = ''
            for i in coded:
                str_coded+=str(i)
            coded=str_coded
            
        else: # use Hanning code
            coded=str(bits[0]*3)
            print("coded:",coded)
        LDPCstr_coded+=coded
        rzs.append([r,z])
        # LDPC_coded.append([coded,r,z])

    return LDPCstr_coded



def llr(ys,var):
    """returns llr of ys.
    Var is the noise variance of the awgn channel
    """
    return 2.0/var*ys

def LDPC_decode(ys,var):
    ys_r_zs=divide_codebits(ys,True)

    # LDPC_coded=[]
    LDPCstr_decoded ='' 
    for i in range(1,len(ys_r_zs)):
        ys_r_z=ys_r_zs[i]
        print("decoding {0}th LDPC block, {1} in total".format(i,len(ys_r_zs)))
        ys=np.array(ys_r_z[0])
        r=ys_r_z[1]
        z=ys_r_z[2]
        if r!=0:
            LDPC_coder = ldpc.code(standard = '802.11n', rate = r, z=z, ptype='A')
            print("about to decode"); print('K:',LDPC_coder.K)
            print('rate:',r); print('z:',z)
            print('standard:','802.11n'); print('ptype:','A')
            print('ys:',ys); print('len(ys):',len(ys))
            llrs = llr(ys,var) # TODO: check
            (app,nit)= LDPC_coder.decode(llrs)

            decoded=(app<0.0) # decoded is np.array of Trues and Falses

            str_decoded = ''
            for i in decoded:
                str_decoded+=str(int(i))
            decoded=str_decoded
            
        else: # use Hanning code
            # decoded = 
            decoded=str(ys)*3
        LDPCstr_decoded+=decoded

    return LDPCstr_decoded


# if __name__=='__main__':

    # bits_r_zs=divide_codebits(bits)

    # LDPC_coded=[]
    
    # for bits_r_z in bits_r_zs:
    #     bits=bitstr_to_np_array(bits_r_z[0])
    #     r=bits_r_z[1]
    #     z=bits_r_z[2]
    #     # print("bits:",bits)
    #     # print('r:',type(r))
    #     # print('z:',z)
    #     if r!=0:
    #         print('about to make coder with rate={0} and z={1}, K=r*z*24'.format(r,z))
    #         LDPC_coder = ldpc.code(standard = '802.11n', rate = r, z=z, ptype='A')
    #         print('about to encode, bits length=',len(bits))
            
    #         print("about to encode")
    #         print('K:',LDPC_coder.K)
    #         print('rate:',r)
    #         print('z:',z)
    #         print('standard:','802.11n')
    #         print('ptype:','A')
    #         print('u:',bits)
    #         coded=LDPC_coder.encode(bits)
    #         a=1
            
    #     else: # use Hanning code
    #         coded=str(bits)*3
            
    #     LDPC_coded.append(coded)
    #     print("coded :",coded)