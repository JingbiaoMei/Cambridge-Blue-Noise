from QAM_EncoderDecoder import *
# from LDPC import *
from ldpc_jossy.py import ldpc
# from ldpc import *
from util import *


def divide_codebits(input_bits):
    avaliable_ks=[]
    for r in [['1/2',1/2],['2/3',2/3],['3/4',3/4], ['5/6',5/6]]:
        for z in [27,54,81]:
            avaliable_ks.append([r[1]*z*24,r[0],z])
    dist=divide_bin_to_sizes(len(input_bits),avaliable_ks)

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


def LDPC_encode(bits):
    bits_r_zs=divide_codebits(bits)

    # LDPC_coded=[]
    LDPCstr_coded ='' 
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
            # print('u:',bits)
            coded=LDPC_coder.encode(bits)
            str_coded = ''
            for i in coded:
                str_coded+=str(i)
            coded=str_coded
            
        else: # use Hanning code
            coded=str(bits)*3
        LDPCstr_coded+=coded
        # LDPC_coded.append([coded,r,z])

    return LDPCstr_coded

if __name__=='__main__':

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
            coded=str(bits)*3
            
        LDPC_coded.append(coded)
        print("coded :",coded)

    