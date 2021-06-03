import numpy as np
import sys
from bitarray import bitarray

positive_infnity = float('inf')

def array2str(array:list):
    st=''
    for i in array:
        st+=str(i)
    return st

def file_type_to_bitarray(file_type:str) ->np.array : 
    """file_type must be one of '.tif' or 'tif', '.txt' or 'txt', '.wav' or 'wav'
    """
    if file_type == '.tif' or file_type == 'tif' or file_type == '.tiff' or file_type == 'tiff':
        return np.array([0,0,1,1,0,0,1,0])
    elif file_type == '.txt' or file_type == 'txt':
        return np.array([0,1,0,0,1,0,0,0])
    elif file_type == '.wav' or file_type == 'wav':
        return np.array([1,0,1,0,0,1,0,1])
    else:
        raise KeyError("file_type incorrect")

def bitstr_to_file_type_str(bitstr:str) ->str : 
    assert len(bitstr) == 8 # for this standard
    tif=0; txt=0; wav=0
    if bitstr[0]=='1':
        wav+=1
        # tif-=1
        # txt-=1
    if bitstr[1]=='1':
        txt+=1
        # tif-=1
        # wav-=1
    if bitstr[2]=='0':
        txt+=1
        # tif-=1
        # wav-=1
    if bitstr[3]=='1':
        tif+=1
        # wav-=1
        # txt-=1
    if bitstr[4]=='1':
        txt+=1
        # tif-=1
        # wav-=1
    if bitstr[5]=='1':
        wav+=1
        # tif-=1
        # txt-=1
    if bitstr[6]=='1':
        tif+=1
        # tif-=1
        # wav-=1
    if bitstr[7]=='1':
        wav+=1
        # tif-=1
        # txt-=1
    print(" tif=",tif," txt=",txt," wav=",wav)
    if tif>txt:
        if tif>wav:
            return (".tif")
        elif tif<wav:
            return (".wav")
        else:
            print('tif=wav, returning tif')
            return(".tif")
    elif tif<txt:
        if txt>wav:
            return (".txt")
        elif txt<wav:
            return (".wav")
        else:
            # raise KeyError('txt=wav')
            print('txt=wav, returning txt')
            return(".txt")
    elif wav>txt:
        return (".wav")
    else:
        # raise KeyError('tif=txt=wav')
        print('txt=tif=wav, returning txt')
        return(".txt")
    

def bitstr_to_file(bin_strings,filename,cut=0):
    """
    Args:
        bin_strings ([str]): 
        filename ([str]): 
        cut (int, optional): [the length of information that you want to cut out at the start]. Defaults to 0.
    """
    print("\nabout to write file, the filename inputed is: ",filename)
    data_bytes = bitarray(bin_strings)

    with open(filename, 'wb') as f:
        f.write(data_bytes.tobytes()[cut:])
    print("bitstr written to ",filename)

def repetitive_decode_str2str(encodedbits,repeatTimes=5):
    decoded_bits=''
    divided_len=int(len(encodedbits)/repeatTimes)
    assert divided_len ==32 or divided_len ==8 #for our standard.
    
    for i in range(0,divided_len):
        zeros=0
        ones=0
        for j in range(repeatTimes):
            if encodedbits[i+j*divided_len]=='0':
                zeros+=1
            elif encodedbits[i+j*divided_len]=='1':
                ones+=1
            else:
                raise ValueError
        
        assert zeros+ones == repeatTimes

        if zeros>ones:
            decoded_bits+='0'
        elif zeros<ones:
            decoded_bits+='1'
        else:
            raise ValueError
        
        if zeros>0 and ones>0:
            print("------error corrected-------")

    return decoded_bits

def divide_bin_to_sizes(N:int,sizes:list):
    # TODO: improve this algorithm
    """
    CAN BE IMPROVED

    sizes: [[K0, r0, z0],[K1, r1, z1],...]
    
    return dictionary:
    {sizes[0]:[n0,r0,str(r0),z0], sizes[1]:[n1,r1,str(r1),z1], sizes[2]:n2, ..., 1:remainder}
    so that n0*sizes[0] + n1*sizes[1] +...+ 1*remainder = N

    Args:
        total_zies (int): [description]
        sizes (list): [description]
    """
    # print(sizes)
    sizes.sort(reverse=True)
    # print(sizes)
    rt={}
    for i in range(len(sizes)):
        # print("i:",i)
        for j in range(i,len(sizes)):
            size=sizes[j][0]
            # print("size:",size)
            n = int(N/size)
            if n == N/size:
                print('append ',n)
                rt[size] = [n,sizes[j][1],sizes[j][2],sizes[j][3]]
                N = N - n*size #==0
                # print("j:",j)
                # print('N should ==0: ',N)
                # print(N==0)
                return rt
        if N==0:
            return rt
        size=sizes[i][0]
        # print("size:",size)
        n = int(N/size)
        if n!=0:
            print('append ',n)
            rt[size] = [n,sizes[i][1],sizes[i][2],sizes[i][3]]
        N = N - n*size
        
        if N==0:
            return rt

    rt[1] = [int(N),0,0,0]
    print('append ',int(N))
    return rt
    print(rt)
    print(N)
    raise NameError('N cannot be separated by sizes')

def bitstr_to_np_array(bitstr):
    array=[]
    for bit in bitstr:
        array.append(int(bit))
    return np.array(array)

def separate_real_img(complex_inputs):
    """returns [input0.img,input0.real, input1.img, input1.real,  ...]
    """
    outputs=[]
    for i in complex_inputs:
        outputs.append(np.imag(i))
        outputs.append(np.real(i))
    return outputs

def deci_to_binstr(number,total_len):
    bin= format(number,'b')
    pad=total_len-len(bin)
    if pad>0:
        return '0'*pad+bin
    if pad<0:
        raise ValueError("total_len not large enough to express number")
    return bin

def awgn(x, noise_var):
    noise = np.sqrt(noise_var)*np.random.randn(len(x))
    return np.add(x, noise)

def binstr_to_deci(number):
    base=1
    deci=0
    for i in range(len(number)-1,-1,-1):
        deci+=int(number[i])*base
        base*=2
    return deci

if __name__=='__main__':
    print(bitstr_to_file_type_str('00110010'))
    print(bitstr_to_file_type_str('01001000'))
    print(bitstr_to_file_type_str('01000101'))

