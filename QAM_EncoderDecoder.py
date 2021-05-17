from os import name
from bitarray import bitarray
import numpy as np

# -------------Encoder----------------
def file_to_bitstr(filename):
    """[convert the binary content in the file to a string of 0s and 1s]

    Args:
        filename ([string])

    Returns:
        [string]: [string of 0s and 1s]
    """
    with open(filename,'rb') as f:
        file_bytes=f.read()
        
    bin_strings=''
    for byte in file_bytes:
        binary_string = '{0:08b}'.format(byte)
        bin_strings+=binary_string
    return bin_strings



def encode_bitstr2symbols(bits):
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
    return symbols


    # np.real(element) >= 0:
    #         if np.imag(element) >= 0:
    #             data += '00'
    #         else:
    #             data += '10'
    #     else:
    #         if np.imag(element) >= 0:
    #             data += '01'
    #         else:
    #             data += '11'


# -------------Decoder----------------
def decode_symbols_2_bitstring(symbols):
    data = ''
    for i in range(len(symbols)):
        element=symbols[i]
        if np.real(element) >= 0:
            if np.imag(element) >= 0:
                data += '00'
            else:
                data += '10'
        else:
            if np.imag(element) >= 0:
                data += '01'
            else:
                data += '11'
    return data


def bitstr_to_file(bin_strings,filename):

    data_bytes = bitarray(bin_strings)

    with open(filename, 'wb') as f:
        f.write(data_bytes.tobytes())
    print("written to ",filename)


if __name__=='__main__':
    filename='trial.png'
    bits=file_to_bitstr(filename)
    symbols=encode_bitstr2symbols(bits)
    bin_strings=decode_symbols_2_bitstring(symbols)
    bitstr_to_file(bin_strings,'trial_encodeddecoded.png')
