from bitarray import bitarray

def file_to_bit_trings(filename):
    """[convert the binary content in the file to a string of 0s and 1s]

    Args:
        filename ([string])

    Returns:
        [string]: [string of 0s and 1s]
    """
    with open(filename,'rb') as f:
        file_bytes=f.read()
        print(type(file_bytes))
        
    bin_strings=''
    for byte in file_bytes:
        # int_value = ord(byte)
        binary_string = '{0:08b}'.format(byte)
        # print(binary_string)
        bin_strings+=binary_string
    return bin_strings




    # data_bytes = bitarray(bin_strings)

    # with open("encoded.png", 'wb') as f:
    #     f.write(data_bytes.tobytes())
