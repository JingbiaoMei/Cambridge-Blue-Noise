import numpy as np
import sys

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
    """returns [input0.real, input0.img, input1.real, input1.img, ...]
    """
    outputs=[]
    for i in complex_inputs:
        outputs.append(np.real(i))
        outputs.append(np.imag(i))
    return outputs


def awgn(x, noise_var):
    noise = np.sqrt(noise_var)*np.random.randn(len(x))
    return np.add(x, noise)


if __name__=='__main__':
    print(divide_bin_to_sizes(11,[5,6,2,8]))

