from LDPC import *
from LDPCQAMCombined import *
from ldpc_jossy.py import ldpc
from util import *

# TODO: test with and without llr modification (LDPC line 475)

bits=file_to_bitstr('test_files/shakespeare_small.txt')[:1500]

print('input file length:',len(bits))



# ------ LDPC encode ------
LDPC_encoded_bits=LDPC_encode(bits,file_type='.txt')
print('\nlen(LDPC_encoded_bits)',len(LDPC_encoded_bits))

# ------ QPSK encode ------
symbols=encode_bitstr2symbols(LDPC_encoded_bits)
print(np.shape(symbols))


# --- Channel ---
data_transmitted=symbols
print("len(data_transmitted): ",len(data_transmitted))
input_P=abs(np.var(data_transmitted)+np.mean(data_transmitted)**2)
print("input average power: ",input_P)
snr=1.1
print("\nsnr:", snr)
var=input_P/snr
print("var:", var)

print("=====================")
data_received=awgn(data_transmitted,var)

#--- change into 1d array of ys---
# ys_=separate_real_img(data_received)

ys_=data_received

#-- LDPC decoding --

cks=np.array([var**0.5]*int(len(ys_)))

print('len(ys_):', len(ys_)) 
assert len(ys_)>=len(data_transmitted)

# -------withllrs_modification=True:------
print("# -------withllrs_modification=True:------")
LDPC_decoded_bits, file_type= LDPC_decode_with_niceCKs(ys_,N='',cks=cks,withllrs_modification=True) #LDPC_coder.encode(bin_strings, dectype='sumprod2', corr_factor=0.7)
print('\noutput file length:',len(LDPC_decoded_bits))
print('\ninput file length:',len(bits))
assert len(LDPC_decoded_bits)==len(bits)
biterrors = np.sum(bitstr_to_np_array(LDPC_decoded_bits) != bitstr_to_np_array(bits) )
print("\nbiterrors:",biterrors)
errorratio=np.log10(biterrors/len(bits))
print("error ratio = 10^",errorratio)
print("\n")

# -------withllrs_modification=False:------
print("# -------withllrs_modification=False:------")

LDPC_decoded_bits, file_type= LDPC_decode_with_niceCKs(ys_,N='',cks=cks,withllrs_modification=False) #LDPC_coder.encode(bin_strings, dectype='sumprod2', corr_factor=0.7)
print('\noutput file length:',len(LDPC_decoded_bits))
print('\ninput file length:',len(bits))
assert len(LDPC_decoded_bits)==len(bits)
biterrors = np.sum(bitstr_to_np_array(LDPC_decoded_bits) != bitstr_to_np_array(bits) )
print("\nbiterrors:",biterrors)
errorratio=np.log10(biterrors/len(bits))
print("error ratio = 10^",errorratio)
print("\n")

# bitstr_to_file(LDPC_decoded_bits,'decoded'+file_type)

print("finished one.")

# input_='1'*450+"0"*36
# # input_*=2

# symbols=encode_bitstr2symbols_via_LDPC(input_)
# ys=separate_real_img(symbols)
# # bits=decode_symbols_2_bitstring(symbols)
# input_P=abs(np.var(symbols)+np.mean(symbols)**2)
# print("input average power: ",input_P)


# snr=1.16

# print("\nsnr:", snr)

# var=input_P/snr
# print("var:", var)
# ys=awgn(ys,var)

# # bits=bitstr_to_np_array(bits)
# llrs=[]
# for i in ys:

#     llrs+=[llr(i,var)]

# llrs[200:200+450]=[-positive_infnity]*450

# # llrs[200+450:200+450+36]=[positive_infnity]*36

# llrs=np.array(llrs)

# LDPC_coder = ldpc.code(standard = '802.11n', rate = '1/2', z=81, ptype='A')

# # decoded=LDPC_coder.decode(llrs)

# (app,nit)= LDPC_coder.decode(llrs)
# transmitted=(app<0.0) # transmitted is np.array of Trues and Falses # this is the LDPC encoded bits before awgn transmission
# decoded=transmitted[:int(len(transmitted)/2)]

# str_decoded = ''
# for i in decoded:
#     str_decoded+=str(int(i))
# decoded=str_decoded[200:450+36+200]

# def error_rate(bits_tran, bits_rec):
#     if len(bits_tran)!=len(bits_rec):
#         raise ValueError("len(bits_tran)!=len(bits_rec), ",len(bits_tran),len(bits_rec))
#     length=len(bits_tran)
#     a1 = np.fromstring(bits_tran, 'u1') - ord('0')
#     a2 = np.fromstring(bits_rec, 'u1') - ord('0')
#     a2 = a2[0:length]
#     return (length - np.sum(a1 == a2)) / length


# print(error_rate(decoded,input_))

# hi=1