from numpy import Infinity
from QAM_EncoderDecoder import *
from EncoderDecoder_test import *
# from LDPC import *
from ldpc_jossy.py import ldpc
from LDPC import *
from util import *
from sys import stdout
import matplotlib.pyplot as plt

#  Initialise the dictionary

dhi=do_it(N=2048,prefix_no=256,snr=1.5,ReturnError=True,len_protection='input_repeat_then_LDPC',inputLenIndicator_len=24,inputGuard_len=8,rate='1/2',r=0.5,z=27,filename='input_small.png',OnlyTestLen=True,repeat_times=1)

# LenDetected_all={} # array start with a number indicating the number of times of tests

# for len_protection in ['no','input_repeat9_then_LDPC','input_repeat3_then_LDPC','guardBits']: #
#     trues=[0] # array start with a number indicating the number of times that Len is detected
#     for snr in range(148,150,1):
#         trues+=[0]
#     LenDetected_all[len_protection]=np.array(trues)
# print(LenDetected_all)


# for len_protection in ['no','input_repeat9_then_LDPC','input_repeat3_then_LDPC']: #,'guardBits'
# # for len_protection in [1,2,3]:
#     trues=[1] # array starts with a number indicating the number of times of tests 
#     snrs=[]
#     for snr in range(98,200,1):
#         snr=snr/100
#         true=do_it(N=2048,prefix_no=256,snr=snr,ReturnError=True,len_protection=len_protection,inputLenIndicator_len=24,inputGuard_len=8,rate='1/2',r=0.5,z=27,filename='input_small.png',OnlyTestLen=True)
        
# #         try:
# # #             true=False
# #             true=do_it(N=2048,prefix_no=256,snr=snr,ReturnError=True,len_protection=len_protection,inputLenIndicator_len=24,inputGuard_len=8,rate='1/2',r=0.5,z=27,filename='test.txt',OnlyTestLen=True)
# #         except:
# #             print("decoder failed, putting error as 10^0")
# #             true=0
#         trues+=[true]
#         snrs+=[snr]
#         print("\n\n\n")
#     # plt.cla()
#     # axes = plt.gca()
#     trials_already=LenDetected_all[len_protection][0]+1
#     print('trials_already: ',trials_already)
#     LenDetected_all[len_protection] = np.add(LenDetected_all[len_protection],np.array(trues))
#     LenDetected_all[len_protection][0] = trials_already
    
print("LenDetected_all: ",dhi)

    


# bitstr_to_file('0101010101001','test.txt',cut=0)

# # TODO
# for len_protection in ['no','input_repeat9_then_LDPC','input_repeat3_then_LDPC','guardBits']:
# # for len_protection in [1,2,3]:
#     errors=[]
#     snrs=[]
#     for snr in range(100,150,1):
#         snr=snr/100
#         try:
#             # error=len_protection
#             error=do_it(N=2048,prefix_no=256,snr=snr,ReturnError=True,len_protection=len_protection,inputLenIndicator_len=24,inputGuard_len=8,rate='1/2',r=0.5,z=27)
#         except:
#             print("decoder failed, putting error as 10^0")
#             error=0
#         errors.append(error)
#         snrs.append(snr)
#     # plt.cla()
#     # axes = plt.gca()
#     plt.plot(snrs,errors,label='len_protection is {0}'.format(len_protection))


# # plt.plot(snrs,errors)
# plt.legend()
# plt.show()
