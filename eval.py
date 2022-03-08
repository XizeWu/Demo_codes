from utils import calculate_top_map
import numpy as np
import scipy.io as sio

path = '/data/s2019020823/ResNetFeat/Vbinary/mir_resnet_binary.mat'
dataset = sio.loadmat(path)

query_B = np.array(dataset['qu_B'], dtype=np.float)
query_L = np.array(dataset['L_te'], dtype=np.float)

retrieval_B = np.array(dataset['re_B'], dtype=np.float)
retrieval_L = np.array(dataset['L_db'], dtype=np.float)




MAP = calculate_top_map(qu_B=query_B, re_B=retrieval_B, qu_L=query_L, re_L=retrieval_L, topk=50)

print("MAP:", MAP)




