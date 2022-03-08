import random
import os
import numpy as np
import torch
from torch.autograd import Variable
import sklearn.preprocessing as pp
import sklearn.metrics.pairwise as pw
from sklearn.metrics.pairwise import rbf_kernel


def seed_setting(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]
# usage
# x=[0, 2, 5, 4]
# num = 8
# one_hot(x, num)

# For protopyte net
def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def cos_similarity(x1,x2):
    t1 = x1.dot(x2.T)

    x1_linalg = np.linalg.norm(x1,axis=1)
    x2_linalg = np.linalg.norm(x2,axis=1)
    x1_linalg = x1_linalg.reshape((x1_linalg.shape[0],1))
    x2_linalg = x2_linalg.reshape((1,x2_linalg.shape[0]))
    t2 = x1_linalg.dot(x2_linalg)
    cos = t1/t2

    return cos

def calculate_S(z):
    S = pw.cosine_similarity(z, z)
    return S

def calculate_S_multi(z1, z2):  # z: np.ndarray
    #z = torch.nn.functional.normalize(z, p=2, dim=1)
    #S = torch.matmul(z1, z2.T)
    z1 = pp.normalize(z1, norm='l2')
    z2 = pp.normalize(z2, norm='l2')
    S = np.matmul(z1, z2.T)
    return S

def calculate_rou(S, rate=0.4):
    m = S.shape[0]
    rou = np.zeros((m))

    t = int(rate * m * m)
    temp = np.sort(S.reshape((m * m,)))
    Sc = temp[-t]
    rou = np.sum(np.sign(S - Sc), axis=1) - np.sign(S.diagonal() - Sc)

    return rou

def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).float()
    return S

def CalcSim(label1, label2):
    # calculate the similar matrix
    #if use_gpu:
    Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
    #else:
    #    Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.FloatTensor)
    return Sim

def log_trick(x):
    lt = torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(
        x, Variable(torch.FloatTensor([0.]).cuda()))
    return lt

def normalize(affnty):
    col_sum = zero2eps(np.sum(affnty, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affnty, axis=0))
    out_affnty = affnty/col_sum # row data sum = 1
    in_affnty = np.transpose(affnty/row_sum) # col data sum = 1 then transpose
    return in_affnty, out_affnty # col, row

def zero2eps(x):
    x[x == 0] = 1
    return x

def affinity_eculi(data_1: np.ndarray, data_2: np.ndarray)->np.ndarray:
    '''
    Use the VGG deep feature to create graph.
    :param data_1:
    :param data_2:
    :return:
    '''
    XYt = np.matmul(data_1, data_2.T)
    X2, Y2 = data_1 ** 2, data_2 ** 2
    X2_sum = np.sum(X2, axis=1).reshape(X2.shape[0], 1)
    Y2_sum = np.sum(Y2, axis=1).reshape(1, Y2.shape[0])
    tmp = X2_sum + Y2_sum - 2 * XYt
    tmp[tmp < 0] = 0 # process float operater error
    affinity = np.sqrt(tmp)
    affinity = np.exp(-affinity)

    #in_aff, out_aff = normalize(affinity) # col row
    return affinity

def affinity_eculi_gpu(data_1: torch.Tensor, data_2: torch.Tensor, I_size=0, theta=2):
    XYt = torch.matmul(data_1, data_2.T)
    X2, Y2 = torch.mul(data_1, data_1), torch.mul(data_2, data_2)
    X2_sum = torch.sum(X2, dim=1).reshape(X2.size()[0], 1)
    Y2_sum = torch.sum(Y2, dim=1).reshape(1, Y2.size()[0])
    tmp = X2_sum + Y2_sum - 2 * XYt
    tmp[tmp < 0] = 0
    affinity = tmp / theta
    # affinity = torch.sqrt(tmp)
    new_affinity = torch.exp(-affinity)

    # print(new_affinity.size(), data_1.size(), data_2.size())
    if I_size != 0:
        I = Variable(torch.eye(I_size)).cuda()
        new_affinity = torch.cat([I, new_affinity], dim=1)

    in_aff = torch.nn.functional.normalize(new_affinity, p=2, dim=0).T
    out_aff = torch.nn.functional.normalize(new_affinity, p=2, dim=1)

    return in_aff, out_aff, new_affinity

def rbf_affnty(X, Y, topk=10):
    X = X.numpy()
    # Y = Y.numpy()

    rbf_k = rbf_kernel(X, Y)
    topk_max = np.argsort(rbf_k, axis=1)[:,-topk:]

    affnty = np.zeros(rbf_k.shape)
    for col_idx in topk_max.T:
        affnty[np.arange(rbf_k.shape[0]), col_idx] = 1.0

    in_affnty, out_affnty = normalize(affnty)
    return torch.Tensor(in_affnty), torch.Tensor(out_affnty)

def affinity_fusion(feature1: np.ndarray, feature2: np.ndarray, flag=True):
    '''
    :param feature1:
    :param feature2:
    :param flag: "flag=True" means that feature will be normalized.
    :param fusion_factor: the factor of feature cosine-similarity
    :return:
    '''
    if flag: # flag==false
        pro_feature1 = pp.normalize(feature1, norm='l2') # to calculate the cos-similarity
        pro_feature2 = pp.normalize(feature2, norm='l2') # to calculate the cos-similarity
    else:
        pro_feature1 = feature1
        pro_feature2 = feature2

    affinity = affinity_eculi(pro_feature1, pro_feature2)

    in_aff, out_aff = normalize(affinity)  # col row
    return in_aff, out_aff, affinity
    
    
    
def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1] # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH
    
    
    
def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap
    
    
    
