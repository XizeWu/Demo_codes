import argparse
from load_data import *
from train import *
import scipy.io as sio

# seed setting
seed_setting(seed = 1)

# parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  type=str, default='mir')  # mir/nus/coco
parser.add_argument('--nbit',     type=int, default=16)

parser.add_argument('--batchsize', type=int, default='64')
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default='0.0001') 

parser.add_argument('--lamda1', type=float, default='10')
parser.add_argument('--lamda2', type=float, default='100')
parser.add_argument('--lamda3', type=float, default='10')

parser.add_argument('--topK', type=int, default='1000')

# coco: lambda1/2/3 = 1/1/1   lr=0.001  epoch70
# nus: lambda1/2/3 = 0.01/1/1   lr=0.005    epoch30
# mir: lambda1/2/3 = 10/100/10  lr=0.0001    epoch50

args = parser.parse_args()

def train_Demo(args):

    dataloader = get_loader(args.batchsize, args.dataset)
    train_loader = dataloader['train']

    model = Training(args, train_loader)

    return model, dataloader

def performance_eval(model, dataloader):
    database_loader = dataloader['database']
    query_loader = dataloader['query']

    model.eval().cuda()
    re_B, re_L, qu_B, qu_L = compress(database_loader, query_loader, model)

    _dict = {
        're_B': re_B,
        'qu_B': qu_B,
        're_L': re_L,
        'qu_L': qu_L
    }
    sava_path = 'hashcode/codes_' + args.dataset + '_' + str(args.nbit) + 'bits.mat'
    sio.savemat(sava_path, _dict)

    return re_B, re_L, qu_B, qu_L

def compress(database_loader, query_loader, model):
    re_BI = list([])
    re_L = list([])
    for _, (data_I, data_L, _) in enumerate(database_loader):
        with torch.no_grad():
            var_data_I = data_I.cuda()
            _, code_I, _ = model(var_data_I.to(torch.float))
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())
        re_L.extend(data_L.cpu().data.numpy())

    qu_BI = list([])
    qu_L = list([])
    for _, (data_I, data_L, _) in enumerate(query_loader):
        with torch.no_grad():
            var_data_I = data_I.cuda()
            _, code_I, _ = model(var_data_I.to(torch.float))
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        qu_L.extend(data_L.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_L = np.array(re_L)

    qu_BI = np.array(qu_BI)
    qu_L = np.array(qu_L)

    return re_BI, re_L, qu_BI, qu_L

if __name__ == '__main__':

    model, loader = train_Demo(args)

    print('[Bit: %d, topK: %d] >> lamda1: %.3f, lamda2: %.3f, lamda3: %.3f' % (args.nbit, args.topK, args.lamda1, args.lamda2, args.lamda3))

    re_B, re_L, qu_B, qu_L = performance_eval(model, loader)
    MAP = calculate_top_map(qu_B=qu_B, re_B=re_B, qu_L=qu_L, re_L=re_L, topk=args.topK)
    print(">>>>>>>>> MAP: %.3f"%(MAP))
