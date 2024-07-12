import os
import argparse
from model import MODEL
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from data import ModelNet40
from util import npmat2euler
import numpy as np
from tqdm import tqdm
from writer import write_metrics_for_reg

torch.backends.cudnn.enabled = False # fix cudnn non-contiguous error

def Error_R(r1, r2):
    '''
    Calculate isotropic rotation degree error between r1 and r2.
    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :return:
    '''
    r2_inv = r2.transpose(0, 2, 1)
    r1r2 = np.matmul(r2_inv, r1)
    tr = r1r2[:, 0, 0] + r1r2[:, 1, 1] + r1r2[:, 2, 2]
    rads = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    degrees = rads / np.pi * 180
    return degrees


def Error_t(t1, t2, r2):
    '''
    Calculate isotropic translation error between t1 and t2.
    :param t1: shape=(B, 3), pred_t
    :param t2: shape=(B, 3), gtt
    :param R2: shape=(B, 3, 3), gtR
    :return:
    '''
    r2 = r2.transpose(0, 2, 1)
    t2 = np.squeeze(- r2 @ t2[..., None], axis=-1)
    error_t = np.squeeze(r2 @ t1[..., None], axis=-1) + t2
    error_t = np.linalg.norm(error_t, axis=-1)
    return error_t

    
def test_one_epoch(args, net, test_loader):
    net.eval()

    R_list = []
    t_list = []
    R_pred_list = []
    t_pred_list = []
    euler_list = []

    for src, target, R, t, euler in tqdm(test_loader):
        
        src = src.clone().detach() 
        target = target.clone().detach() 
        R = R.clone().detach() 
        t = t.clone().detach() 

        src = src.cuda()
        target = target.cuda()
        R = R.cuda()
        t = t.cuda()

        R_pred, t_pred, *_ = net(src, target)

        R_list.append(R.detach().cpu().numpy())
        t_list.append(t.detach().cpu().numpy())
        R_pred_list.append(R_pred.detach().cpu().numpy())
        t_pred_list.append(t_pred.detach().cpu().numpy())
        euler_list.append(euler.numpy())

    R = np.concatenate(R_list, axis=0)
    t = np.concatenate(t_list, axis=0)
    R_pred = np.concatenate(R_pred_list, axis=0)
    t_pred = np.concatenate(t_pred_list, axis=0)
    euler = np.concatenate(euler_list, axis=0)

    euler_pred = npmat2euler(R_pred)
    r_mse = np.mean((euler_pred - np.degrees(euler)) ** 2)
    r_rmse = np.sqrt(r_mse)
    r_mae = np.mean(np.abs(euler_pred - np.degrees(euler)))
    t_mse = np.mean((t - t_pred) ** 2)
    t_rmse = np.sqrt(t_mse)
    t_mae = np.mean(np.abs(t - t_pred))
    r_error = Error_R(R_pred, R)
    t_error = Error_t(t_pred, t, R)
    r_error = np.mean(r_error)
    t_error = np.mean(t_error)
    write_metrics_for_reg(os.path.join('./log/', "test.txt"), epoch=0, train=1, r_rmse=r_rmse, r_mae=r_mae, t_rmse=t_rmse, t_mae=t_mae)
    return r_rmse, r_mae, t_rmse, t_mae, r_error, t_error        

def train_one_epoch(args, net, train_loader, opt):
    net.train()

    R_list = []
    t_list = []
    R_pred_list = []
    t_pred_list = []
    euler_list = []

    for src, target, R, t, euler in tqdm(train_loader):
        src = src.cuda()
        target = target.cuda()
        R = R.cuda()
        t = t.cuda()

        opt.zero_grad()
        R_pred, t_pred, loss, *_ = net(src, target, R, t)

        R_list.append(R.detach().cpu().numpy())
        t_list.append(t.detach().cpu().numpy())
        R_pred_list.append(R_pred.detach().cpu().numpy())
        t_pred_list.append(t_pred.detach().cpu().numpy())
        euler_list.append(euler.numpy())

        loss.backward()
        opt.step()

    R = np.concatenate(R_list, axis=0)
    t = np.concatenate(t_list, axis=0)
    R_pred = np.concatenate(R_pred_list, axis=0)
    t_pred = np.concatenate(t_pred_list, axis=0)
    euler = np.concatenate(euler_list, axis=0)

    euler_pred = npmat2euler(R_pred)
    r_mse = np.mean((euler_pred - np.degrees(euler)) ** 2)
    r_rmse = np.sqrt(r_mse)
    r_mae = np.mean(np.abs(euler_pred - np.degrees(euler)))
    t_mse = np.mean((t - t_pred) ** 2)
    t_rmse = np.sqrt(t_mse)
    t_mae = np.mean(np.abs(t - t_pred))
    r_error = Error_R(R_pred, R)
    t_error = Error_t(t_pred, t, R)
    r_error = np.mean(r_error)
    t_error = np.mean(t_error)
    
    return r_rmse, r_mae, t_rmse, t_mae, r_error, t_error


def train(args, net, train_loader, test_loader):
    opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
    
  
    scheduler = MultiStepLR(opt, milestones=[40,55], gamma=0.1)


    for epoch in range(args.epochs):

        train_stats = train_one_epoch(args, net, train_loader, opt)
        test_stats = test_one_epoch(args, net, test_loader)

        print('=====  EPOCH %d  =====' % (epoch+1))
        print('TRAIN,  rot_RMSE: %f, rot_MAE: %f, trans_RMSE: %f, trans_MAE: %f, rot_ERROR: %f, trans_ERROR: %f' % train_stats)
        print('TEST,  rot_RMSE: %f, rot_MAE: %f, trans_RMSE: %f, trans_MAE: %f, rot_ERROR: %f, trans_ERROR: %f' % test_stats)

        torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        scheduler.step()

def test(args, net, test_loader):
    net.eval()

    if args.model_path is not '':
        assert os.path.exists(args.model_path), "Trying to resume, but model given doesn't exists."
        print("start test")
        state = torch.load(args.model_path)  
        net.load_state_dict(state)
    else:
        start_epoch = 0
        info_test_best = None

    test_stats = test_one_epoch(args, net, test_loader)        
    print('EVAL,  rot_RMSE: %f, rot_MAE: %f, trans_RMSE: %f, trans_MAE: %f, rot_ERROR: %f, trans_ERROR: %f' % test_stats)
    

def main():
    arg_bool = lambda x: x.lower() in ['true', 't', '1']
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--num_iter', type=int, default=3, metavar='N',
                        help='Number of iteration inside the network')
    parser.add_argument('--emb_dims', type=int, default=64, metavar='N',
                        help='Dimension of embeddings.')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--unseen', type=arg_bool, default='False',
                        help='Test on unseen categories')
    parser.add_argument('--gaussian_noise', type=arg_bool, default='False',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--alpha', type=float, default=0.75, metavar='N',
                        help='Fraction of points when sampling partial point cloud')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--clip', type=float, default=0.05, metavar='N',
                        help='noise')
    parser.add_argument('--model_path', type=str, default='premodels/modelt.t7', metavar='N',
                        help='Pretrained model path. Can be used to resume training, or to evaluate a specific checkpoint.')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        help='Choose dataset.')
    parser.add_argument('--num_point_preserved', type=int, default=384, metavar='N',
                        help='a half of point')
                        
                        
    args = parser.parse_args()
    print(args)
    
    ##### make checkpoint directory and backup #####
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name): 
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'): 
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')
    ##### make checkpoint directory and backup #####

    net = MODEL(args).cuda()
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(
            ModelNet40(partition='train', alpha=args.alpha, gaussian_noise=args.gaussian_noise, unseen=args.unseen, factor=args.factor),
            batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
        test_loader = DataLoader(
            ModelNet40(partition='test', alpha=args.alpha, gaussian_noise=args.gaussian_noise, unseen=args.unseen, factor=args.factor),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=8)
            
    if args.eval:
        test(args, net, test_loader)  
    else:
        train(args, net, train_loader, test_loader)     




if __name__ == '__main__':
    main()