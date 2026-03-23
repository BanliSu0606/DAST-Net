import argparse
import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
import sys
import time


sys.path.append(os.getcwd())

from torch import optim
from torch.utils.tensorboard import SummaryWriter
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_assemble import DatasetAsb
from motion_pred.utils.dataset_humaneva import DatasetHumanEva
from torch import nn
from models.DAST import DAST
from uniutils import *


def loss_function(X, Y_r, Y, mu, logvar):
    MSE = (Y_r - Y).pow(2).sum() / Y.shape[1]
    MSE_v = (X[-1] - Y_r[0]).pow(2).sum() / Y.shape[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / Y.shape[1]
    loss_r = MSE + cfg.lambda_v * MSE_v + cfg.beta * KLD
    return loss_r, np.array([loss_r.item(), MSE.item(), MSE_v.item(), KLD.item()])


def loss_function_va(X, Y_r, Y, V_r, V, mu, logvar):
    MSE_x = (Y_r - Y).pow(2).sum() / Y.shape[1]
    MSE_va = (V_r - V).pow(2).sum() / V.shape[1]
    MSE_v = (X[-1] - Y_r[0]).pow(2).sum() / Y.shape[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / Y.shape[1]
    loss_r = MSE_x + MSE_v + cfg.lambda_v * MSE_v + cfg.beta * KLD
    return loss_r, np.array([loss_r.item(), MSE_x.item(), MSE_va.item(), MSE_v.item(), KLD.item()])


def save_best_model(model, dataset, epoch, cfg):
    cp_path = os.path.join(cfg.model_dir, f'best_model_epoch{epoch+1}.pkl')
    model_cp = {
        'model_dict': model.state_dict(), 
        'meta': {
            'std': dataset.std, 
            'mean': dataset.mean, 
            'epoch': epoch+1,
            'loss': best_loss 
        }
    }
    pickle.dump(model_cp, open(cp_path, 'wb'))
    logger.info(f'Saved best model at epoch {epoch+1} to {cp_path}')

def load_best_model(model, cfg):
    import glob
    best_model_files = glob.glob(os.path.join(cfg.model_dir, 'best_model_epoch*.pkl'))
    if not best_model_files:
        logger.warning('No best model found to load')
        return

    latest_best = max(best_model_files, key=lambda x: int(x.split('epoch')[1].split('.')[0]))
    logger.info(f'Loading best model from {latest_best}')
    model_cp = pickle.load(open(latest_best, "rb"))
    model.load_state_dict(model_cp['model_dict'])
    return model_cp['meta']['epoch']


def save_interval_model(model, dataset, epoch, cfg):
    cp_path = cfg.ft_model_path % (epoch + 1)
    model_cp = {
        'model_dict': model.state_dict(), 
        'meta': {'std': dataset.std, 'mean': dataset.mean}
    }
    pickle.dump(model_cp, open(cp_path, 'wb'))


def train_asb(epoch, dct_m=None, idct_m=None, finetune=False):
    t_s = time.time()
    train_losses = 0
    total_num_sample = 0
    loss_names = ['MSE']
    mse_m = nn.MSELoss()
    
    if not finetune:
        padding_mode = 'LastFrame'
        stride = 2
    else:
        padding_mode = 'Zero'
        stride = 1

    if cfg.dataset == 'asb':
        generator = dataset.sampling_generator(num_samples=cfg.num_mae_data_sample, batch_size=cfg.batch_size, aug=True, stride=stride)
    else:
        generator = dataset.sampling_generator(num_samples=cfg.num_mae_data_sample, batch_size=cfg.batch_size, aug=True)
    padding_mode = 'LastFrame'
    padding_idx, zero_idx = generate_pad(padding_mode, cfg.t_his, cfg.t_pred)

    epoch_loss = 0
    
    for traj_np in generator:
        B, T, V = traj_np.shape[:3]
        traj = tensor(traj_np, device=device, dtype=dtype)

        if cfg.dataset == 'asb':
            traj = traj.reshape(B, T, -1)
            traj_masked = traj_masked.reshape(B, T, -1).to(device)
        else:
            traj = traj[..., 1:, :].reshape(B, T, -1)

        traj_padding = padding_traj(traj, padding_mode, padding_idx, zero_idx).to(device)
        traj_his = traj[:, :cfg.t_his, :]

        va = cal_vel_acc(traj)
        va_padding = padding_vel(va, padding_mode, padding_idx, zero_idx)
        va = va.reshape(B, T, -1)
        va_padding = va_padding.reshape(B, T, -1)


        if not finetune:
            traj_pred = model(traj_masked)
        else:
            traj_pred = model.forward(traj_his)


        if finetune:
            T_pred = traj_pred.shape[1]
            va_hat = cal_vel_acc(traj_pred)
            va_hat = va_hat.reshape(B, T_pred, -1)
            va_cropped = va[:, cfg.t_his:, :]
            l_reg = mse_m(traj[:, cfg.t_his:, :], traj_pred)
            l_va = mse_m(va_cropped, va_hat) * 0.1
            loss = l_reg + l_va

        losses = np.array([loss.item()])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses += losses
        epoch_loss += loss.item()
        total_num_sample += 1

    scheduler.step()
    dt = time.time() - t_s
    train_losses /= total_num_sample
    epoch_avg_loss = epoch_loss / total_num_sample
    
    lr = optimizer.param_groups[0]['lr']
    if not finetune:
        stage_name = 'Train '
    else:
        stage_name = 'Finetune '
    losses_str = stage_name + ' '.join(['{}: {:.9f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    logger.info('====> Epoch: {} Time: {:.2f} {} lr: {:.5f}'.format(epoch, dt, losses_str, lr))
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalar('mae_' + name, loss, epoch)
    
    return epoch_avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="humaneva_25_100")
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--ft_iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--milestones', type=list, default=[35, 75, 125, 175, 225, 275, 325])
    parser.add_argument('--ft_milestones', type=list, default=[50, 75, 100, 125, 150, 200, 250, 300])
    parser.add_argument('--gamma', type=float, default=0.9)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    cfg = Config(args.cfg, test=args.test)
    tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    """parameter"""
    mode = args.mode
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    if cfg.use_dct:
        num_frame = cfg.n_pre
    else:
        num_frame = cfg.t_his + cfg.t_pred

    """data"""
    if cfg.dataset == 'asb':
        dataset_cls = DatasetAsb
    elif cfg.dataset == 'humaneva':
        dataset_cls = DatasetHumanEva
    dataset = dataset_cls(mode, t_his, t_pred, actions='all', use_vel=cfg.use_vel)
    if cfg.normalize_data:
        dataset.normalize_data()

    """creat model """
    if cfg.dataset == 'asb':
        joints_num = 13
    elif cfg.dataset == 'humaneva':
        joints_num = 14

    if cfg.vae_specs['model_name'] == 'DAST':
        model = DAST(3, 256,[64, 128, 256],3,cfg.t_his, cfg.t_pred,joints_num,0.1,8).to(device)
    


    optimizer = optim.Adam(model.parameters(), lr=cfg.mae_lr)
    scheduler = get_scheduler(optimizer, policy='multistep', milestones=args.milestones, gamma=0.9)

    dct, idct = get_dct_matrix(cfg.t_his + cfg.t_pred)
    dct = dct.float().to(device)
    idct = idct.float().to(device)

    if args.iter > 0:
        cp_path = cfg.mae_model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])

    if cfg.pre_train:
        model.to(device)
        model.train()
        for i in range(args.iter, cfg.num_mae_epoch):
            if cfg.dataset == 'asb':
                train_loss = train_asb(i, dct, idct, finetune=False)
            if cfg.save_model_interval > 0 and (i + 1) % cfg.save_model_interval == 0:
                cp_path = cfg.mae_model_path % (i + 1)
                model_cp = {'model_dict': model.state_dict(), 'meta': {'std': dataset.std, 'mean': dataset.mean}}
                pickle.dump(model_cp, open(cp_path, 'wb'))

    if args.ft_iter > 0:
        cp_path = cfg.ft_model_path % args.ft_iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])

    if cfg.ft:
        model.to(device)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=cfg.ft_lr)
        scheduler = get_scheduler(optimizer, policy='multistep', milestones=args.ft_milestones)
        best_loss = float('inf')
        no_improve_count = 0
        patience = int(cfg.early_stopping_patience) 
        min_delta = float(cfg.early_stopping_min_delta) 
        
        logger.info(f'Starting fine-tuning with early stopping (patience={patience}, min_delta={min_delta})')
        
        for i in range(args.ft_iter, cfg.num_ft_epoch):
            train_loss = train_asb(i, dct, idct, finetune=True)
            
            if isinstance(train_loss, np.ndarray):
                train_loss = train_loss.item() if train_loss.size == 1 else float(train_loss[0])
            elif isinstance(train_loss, torch.Tensor):
                train_loss = train_loss.item()
        
            improvement = best_loss - train_loss
            if improvement > min_delta:
                logger.info(f'Epoch {i+1}: Loss improved from {best_loss:.6f} to {train_loss:.6f} (improvement: {improvement:.6f})')
                best_loss = train_loss
                no_improve_count = 0
                save_best_model(model, dataset, i, cfg)
            else:
                no_improve_count += 1
                logger.info(f'Epoch {i+1}: Loss={train_loss:.6f}, Best={best_loss:.6f}, NoImprove={no_improve_count}/{patience}')
        
            tb_logger.add_scalar('train_loss', train_loss, i)
            tb_logger.add_scalar('best_loss', best_loss, i)
            tb_logger.add_scalar('no_improve_count', no_improve_count, i)
        
            if no_improve_count >= patience:
                logger.info(f'Early stopping triggered at epoch {i+1}')
                logger.info(f'Best loss achieved: {best_loss:.6f}')
            
                best_epoch = load_best_model(model, cfg)
                if best_epoch:
                    logger.info(f'Resumed best model from epoch {best_epoch}')
            
                break
        
            if cfg.save_model_interval > 0 and (i + 1) % cfg.save_model_interval == 0:
                save_interval_model(model, dataset, i, cfg)
        
        logger.info(f'Training completed. Best loss: {best_loss:.6f}')
        logger.info(f'Total epochs trained: {i+1}')
        

        save_interval_model(model, dataset, i, cfg)