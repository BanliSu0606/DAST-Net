import time
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import pickle
import csv

import numpy as np

from models.DAST import DAST
from motion_pred.utils.dataset_assemble import DatasetAsb

sys.path.append(os.getcwd())
from motion_pred.utils.config import Config
from motion_pred.utils.visualization import render_animation
from scipy.spatial.distance import pdist, squareform
from uniutils import *


def denomarlize(*data):
    out = []
    for x in data:
        x = x * dataset.std + dataset.mean
        out.append(x)
    return out

def get_gt_asb(data):
    gt = data.reshape(data.shape[0], data.shape[1], -1)
    return gt[:, t_his:, :]


"""metrics"""
def compute_mpjpe(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2).mean(axis=1)
    return dist.min()


def compute_fde(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)[:, -1]
    return dist.min()


def get_prediction_mae(data, algo, sample_num, num_seeds=1, concat_hist=True, dct_m=None, idct_m=None, ft=False):
    traj_np = data.reshape(data.shape[0], data.shape[1], -1)

    traj = tensor(traj_np, device=device, dtype=dtype)

    va = cal_vel_acc(traj).to(device)
    va_padding = padding_vel(va, 'LastFrame', padding_idx, zero_idx).to(device)
    va = va.reshape(va.shape[0], va.shape[1], -1)
    va_padding = va_padding.reshape(va_padding.shape[0], va_padding.shape[1], -1)

    traj_padding = padding_traj(traj, 'LastFrame', padding_idx, zero_idx).to(device)

    if cfg.use_dct:
        traj_dct = torch.matmul(dct_m[:cfg.n_pre], traj)
        traj_padding_dct = torch.matmul(dct_m[:cfg.n_pre], traj_padding)
        va_dct = torch.matmul(dct_m[:cfg.n_pre], va.squeeze(0))
        va_padding_dct = torch.matmul(dct_m[:cfg.n_pre], va_padding.squeeze(0))
        X = traj_padding_dct
        V = va_padding_dct
    else:
        X = traj_padding
        V = va_padding

    if not ft:
        Y = models[algo](X)
    else:
        X = traj[:, :cfg.t_his, :]
        Y = models[algo].forward(X)

    if cfg.use_dct:
        Y = torch.matmul(idct_m[:, :cfg.n_pre], Y)


    if Y.shape[0] > 1:
        Y = Y.reshape(-1, sample_num, Y.shape[-2], Y.shape[-1])
    else:
        Y = Y[None, ...]
    if cfg.vae_specs['model_name'] == 'DAST':
        return Y
    else:
        return Y[:, :, t_his:, :]




def compute_stats():
    start_time = time.time()
    stats_func = {'MPJPE': compute_mpjpe,
                  'FDE': compute_fde}
    stats_names = list(stats_func.keys())
    stats_meter = {x: {y: AverageMeter() for y in algos} for x in stats_names}
    data_gen = dataset.iter_generator(step=cfg.t_his)
    num_samples = 0
    num_seeds = args.num_seeds
    dct, idct = get_dct_matrix(cfg.t_his + cfg.t_pred)
    dct = dct.float().to(device)
    idct = idct.float().to(device)
    all_gt = []
    all_pred = []

    for i, data in enumerate(data_gen):
        num_samples += 1
        data = data[np.newaxis, :, :, :]
        gt = get_gt_asb(data)
        for algo in algos:
            pred = get_prediction_mae(data, algo, sample_num=1, concat_hist=False, dct_m=dct, idct_m=idct,
                                          ft=cfg.ft)
            all_gt.append(gt.reshape(1, cfg.t_pred, dataset.skeleton.num_joints(), 3)[0])
            all_pred.append(pred.cpu().numpy()[0].reshape(cfg.t_pred, dataset.skeleton.num_joints(), 3))

            for stats in stats_names:
                val = 0
                for pred_i in pred:
                    val += stats_func[stats](pred_i.cpu(), gt) / num_seeds
                stats_meter[stats][algo].update(val)

        print('-' * 80)
        for stats in stats_names:
            str_stats = f'{num_samples:04d} {stats}: ' + ' '.join(
                [f'{x}: {y.val:.4f}({y.avg:.4f})' for x, y in stats_meter[stats].items()])
            print(str_stats)

    logger.info('=' * 80)
    for stats in stats_names:
        str_stats = f'Total {stats}: ' + ' '.join([f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()])
        logger.info(str_stats)
    logger.info('=' * 80)

    logger.info('Time cost: {:.4f} seconds'.format(time.time() - start_time))

    with open('%s/stats_%s.csv' % (cfg.result_dir, args.num_seeds), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + algos)
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {x: y.avg for x, y in meter.items()}
            new_meter['Metric'] = stats
            writer.writerow(new_meter)

    all_gt = np.array(all_gt)
    all_pred = np.array(all_pred)
    print(all_gt.shape, all_pred.shape)
    np.savez('%s/gt_pred.npz' % (cfg.result_dir), gt=all_gt, pred=all_pred)


def get_mae_model(cfg):
    if cfg.dataset == 'asb':
        joints_num = 13
    if cfg.vae_specs['model_name'] == 'DAST':
        model = DAST(3, 256,[64, 128, 256],3,cfg.t_his,cfg.t_pred,joints_num,0.1,8).to(device)
    return model


if __name__ == '__main__':

    all_algos = ['mae']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="asb_25_100")
    parser.add_argument('--mode', default='stats')
    parser.add_argument('--data', default='test')
    parser.add_argument('--action', default='all')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--ft', type=bool, default=False)
    parser.add_argument('--iter_ft', type=int, default=70)
    parser.add_argument('--use_best_model', action='store_true', default=True, 
                    help='Whether to use the best model saved by early stopping')
    for algo in all_algos:
        parser.add_argument('--iter_%s' % algo, type=int, default=100)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if args.gpu_index >= 0 and torch.cuda.is_available() \
        else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    torch.set_grad_enabled(False)
    cfg = Config(args.cfg)
    logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

    algos = []
    for algo in all_algos:
        iter_algo = 'iter_%s' % algo
        num_algo = 'num_%s_epoch' % algo
        setattr(args, iter_algo, getattr(cfg, num_algo))
        print(iter_algo, num_algo)
        algos.append(algo)
    vis_algos = algos.copy()

    if args.action != 'all':
        args.action = set(args.action.split(','))

    """parameter"""
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    if cfg.use_dct:
        num_frame = cfg.n_pre
    else:
        num_frame = cfg.t_his + cfg.t_pred

    """data"""
    if cfg.dataset == 'asb':
        dataset_cls = DatasetAsb
    dataset = dataset_cls(args.data, t_his, t_pred, actions='all', use_vel=cfg.use_vel)

    padding_idx, zero_idx = generate_pad('LastFrame', cfg.t_his, cfg.t_pred) 

    """models"""
    model_generator = {
        'mae': get_mae_model,
    }

    models = {}
    for algo in algos:
        models[algo] = model_generator[algo](cfg)
        if not cfg.ft:
            model_path = getattr(cfg, f"{algo}_model_path") % getattr(args, f'iter_{algo}')
        else:
            if args.use_best_model:
                import glob
                best_model_files = glob.glob(os.path.join(cfg.model_dir, 'best_model_epoch*.pkl'))
                if best_model_files:
                    latest_best = max(best_model_files, key=lambda x: int(x.split('epoch')[1].split('.')[0]))
                    model_path = latest_best
                else:
                    raise FileNotFoundError("No best model found from early stopping")
            else:
                model_path = getattr(cfg, f"ft_model_path") % getattr(args, f'iter_ft')

        print(f'loading {algo} model from checkpoint: {model_path}')
        model_cp = pickle.load(open(model_path, "rb"))
        models[algo].load_state_dict(model_cp['model_dict'])
        models[algo].to(device)
        models[algo].eval()
        total_params = sum(p.numel() for p in list(models[algo].parameters())) / 1000000.0
        print(algo, " params: {:.3f}M".format(total_params))
    if cfg.normalize_data:
        dataset.normalize_data(model_cp['meta']['mean'], model_cp['meta']['std'])

    if args.mode == 'stats':
        compute_stats()
