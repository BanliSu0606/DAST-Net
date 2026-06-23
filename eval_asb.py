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

    X = traj_padding
    V = va_padding

    if not ft:
        Y = models[algo](X)
    else:
        X = traj[:, :cfg.t_his, :]
        Y = models[algo].forward(X)

    if Y.shape[0] > 1:
        Y = Y.reshape(-1, sample_num, Y.shape[-2], Y.shape[-1])
    else:
        Y = Y[None, ...]
    
    return Y


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


def estimate_flops_manual(model, input_tensor):
    try:
        from torch.profiler import profile, record_function, ProfilerActivity
        
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True) as prof:
            with record_function("model_inference"):
                _ = model(input_tensor)
        
        total_flops = 0
        for event in prof.key_averages():
            if event.flops is not None:
                total_flops += event.flops
        
        return total_flops
    except:
        return sum(p.numel() for p in model.parameters()) * 2 * input_tensor.shape[1]

def compute_extended_metrics():
    start_time = time.time()
    
    # ==================== Parameters ====================
    logger.info('=' * 80)
    logger.info('1. Model Parameters')
    logger.info('-' * 40)
    for algo in algos:
        total_params = sum(p.numel() for p in models[algo].parameters())
        trainable_params = sum(p.numel() for p in models[algo].parameters() if p.requires_grad)
        logger.info(f'  {algo}:')
        logger.info(f'    Total params:     {total_params:,} ({total_params/1e6:.3f}M)')
        logger.info(f'    Trainable params: {trainable_params:,} ({trainable_params/1e6:.3f}M)')
    
    # ==================== Inference Latency ====================
    logger.info('=' * 80)
    logger.info('2. Inference Latency')
    logger.info('-' * 40)
    
    data_gen = dataset.iter_generator(step=cfg.t_his)
    
    sample_data_list = []
    for i, data in enumerate(data_gen):
        if i == 0:
            sample_data_list.append(data)
            break
    
    sample_data = sample_data_list[0]
    
    if len(sample_data.shape) == 3:
        sample_data = sample_data[np.newaxis, :, :, :]
    
    dct, idct = get_dct_matrix(cfg.t_his + cfg.t_pred)
    dct = dct.float().to(device)
    idct = idct.float().to(device)
    
    # Warm-up
    for algo in algos:
        for _ in range(10):
            _ = get_prediction_mae(sample_data, algo, sample_num=1, concat_hist=False,
                                   dct_m=dct, idct_m=idct, ft=cfg.ft)
    
    num_measure = 100
    latency_results = {}
    memory_results = {}
    
    for algo in algos:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            times = []
            
            for _ in range(num_measure):
                starter.record()
                _ = get_prediction_mae(sample_data, algo, sample_num=1, concat_hist=False,
                                       dct_m=dct, idct_m=idct, ft=cfg.ft)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))
            
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            
        else:
            times = []
            for _ in range(num_measure):
                t0 = time.perf_counter()
                _ = get_prediction_mae(sample_data, algo, sample_num=1, concat_hist=False,
                                       dct_m=dct, idct_m=idct, ft=cfg.ft)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
            peak_memory = 0
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        latency_results[algo] = avg_time
        memory_results[algo] = peak_memory
        
        logger.info(f'  {algo}:')
        logger.info(f'    Mean inference time: {avg_time:.4f} ms ± {std_time:.4f} ms')
        logger.info(f'    Peak GPU memory:     {peak_memory:.2f} MB')
        logger.info(f'    (measured over {num_measure} runs on {device.type})')

    # ==================== FLOPs ====================
    logger.info('=' * 80)
    logger.info('FLOPs (Floating Point Operations)')
    logger.info('-' * 40)
    
    flop_results = {}
    try:
        from fvcore.nn import FlopCountAnalysis
        for algo in algos:
            model = models[algo]
            
            traj_np = sample_data
            if cfg.dataset == 'humaneva':
                traj_np = traj_np[..., 1:, :]
            elif cfg.dataset == 'h36m':
                traj_np = traj_np[..., 1:, :]
            
            traj = tensor(traj_np, device=device, dtype=dtype).reshape(traj_np.shape[0], traj_np.shape[1], -1)
            
            model_name = cfg.vae_specs['model_name']
            if model_name in ['STSGCN', 'GCNext', 'ST']:
                input_tensor = traj[:, :cfg.t_his, :]
            else:
                if cfg.use_dct:
                    input_tensor = torch.matmul(dct[:cfg.n_pre], traj)
                else:
                    input_tensor = traj
            
            try:
                flops = FlopCountAnalysis(model, input_tensor)
                total_flops = flops.total()
                flop_results[algo] = total_flops
                logger.info(f'  {algo}: {total_flops/1e9:.4f} GFLOPs')
            except Exception as e:
                logger.warning(f'  {algo}: fvcore FLOPs computation failed: {e}')
                total_flops = estimate_flops_manual(model, input_tensor)
                flop_results[algo] = total_flops
                logger.info(f'  {algo}: ~{total_flops/1e9:.4f} GFLOPs (estimated)')
    except ImportError:
        logger.warning('fvcore not installed. Install with: pip install fvcore')
        logger.info('Using manual FLOPs estimation instead...')
        for algo in algos:
            model = models[algo]
            traj_np = sample_data
            if cfg.dataset == 'humaneva':
                traj_np = traj_np[..., 1:, :]
            elif cfg.dataset == 'h36m':
                traj_np = traj_np[..., 1:, :]
            
            traj = tensor(traj_np, device=device, dtype=dtype).reshape(traj_np.shape[0], traj_np.shape[1], -1)
            
            model_name = cfg.vae_specs['model_name']
            if model_name in ['STSGCN', 'GCNext', 'ST']:
                input_tensor = traj[:, :cfg.t_his, :]
            else:
                if cfg.use_dct:
                    input_tensor = torch.matmul(dct[:cfg.n_pre], traj)
                else:
                    input_tensor = traj
            
            total_flops = estimate_flops_manual(model, input_tensor)
            flop_results[algo] = total_flops
            logger.info(f'  {algo}: ~{total_flops/1e9:.4f} GFLOPs (estimated)')
    
    
    logger.info('=' * 80)
    logger.info('Extended metrics evaluation completed.')
    logger.info(f'Time cost: {time.time() - start_time:.2f} seconds')
    logger.info('=' * 80)
    
    # ==================== BLE ====================
    logger.info('=' * 80)
    logger.info('3. Bone Length Error (BLE)')
    logger.info('-' * 40)
    
    bone_pairs = [
        (0, 1), (0, 7),
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        (7, 8), (8, 9), (9, 10), (10, 11), (11, 12)]
    
    data_gen = dataset.iter_generator(step=cfg.t_his)
    bone_length_errors = {algo: [] for algo in algos}
    num_samples_processed = 0
    max_samples = 100
    
    for i, data in enumerate(data_gen):
        if i >= max_samples:
            break
            
        if len(data.shape) == 3:
            data = data[np.newaxis, :, :, :]
        
        try:
            gt = get_gt_asb(data)
            
            for algo in algos:
                pred = get_prediction_mae(data, algo, sample_num=1, concat_hist=False,
                                          dct_m=dct, idct_m=idct, ft=cfg.ft)
                
                if torch.is_tensor(pred):
                    pred = pred.cpu().numpy()
                
                if len(pred.shape) == 3:
                    num_joints = 13
                    pred = pred.reshape(cfg.t_pred, num_joints, 3)
                elif len(pred.shape) == 4:
                    pred = pred[0]
                
                if len(gt.shape) == 4:
                    gt_sample = gt[0]
                else:
                    gt_sample = gt
                
                num_joints = min(pred.shape[1], gt_sample.shape[1])
                pred = pred[:, :num_joints, :]
                gt_sample = gt_sample[:, :num_joints, :]
                
                for p in range(cfg.t_pred):
                    for (pa, ch) in bone_pairs:
                        if pa < num_joints and ch < num_joints:
                            gt_bone_len = np.linalg.norm(gt_sample[p, ch, :] - gt_sample[p, pa, :])
                            pred_bone_len = np.linalg.norm(pred[p, ch, :] - pred[p, pa, :])
                            bone_error = np.abs(pred_bone_len - gt_bone_len)
                            bone_length_errors[algo].append(bone_error)
            
            num_samples_processed += 1
            if (i + 1) % 20 == 0:
                logger.info(f'  Processed {i+1} samples for bone length consistency')
                
        except Exception as e:
            continue
    
    for algo in algos:
        if bone_length_errors[algo]:
            errors = np.array(bone_length_errors[algo])
            mean_ble = np.mean(errors)
            std_ble = np.std(errors)
            max_ble = np.max(errors)
            logger.info(f'  {algo}:')
            logger.info(f'    Mean Bone Length Error: {mean_ble:.6f} m')
            logger.info(f'    Std  Bone Length Error: {std_ble:.6f} m')
            logger.info(f'    Max  Bone Length Error: {max_ble:.6f} m')
        else:
            logger.warning(f'  {algo}: No valid bone length measurements')
    
    logger.info('=' * 80)
    logger.info('Extended metrics evaluation completed.')
    logger.info(f'Time cost: {time.time() - start_time:.2f} seconds')
    logger.info('=' * 80)

    
def get_mae_model(cfg):
    joints_num = 13
    model = DAST(3, 256,[64, 128, 256],[2,3,5],cfg.t_his,cfg.t_pred,joints_num,0.1,8).to(device)
    return model


if __name__ == '__main__':

    all_algos = ['mae']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="asb_st_4")
    parser.add_argument('--mode', default='extended', choices=['stats', 'extended'])
    parser.add_argument('--data', default='test')
    parser.add_argument('--action', default='all')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--ft', type=bool, default=False)
    parser.add_argument('--iter_ft', type=int, default=70)
    parser.add_argument('--use_best_model', action='store_true', default=True)
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
        models[algo].load_state_dict(model_cp['model_dict'], strict=False)
        models[algo].to(device)
        models[algo].eval()
        total_params = sum(p.numel() for p in list(models[algo].parameters())) / 1000000.0
        print(algo, " params: {:.3f}M".format(total_params))
    if cfg.normalize_data:
        dataset.normalize_data(model_cp['meta']['mean'], model_cp['meta']['std'])

    if args.mode == 'stats':
        compute_stats()
    elif args.mode == 'extended':
        compute_extended_metrics()
