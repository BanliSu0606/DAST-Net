import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from collections import defaultdict

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8

COLORS = {
    'DAST': '#0077BB',      # 蓝色
    'GCNext': '#EE7733',    # 橙色
    'DAST_light': '#77BBEE',
    'GCNext_light': '#EEAA77',
}
MARKERS = {
    'DAST': 'o',
    'GCNext': 's',
}
# 模型显示名称映射
MODEL_NAMES = {
    'DAST': 'DAST',
    'GCNext': 'GCNext',
}

sys.path.append(os.getcwd())
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_h36m import DatasetH36M
from motion_pred.utils.dataset_humaneva import DatasetHumanEva
from models.DAST import DAST
# from models.GCNext import GCNext
from uniutils import *

def get_model(cfg, model_name, dataset_name):
    if dataset_name == 'humaneva':
        joints_num = 14
    else:
        joints_num = 16
    
    if not hasattr(cfg, 'vae_specs'):
        cfg.vae_specs = {}
    cfg.vae_specs['model_name'] = model_name
    
    if model_name == 'DAST':
        model = DAST(3, 256, [64, 128, 256], [2, 3, 5], 
                   cfg.t_his, cfg.t_pred, joints_num, 0.1, 8).to(device)
    # elif model_name == 'GCNext':
    #     model = GCNext(input_n=cfg.t_his, pred_n=cfg.t_pred, motion_dim=joints_num*3, dyna_idx=[0, 2]).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def load_model(cfg, model_name, dataset_name, use_best=True):
    model = get_model(cfg, model_name, dataset_name)
    
    if dataset_name == 'h36m':
        if model_name == 'DAST':
            model_dir = os.path.join('results', 'h36m_25_100', 'models')
        else:  # GCNext
            model_dir = os.path.join('results', 'h36m_25_100_gcnext', 'models')
    else:  # humaneva
        if model_name == 'DAST':
            model_dir = os.path.join('results', 'humaneva_25_100', 'models')
        else:  # GCNext
            model_dir = os.path.join('results', 'humaneva_25_100_gcnext', 'models')
    
    print(f'Looking for models in: {model_dir}')
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    import glob
    
    if use_best:
        best_model_files = glob.glob(os.path.join(model_dir, 'best_model_epoch*.pkl'))
        if best_model_files:
            latest_best = max(best_model_files, key=lambda x: int(x.split('epoch')[1].split('.')[0]))
            model_path = latest_best
        else:
            possible_epochs = [cfg.num_mae_epoch, 200, 150, 100, 50, 70, 80, 120, 164]
            model_path = None
            for epoch in possible_epochs:
                test_path = os.path.join(model_dir, f'model_epoch{epoch}.pkl')
                if os.path.exists(test_path):
                    model_path = test_path
                    break
            if model_path is None:
                pkl_files = glob.glob(os.path.join(model_dir, '*.pkl'))
                if pkl_files:
                    model_path = pkl_files[0]
                else:
                    raise FileNotFoundError(f"No model file found in {model_dir}")
    else:
        model_path = os.path.join(model_dir, f'model_epoch{cfg.num_mae_epoch}.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f'Loading {model_name} from: {model_path}')
    model_cp = pickle.load(open(model_path, "rb"))
    
    print(f"Checkpoint keys: {model_cp.keys()}")
    if 'meta' in model_cp:
        print(f"Meta keys: {model_cp['meta'].keys() if isinstance(model_cp['meta'], dict) else 'not a dict'}")
    
    model.load_state_dict(model_cp['model_dict'])
    model.to(device)
    model.eval()
    
    return model, model_cp

def get_predictions(model, data, cfg, dct, idct, device, model_cp=None, use_dct=True):
    if cfg.dataset in ['humaneva', 'h36m']:
        traj_np = data
        traj_np = traj_np[..., 1:, :]
    else:
        traj_np = data
    
    traj = torch.tensor(traj_np, device=device, dtype=torch.float)
    traj_flat = traj.reshape(traj.shape[0], traj.shape[1], -1)
    
    if use_dct and cfg.use_dct:
        traj_dct = torch.matmul(dct[:cfg.n_pre], traj_flat)
        X = traj_dct
    else:
        X = traj_flat
    
    with torch.no_grad():
        if cfg.vae_specs['model_name'] == 'DAST':
            Y = model.forward(X[:, :cfg.t_his, :])
        else:  # GCNext

            Y = model.forward(X[:, :cfg.t_his, :])
    
    if use_dct and cfg.use_dct:
        Y = torch.matmul(idct[:, :cfg.n_pre], Y)
    
    Y = Y.cpu().numpy()
    
    if model_cp is not None and 'meta' in model_cp:
        meta = model_cp['meta']
        if 'mean' in meta and meta['mean'] is not None and 'std' in meta and meta['std'] is not None:
            if isinstance(meta['mean'], torch.Tensor):
                mean = meta['mean'].cpu().numpy()
                std = meta['std'].cpu().numpy()
            else:
                mean = meta['mean']
                std = meta['std']
            Y = Y * std + mean
        else:
            if hasattr(dataset, 'mean') and dataset.mean is not None:
                Y = Y * dataset.std + dataset.mean
    
    return Y

def compute_mpjpe(pred, gt):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=-1)  # (T, J)
    return np.mean(dist, axis=-1)  # (T,)

def compute_action_mpjpe(dataset_class, model, cfg, action_name, dct, idct, device, model_cp=None):
    try:
        action_dataset = dataset_class(
            cfg.data, cfg.t_his, cfg.t_pred, 
            actions=action_name, 
            use_vel=cfg.use_vel
        )
    except Exception as e:
        print(f"  Error creating dataset for action '{action_name}': {e}")
        return np.zeros(cfg.t_pred)
    
    try:
        data_gen = action_dataset.iter_generator(step=cfg.t_his)
        first_sample = next(iter(data_gen))
    except (StopIteration, Exception) as e:
        print(f"  No samples for action: {action_name}")
        return np.zeros(cfg.t_pred)
    
    all_mpjpe_frames = []
    data_gen = action_dataset.iter_generator(step=cfg.t_his)
    
    sample_count = 0
    for data in data_gen:
        sample_count += 1
        gt = data[:, cfg.t_his:, :, :]
        
        Y = get_predictions(model, data, cfg, dct, idct, device, model_cp)
        
        if cfg.dataset == 'humaneva':
            num_joints = 14
        elif cfg.dataset == 'h36m':
            num_joints = 16
        else:
            num_joints = gt.shape[-2] - 1
        
        Y = Y.reshape(Y.shape[0], cfg.t_pred, num_joints, 3)
        
        root = np.zeros((Y.shape[0], Y.shape[1], 1, 3))
        pred_full = np.concatenate([root, Y], axis=2)
        
        gt_np = gt.numpy() if hasattr(gt, 'numpy') else gt
        mpjpe_per_frame = compute_mpjpe(pred_full[0], gt_np[0])
        all_mpjpe_frames.append(mpjpe_per_frame)
        
        if sample_count == 1:
            print(f"  Sample 1 - Pred shape: {pred_full.shape}, GT shape: {gt_np.shape}")
            print(f"  Sample 1 - MPJPE first 5 frames: {mpjpe_per_frame[:5]}")
        
        if sample_count >= 30:
            break
    
    if not all_mpjpe_frames:
        print(f"  No valid samples for action: {action_name}")
        return np.zeros(cfg.t_pred)
    
    avg_mpjpe = np.mean(np.array(all_mpjpe_frames), axis=0)
    print(f"  Average MPJPE over {sample_count} samples: {np.mean(avg_mpjpe):.4f}")
    return avg_mpjpe

def get_actions_for_dataset(dataset_name):
    if dataset_name == 'h36m':
        return ['Directions', 'Discussion', 'Eating', 'Greeting', 
                'Phoning', 'Posing', 'Purchases', 'Sitting', 
                'SittingDown', 'Smoking', 'Photo', 'Waiting', 
                'Walking', 'WalkDog', 'WalkTogether']
    elif dataset_name == 'humaneva':
        return ['Walking', 'Jog', 'Box', 'Gestures', 'ThrowCatch']
    else:
        return []

def plot_action_mpjpe_curves(all_results, dataset_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("DEBUG: Checking results for each action")
    print("="*60)
    
    valid_actions = []
    for action, results in all_results.items():
        has_valid_data = False
        action_debug_info = []
        
        for model_key in ['DAST', 'GCNext']:
            if results.get(model_key) is not None and len(results[model_key]) > 0:
                mean_val = np.mean(results[model_key])
                action_debug_info.append(f"{MODEL_NAMES[model_key]}: mean={mean_val:.6f}, len={len(results[model_key])}")
                if mean_val > 1e-6:
                    has_valid_data = True
            else:
                action_debug_info.append(f"{MODEL_NAMES[model_key]}: No data or empty array")
        
        print(f"\nAction: {action}")
        for info in action_debug_info:
            print(f"  {info}")
        print(f"  Valid: {has_valid_data}")
        
        if has_valid_data:
            valid_actions.append(action)
    
    if not valid_actions:
        print("\nNo valid results to plot! (all MPJPE values are near zero)")
        return
    
    print(f"\nValid actions to plot: {valid_actions}")
    print("="*60 + "\n")
    
    t_pred = None
    for action in valid_actions:
        for model_key in ['DAST', 'GCNext']:
            if all_results[action].get(model_key) is not None:
                if len(all_results[action][model_key]) > 0:
                    t_pred = len(all_results[action][model_key])
                    break
        if t_pred is not None:
            break
    
    if t_pred is None:
        print("No valid MPJPE data found!")
        return
    
    for action in valid_actions:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        
        plotted_models = []
        for model_key in ['DAST', 'GCNext']:
            display_name = MODEL_NAMES[model_key]
            
            if model_key in all_results[action] and all_results[action][model_key] is not None:
                mpjpe = all_results[action][model_key]
                
                if len(mpjpe) > 0:
                    if np.all(np.abs(mpjpe) < 1e-8):
                        print(f"  Warning: {display_name} for {action} has all near-zero values, skipping")
                        continue
                    
                    mpjpe_mm = mpjpe * 1000
                    
                    ax.plot(range(1, len(mpjpe_mm) + 1), mpjpe_mm, 
                           label=display_name, 
                           color=COLORS[display_name],
                           marker=MARKERS[display_name],
                           linewidth=3,
                           markersize=10,
                           markevery=max(1, len(mpjpe_mm)//15),
                           linestyle='-')
                    plotted_models.append(display_name)
                    
                    print(f"  Plotted {display_name} for {action}, mean MPJPE: {np.mean(mpjpe_mm):.2f} mm")
        
        if not plotted_models:
            print(f"  No data to plot for {action}, skipping...")
            plt.close()
            continue
        
        ax.set_xlabel('Prediction Horizon (frames)', fontsize=18, fontweight='bold')
        ax.set_ylabel('MPJPE (mm)', fontsize=18, fontweight='bold')
        ax.set_title(f'{dataset_name.upper()} - {action}', fontsize=20, fontweight='bold')
        
        legend = ax.legend(loc='best', fontsize=16, frameon=True, 
                          fancybox=True, shadow=True, edgecolor='black')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_linewidth(1.5)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        ax.set_xlim([0, t_pred + 2])
        y_min, y_max = ax.get_ylim()
        if y_max > 0:
            ax.set_ylim([0, y_max * 1.05])  # 从0开始
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        ax.grid(True, which='minor', alpha=0.1, linestyle=':', linewidth=0.5)
        ax.minorticks_on()
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'{dataset_name}_{action}_mpjpe_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {save_path}')
        
        pdf_path = os.path.join(save_dir, f'{dataset_name}_{action}_mpjpe_curves.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f'Saved: {pdf_path}')
        
        plt.close()
    
    n_actions = len(valid_actions)
    if n_actions > 0:
        n_cols = min(2, n_actions)
        n_rows = (n_actions + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12*n_cols, 8*n_rows), dpi=300)
        
        if n_actions == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, action in enumerate(valid_actions[:4]):  # 只显示前4个
            ax = axes[idx]
            plotted_models = []
            
            for model_key in ['DAST', 'GCNext']:
                display_name = MODEL_NAMES[model_key]
                if model_key in all_results[action] and all_results[action][model_key] is not None:
                    mpjpe = all_results[action][model_key]
                    
                    if len(mpjpe) > 0 and not np.all(np.abs(mpjpe) < 1e-8):
                        mpjpe_mm = mpjpe * 1000
                        ax.plot(range(1, len(mpjpe_mm) + 1), mpjpe_mm, 
                               label=display_name,
                               color=COLORS[display_name],
                               marker=MARKERS[display_name],
                               linewidth=2.5,
                               markersize=8,
                               markevery=max(1, len(mpjpe_mm)//12))
                        plotted_models.append(display_name)
            
            if plotted_models:
                ax.set_xlabel('Prediction Frame', fontsize=14, fontweight='bold')
                ax.set_ylabel('MPJPE (mm)', fontsize=14, fontweight='bold')
                ax.set_title(f'{action}', fontsize=16, fontweight='bold')
                legend = ax.legend(loc='best', fontsize=12, frameon=True, fancybox=True)
                legend.get_frame().set_alpha(0.9)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlim([0, t_pred + 2])
                y_min, y_max = ax.get_ylim()
                if y_max > 0:
                    ax.set_ylim([0, y_max * 1.05])
                ax.tick_params(axis='both', which='major', labelsize=12)
        
        for idx in range(min(n_actions, 4), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{dataset_name.upper()} - MPJPE vs Prediction Horizon', fontsize=22, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'{dataset_name}_all_actions_mpjpe_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {save_path}')
        
        pdf_path = os.path.join(save_dir, f'{dataset_name}_all_actions_mpjpe_curves.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f'Saved: {pdf_path}')
        
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='h36m', 
                        choices=['h36m', 'humaneva'],
                        help='Dataset to evaluate')
    parser.add_argument('--cfg', type=str, default='h36m_25_100',
                        help='Config name')
    parser.add_argument('--data', type=str, default='test',
                        help='Data split')
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--use_best', action='store_true', default=True,
                        help='Use best model')
    args = parser.parse_args()
    
    global device, dataset
    device = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    cfg = Config(args.cfg)
    cfg.data = args.data
    cfg.dataset = args.dataset
    
    if not hasattr(cfg, 'vae_specs'):
        cfg.vae_specs = {}
    
    print(f'Result directory: {cfg.result_dir}')
    print(f"Current working directory: {os.getcwd()}")
    
    dct, idct = get_dct_matrix(cfg.t_his + cfg.t_pred)
    dct = dct.float().to(device)
    idct = idct.float().to(device)
    
    if args.dataset == 'h36m':
        dataset_cls = DatasetH36M
    else:
        dataset_cls = DatasetHumanEva
    
    dataset = dataset_cls(args.data, cfg.t_his, cfg.t_pred, 
                         actions='all', use_vel=cfg.use_vel)
    
    actions = get_actions_for_dataset(args.dataset)
    print(f'Actions to evaluate: {actions}')
    
    all_results = {action: {} for action in actions}
    
    for action in actions:
        print(f'\n{"="*50}')
        print(f'Evaluating action: {action}')
        print(f'{"="*50}')
        
        try:
            print('Loading DAST model...')
            dast_model, dast_cp = load_model(cfg, 'DAST', args.dataset, args.use_best)
            cfg.vae_specs['model_name'] = 'DAST'
            dast_mpjpe = compute_action_mpjpe(
                dataset_cls, dast_model, cfg, action, dct, idct, device, dast_cp
            )
            all_results[action]['DAST'] = dast_mpjpe
            
            if len(dast_mpjpe) > 0 and np.mean(dast_mpjpe) > 0.001:
                print(f'DAST MPJPE (avg over frames): {np.mean(dast_mpjpe)*1000:.2f} mm')
            else:
                print(f'DAST MPJPE: No valid predictions (mean={np.mean(dast_mpjpe) if len(dast_mpjpe)>0 else "empty"})')
        except Exception as e:
            print(f'Error loading DAST model: {e}')
            import traceback
            traceback.print_exc()
            all_results[action]['DAST'] = np.array([])
        
        try:
            print('Loading GCNext model...')
            gcnext_model, gcnext_cp = load_model(cfg, 'GCNext', args.dataset, args.use_best)
            cfg.vae_specs['model_name'] = 'GCNext'
            gcnext_mpjpe = compute_action_mpjpe(
                dataset_cls, gcnext_model, cfg, action, dct, idct, device, gcnext_cp
            )
            all_results[action]['GCNext'] = gcnext_mpjpe
            
            if len(gcnext_mpjpe) > 0 and np.mean(gcnext_mpjpe) > 0.001:
                print(f'GCNext MPJPE (avg over frames): {np.mean(gcnext_mpjpe)*1000:.2f} mm')
            else:
                print(f'GCNext MPJPE: No valid predictions (mean={np.mean(gcnext_mpjpe) if len(gcnext_mpjpe)>0 else "empty"})')
        except Exception as e:
            print(f'Error loading GCNext model: {e}')
            import traceback
            traceback.print_exc()
            all_results[action]['GCNext'] = np.array([])
    
    save_dir = os.path.join(cfg.result_dir, 'mpjpe_curves')
    os.makedirs(save_dir, exist_ok=True)
    
    save_dict = {}
    for action in actions:
        for model_name in ['DAST', 'GCNext']:
            if all_results[action].get(model_name) is not None:
                if len(all_results[action][model_name]) > 0 and np.mean(all_results[action][model_name]) > 0.001:
                    key = f'{action}_{model_name}'
                    save_dict[key] = all_results[action][model_name]
    
    if save_dict:
        np.savez(os.path.join(save_dir, f'{args.dataset}_mpjpe_results.npz'), **save_dict)
        print(f'\nResults saved to: {save_dir}')
    
    plot_action_mpjpe_curves(all_results, args.dataset, save_dir)
    
    print('\n' + '='*60)
    print('SUMMARY - Average MPJPE per action (averaged over all frames)')
    print('='*60)
    print(f"{'Action':<15} {'DAST':<15} {'GCNext':<15} {'Difference':<15}")
    print('-'*60)
    
    for action in actions:
        dast_val = np.mean(all_results[action]['DAST']) * 1000 if (all_results[action]['DAST'] is not None and len(all_results[action]['DAST']) > 0 and np.mean(all_results[action]['DAST']) > 0.001) else float('nan')
        gc_val = np.mean(all_results[action]['GCNext']) * 1000 if (all_results[action]['GCNext'] is not None and len(all_results[action]['GCNext']) > 0 and np.mean(all_results[action]['GCNext']) > 0.001) else float('nan')
        
        diff = dast_val - gc_val if not np.isnan(dast_val) and not np.isnan(gc_val) else float('nan')
        if not np.isnan(dast_val) and not np.isnan(gc_val):
            print(f'{action:<15} {dast_val:<15.2f} {gc_val:<15.2f} {diff:<15.2f}')
        elif not np.isnan(dast_val):
            print(f'{action:<15} {dast_val:<15.2f} {"N/A":<15} {"N/A":<15}')
        elif not np.isnan(gc_val):
            print(f'{action:<15} {"N/A":<15} {gc_val:<15.2f} {"N/A":<15}')
        else:
            print(f'{action:<15} {"N/A":<15} {"N/A":<15} {"N/A":<15}')
    
    print('='*60)

if __name__ == '__main__':
    main()