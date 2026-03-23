import yaml
import os


class Config:

    def __init__(self, cfg_id, test=False):
        self.id = cfg_id
        cfg_name = './cfg/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        # create dirs
        self.base_dir = '/tmp' if test else 'results'

        self.cfg_dir = '%s/%s' % (self.base_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.result_dir = '%s/results' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.tb_dir = '%s/tb' % self.cfg_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        # common
        self.dataset = cfg.get('dataset', 'h36m')
        self.batch_size = cfg.get('batch_size', 8)
        self.normalize_data = cfg.get('normalize_data', False)
        self.save_model_interval = cfg.get('save_model_interval', 20)
        self.t_his = cfg['t_his']
        self.t_pred = cfg['t_pred']
        self.use_vel = cfg.get('use_vel', False)
        self.vel_anchors = cfg.get('vel_anchors', False)
        self.use_dct = cfg.get('use_dct', False)
        self.n_pre = cfg.get('n_pre', 20)
        self.pred_va = cfg.get('pred_va', False)
        self.mask_ratio = cfg.get('mask_ratio', 0.5)
        self.mask_stage = cfg.get('mask_stage', 'pre')
        self.mask_mode = cfg.get('mask_mode', 'random')
        
        # vae specs
        self.vae_specs = cfg.get('vae_specs', dict())

        # mae
        self.mae_lr = cfg.get('mae_lr', 3.e-4)
        self.num_mae_epoch = cfg.get('num_mae_epoch', 300)
        self.num_mae_epoch_fix = cfg.get('num_mae_epoch_fix', self.num_mae_epoch)
        self.num_mae_data_sample = cfg.get('num_mae_data_sample', 10000)
        self.mae_model_path = os.path.join(self.model_dir, 'mae_%04d.p')
        self.pre_train = cfg.get('pre_train', True)
        
        # fine-tuning
        self.ft = cfg.get('ft', True)
        self.ft_lr = cfg.get('ft_lr', 3.e-4)
        self.num_ft_epoch = cfg.get('num_ft_epoch', 100)
        self.num_ft_epoch_fix = cfg.get('num_ft_epoch_fix', 30)
        self.ft_model_path = os.path.join(self.model_dir, 'ft_mae_%04d.p')
        
        # early stopping
        early_stopping_config = cfg.get('early_stopping', {})
        self.early_stopping_enabled = early_stopping_config.get('enabled', True)
        self.early_stopping_patience = early_stopping_config.get('patience', 25)
        self.early_stopping_min_delta = early_stopping_config.get('min_delta', 1e-4)
        self.early_stopping_monitor = early_stopping_config.get('monitor', 'train_loss')
        self.early_stopping_restore_best_weights = early_stopping_config.get('restore_best_weights', True)
        self.early_stopping_verbose = early_stopping_config.get('verbose', True)
        
        # best model
        self.best_model_pattern = os.path.join(self.model_dir, 'best_model_epoch*.pkl')
