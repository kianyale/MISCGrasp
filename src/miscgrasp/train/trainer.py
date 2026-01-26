import sys

from colorama import *
from torch.amp import autocast, GradScaler
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp')

from src.miscgrasp.dataset.name2dataset import name2dataset
from src.miscgrasp.network.loss import name2loss
from src.miscgrasp.network.wrapper import name2network
from src.miscgrasp.network.metrics import name2metrics
from src.miscgrasp.train.train_tools import to_cuda, Logger
from src.miscgrasp.train.train_valid import ValidationEvaluator
from src.miscgrasp.utils.dataset_utils import *
from src.miscgrasp.asset import vgn_val_scene_names

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

class Trainer:
    default_cfg = {
        "optimizer_type": 'adam',
        "multi_gpus": False,
    }

    def _init_dataset(self):
        if self.cfg['train_dataset_type'].endswith('gen'):
            self.train_data = name2dataset[self.cfg['train_dataset_type']](self.cfg, True, True, False)
            self.train_set = DataLoader(self.train_data, 4, True, num_workers=self.cfg['worker_num'],
                                        collate_fn=efficient_collate_fn2)
            print(Fore.GREEN + f'train set len: {len(self.train_set)}' + Fore.RESET)

            self.val_set_list, self.val_set_names = [], []
            names, val_types, val_cfgs = [], [], []
            for val_set_cfg in self.cfg['val_set_list']:
                name, val_type, val_cfg = val_set_cfg['name'], val_set_cfg['type'], val_set_cfg['cfg']
                names.append(name)
                val_types.append(val_type)
                val_cfgs.append({**val_cfg, **self.cfg})

            for name, val_type, val_cfg in zip(names, val_types, val_cfgs):
                val_set = name2dataset[val_type](val_cfg, False, False, False)
                val_set = DataLoader(val_set, 4, False, num_workers=self.cfg['worker_num'],
                                     collate_fn=efficient_collate_fn2)
                self.val_set_list.append(val_set)
                self.val_set_names.append(name)
            print('[I] ' + Fore.GREEN + f'Val set len: {len(self.val_set_list)}' + Fore.RESET)

    def _init_network(self):
        self.network = name2network[self.cfg['network']](self.cfg).cuda()

        # loss
        self.val_losses = []
        for loss_name in self.cfg['loss']:
            self.val_losses.append(name2loss[loss_name](self.cfg))

        # metrics
        self.val_metrics = []
        for metric_name in self.cfg['val_metric']:
            if metric_name in name2metrics:
                self.val_metrics.append(name2metrics[metric_name](self.cfg))
            else:
                self.val_metrics.append(name2loss[metric_name](self.cfg))

        # we do not support multi gpu
        if self.cfg['multi_gpus']:
            raise NotImplementedError

        else:
            self.train_network = self.network
            self.train_losses = self.val_losses

        if self.cfg['optimizer_type'] == 'adam':
            self.optimizer = Adam(self.network.parameters(), float(self.cfg['lr_cfg']['lr_init']))
        elif self.cfg['optimizer_type'] == 'sgd':
            self.optimizer = SGD(self.network.parameters(), float(self.cfg['lr_cfg']['lr_init']))
        else:
            raise NotImplementedError

        self.val_evaluator = ValidationEvaluator(self.cfg)

        if self.cfg['lr_type'] == 'exp':
            self.scheduler = ExponentialLR(self.optimizer,
                                           float(self.cfg['lr_cfg']['gamma']))
        elif self.cfg['lr_type'] == '1cycle':
            self.scheduler = OneCycleLR(self.optimizer,
                                        total_steps=int(self.cfg['total_step']),
                                        max_lr=float(self.cfg['lr_cfg']['max_lr']),
                                        div_factor=float(self.cfg['lr_cfg']['div_factor']),
                                        final_div_factor=float(self.cfg['lr_cfg']['final_div_factor']))
        else:
            raise NotImplementedError

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}
        if self.cfg['use_amp']:
            self.scaler = GradScaler("cuda")
        if self.cfg['fix_seed']:
            seed = 0
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed_all(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            print("fix seed")
        self.model_name = cfg['name']
        self.model_dir = os.path.join('data/model', cfg['group_name'], cfg['name'])
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        assert self.cfg["key_metric_prefer"] in ['higher', 'lower']
        self.better = lambda x, y: x > y if self.cfg["key_metric_prefer"] == 'higher' else x < y

    def run(self):
        self._init_dataset()
        self._init_network()
        self._init_logger()

        best_para, start_step = self._load_model()
        if self.cfg["key_metric_prefer"] == 'lower' and start_step == 0:
            best_para = 1e6
        train_iter = iter(self.train_set)

        pbar = tqdm(total=self.cfg['total_step'], ncols=100)
        pbar.set_description('   itr')
        pbar.update(start_step)
        for step in range(start_step, self.cfg['total_step']):
            try:
                train_data = next(train_iter)
            except StopIteration:
                del self.train_set
                self.train_set = DataLoader(self.train_data, 4, True, num_workers=self.cfg['worker_num'],
                                        collate_fn=efficient_collate_fn2)
                train_iter = iter(self.train_set)
                train_data = next(train_iter)
            if not self.cfg['multi_gpus']:
                train_data = to_cuda(train_data)
            train_data['step'] = step
            train_data['is_train'] = True

            self.train_network.train()
            self.network.train()
            lr = get_lr(self.optimizer)

            self.optimizer.zero_grad()
            self.train_network.zero_grad()

            log_info = {}

            if self.cfg['use_amp']:
                with autocast("cuda"):
                    outputs = self.train_network(train_data)

                    for loss in self.train_losses:
                        loss_results = loss(outputs, train_data, step)
                        for k, v in loss_results.items():
                            log_info[k] = v

                    loss = 0
                    for k, v in log_info.items():
                        if k.startswith('loss'):
                            loss = loss + torch.mean(v)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.cfg['lr_type'] == 'exp':
                    if step % self.cfg['lr_cfg']['decay_step'] == 0 and lr > self.cfg['lr_cfg']['lr_min']:
                        self.scheduler.step()
                elif self.cfg['lr_type'] == '1cycle':
                    self.scheduler.step()
            else:
                outputs = self.train_network(train_data)

                for loss in self.train_losses:
                    loss_results = loss(outputs, train_data, step)
                    for k, v in loss_results.items():
                        log_info[k] = v

                loss = 0
                for k, v in log_info.items():
                    if k.startswith('loss'):
                        loss = loss + torch.mean(v)

                loss.backward()
                self.optimizer.step()
                if self.cfg['lr_type'] == 'exp':
                    if step % float(self.cfg['lr_cfg']['decay_step']) == 0 and lr > float(self.cfg['lr_cfg']['lr_min']):
                        self.scheduler.step()
                elif self.cfg['lr_type'] == '1cycle':
                    self.scheduler.step()

            if ((step + 1) % self.cfg['train_log_step']) == 0:
                self._log_data(log_info, step + 1, 'train')

            if step == 0 or (step + 1) % self.cfg['val_interval'] == 0 or (step + 1) == self.cfg['total_step']:
            # if step > 120000 and (step == 0 or (step + 1) % self.cfg['val_interval'] == 0 or (step + 1) == self.cfg['total_step']):
            # passs = 0
            # if passs:
                torch.cuda.empty_cache()
                val_results = {}
                val_para = 0
                _tqdm1 = tqdm(total=len(self.val_set_list), ncols=100)
                for vi, val_set in enumerate(self.val_set_list):
                    # tqdm.write(f'\n{vi}\n')
                    _tqdm1.set_description('   val_epoch')
                    val_results_cur, val_para_cur = self.val_evaluator(
                        self.network, self.val_losses + self.val_metrics, val_set, step,
                        self.model_name, val_set_name=self.val_set_names[vi])
                    for k, v in val_results_cur.items():
                        key = f'{self.val_set_names[vi]}-{k}'
                        if key not in val_results:
                            val_results[key] = v
                        else:
                            val_results[key] += v
                    val_para += val_para_cur
                    _tqdm1.update(1)
                _tqdm1.close()

                # average all items 
                for k, v in val_results.items():
                    val_results[k] /= len(self.val_set_list)
                val_para /= len(self.val_set_list)

                if step and self.better(val_para, best_para):  # do not save the first step
                    tqdm.write(
                        Fore.GREEN + f'New best model {self.cfg["key_metric_name"]}: {val_para:.5f} previous {best_para:.5f}'
                        + Fore.RESET)
                    best_para = val_para
                    self.best_pth_fn = os.path.join(self.model_dir, f'best_model_{step + 1}.pth')
                    self._save_model(step + 1, best_para, self.best_pth_fn)
                self._log_data(val_results, step + 1, 'val')
                del val_results, val_para, val_para_cur, val_results_cur

            if (step + 1) % self.cfg['save_interval'] == 0:
                self._save_model(step + 1, best_para)

            pbar.set_postfix(loss=float(loss.detach().cpu().numpy()), lr=lr)
            pbar.update(1)
            del loss, log_info
        pbar.close()

    def _load_model(self):
        best_para, start_step = 0, 0
        if os.path.exists(self.model_dir):
            filter = [i for i in os.listdir(self.model_dir) if i.startswith('model_')]
            if filter:
                filter.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))
                self.pth_fn = os.path.join(self.model_dir, filter[-1])
                tqdm.write(f'{self.pth_fn}')
                # raise NotImplementedError

                checkpoint = torch.load(self.pth_fn)
                best_para = checkpoint['best_para']
                start_step = checkpoint['step']

                self.network.load_state_dict(checkpoint['network_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                tqdm.write(
                    f'==> Resuming from the latest {self.pth_fn} at step {start_step}, with best metric {best_para}.')

        return best_para, start_step

    def _save_model(self, step, best_para, save_fn=None):
        save_fn = os.path.join(self.model_dir, f'model_{step}.pth') if save_fn is None else save_fn
        torch.save({
            'step': step,
            'best_para': best_para,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, save_fn)

    def _init_logger(self):
        self.logger = Logger(self.model_dir)

    def _log_data(self, results, step, prefix='train', verbose=False):
        log_results = {}
        for k, v in results.items():
            if isinstance(v, float) or np.isscalar(v):
                log_results[k] = v
            elif type(v) == np.ndarray:
                log_results[k] = np.mean(v)
            else:
                log_results[k] = np.mean(v.detach().cpu().numpy())
        self.logger.log(log_results, prefix, step, verbose)

