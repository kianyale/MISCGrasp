import os
import warnings
from pathlib import Path

from colorama import *

from src.miscgrasp.utils.base_utils import load_cfg
from src.gd.io import *

CONFIG_DIR = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/src/miscgrasp/configs/miscgrasp.yaml')
cfg = load_cfg(CONFIG_DIR)
source = cfg['source']
if source == 'egad':
    DATA_ROOT_DIR = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/datasets/data_misc')
    num_val_pile = 198  # NOTE
    num_val_pack = 174  # NOTE
elif source == 'vgn':
    DATA_ROOT_DIR = Path('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/datasets/data_vgn')
    num_val_pile = 200  # NOTE
    num_val_pack = 200  # NOTE
else:
    raise ValueError(f'Unknown source: {source}')
VGN_TRAIN_ROOT = DATA_ROOT_DIR / 'processed'

if cfg['use_sep']:
    def separate_scenes(scene_names, train):
        if train:
            new_scene_names = []
            for name in scene_names:
                for grp in cfg['gripper_types']:
                    new_scene_names += [name + f'/{grp}']
        else:
            new_scene_names = []
            for name in scene_names:
                for grp in cfg['val_gripper_types']:
                    new_scene_names += [name + f'/{grp}']
        return new_scene_names

    def add_scenes(root, type):
        root = root / type / 'scenes'
        scene_names = []
        scene_names += [f'{source}/train/{type}/{fn.stem}' for fn in root.glob('*')]
        return scene_names

    if os.path.exists(VGN_TRAIN_ROOT):
        vgn_pile_train_scene_names = sorted(add_scenes(VGN_TRAIN_ROOT, 'pile'),
                                            key=lambda x: x.split('/')[3])
        vgn_pack_train_scene_names = sorted(add_scenes(VGN_TRAIN_ROOT, 'packed'),
                                            key=lambda x: x.split('/')[3])

        num_scenes_pile = len(vgn_pile_train_scene_names)
        num_scenes_pack = len(vgn_pack_train_scene_names)

        print('[I] ' + Fore.GREEN + f"Total: {num_scenes_pile + num_scenes_pack} Pile: {num_scenes_pile} Pack: {num_scenes_pack}" + Fore.RESET)

        vgn_val_scene_names = vgn_pile_train_scene_names[-num_val_pile:] + vgn_pack_train_scene_names[-num_val_pack:]
        vgn_train_scene_names = vgn_pile_train_scene_names[:-num_val_pile] + vgn_pack_train_scene_names[:-num_val_pack]
        
        vgn_val_scene_names = separate_scenes(vgn_val_scene_names, False)
        vgn_train_scene_names = separate_scenes(vgn_train_scene_names, True)

        num_train = len(vgn_train_scene_names)
        num_val = len(vgn_val_scene_names)
        print(f'[I] After separating scenes: {Fore.GREEN} Train: {num_train} Validate: {num_val} {Fore.RESET}')

else:
    def add_scenes(root, type):
        root = root / type / 'scenes'
        scene_names = []
        scene_names += [f'{source}/train/{type}/{fn.stem}' for fn in root.glob('*')]
        return scene_names

    if os.path.exists(VGN_TRAIN_ROOT):
        vgn_pile_train_scene_names = sorted(add_scenes(VGN_TRAIN_ROOT, 'pile'),
                                            key=lambda x: x.split('/')[3])
        vgn_pack_train_scene_names = sorted(add_scenes(VGN_TRAIN_ROOT, 'packed'),
                                            key=lambda x: x.split('/')[3])

        num_scenes_pile = len(vgn_pile_train_scene_names)
        num_scenes_pack = len(vgn_pack_train_scene_names)

        num_val_pile = 198  # NOTE
        num_val_pack = 174  # NOTE
        print(Fore.GREEN + f"Total: {num_scenes_pile + num_scenes_pack} pile: {num_scenes_pile} pack: {num_scenes_pack}" +
              Fore.RESET)
        vgn_val_scene_names = vgn_pile_train_scene_names[-num_val_pile:] + vgn_pack_train_scene_names[-num_val_pack:]
        vgn_train_scene_names = vgn_pile_train_scene_names[:-num_val_pile] + vgn_pack_train_scene_names[:-num_val_pack]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    t0 = time.time()
    VGN_PACK_TRAIN_CSV = read_df(VGN_TRAIN_ROOT / 'packed')
    VGN_PILE_TRAIN_CSV = read_df(VGN_TRAIN_ROOT / 'pile')

    print('[I] ' + Fore.GREEN + f"Finished loading csv and val info in {time.time() - t0} s" + Fore.RESET)
    VGN_PACK_TEST_CSV = None
    VGN_PILE_TEST_CSV = None
