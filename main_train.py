import os, sys
import argparse
import shutil
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.trainer import Trainer
    
def train(subject_name, exp_cfg, args=None):
    cfg = get_cfg_defaults()
    cfg = update_cfg(cfg, exp_cfg)
    cfg = update_cfg(cfg, data_cfg)
    cfg.cfg_file = data_cfg
    cfg.group = data_type
    cfg.dataset.path = os.path.abspath(cfg.dataset.path)
    cfg.clean = args.clean
    cfg.output_dir = os.path.join(args.exp_dir, data_type, subject_name)
    cfg.wandb_name = args.wandb_name
    
    if 'nerf' in exp_cfg:
        cfg.exp_name = f'{subject_name}_nerf'
        cfg.output_dir = os.path.join(args.exp_dir, data_type, subject_name, 'nerf')
        cfg.ckpt_path = os.path.abspath('./exps/snapshot/male-3-casual/model.tar') # any pretrained nerf model to have a better initialization
    else:
        cfg.exp_name = f'{subject_name}_hybrid'    
        cfg.output_dir = os.path.join(args.exp_dir, data_type, subject_name, 'hybrid')
        cfg.ckpt_path = os.path.join(args.exp_dir, data_type, subject_name, 'nerf', 'model.tar')
    if args.clean:
        shutil.rmtree(cfg.output_dir)
    os.makedirs(os.path.join(cfg.output_dir), exist_ok=True)
    shutil.copy(data_cfg, os.path.join(cfg.output_dir, 'config.yaml'))
    shutil.copy(exp_cfg, os.path.join(cfg.output_dir, 'exp_config.yaml'))
    # creat folders 
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.vis_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.val_vis_dir), exist_ok=True)
    with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    # start training
    trainer = Trainer(config=cfg)
    trainer.fit()

if __name__ == '__main__':
    from lib.utils.config import get_cfg_defaults, update_cfg
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_name', type=str, default = 'SCARF', help='project name')
    parser.add_argument('--exp_dir', type=str, default = './exps', help='exp dir')
    parser.add_argument('--data_cfg', type=str, default = 'configs/data/mpiis/DSC_7157.yml', help='data cfg file path')
    parser.add_argument('--exp_cfg', type=str, default = 'configs/exp/tage_0_nerf.yml', help='exp cfg file path')
    parser.add_argument('--clean', action="store_true", help='delete output dir if exists, if not, the training will be resumed.')    
    args = parser.parse_args()
    # 
    #-- data setting
    data_cfg = args.data_cfg
    data_type = data_cfg.split('/')[-2]
    subject_name = data_cfg.split('/')[-1].split('.')[0]
    
    #-- exp setting
    exp_cfg = args.exp_cfg
    
    # ### ------------- start training 
    train(subject_name, exp_cfg, args)
