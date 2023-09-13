import os, sys
import argparse
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.visualizer import Visualizer
from lib.utils.config import get_cfg_defaults, update_cfg

if __name__ == '__main__':
    from lib.utils.config import get_cfg_defaults, update_cfg
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_name', type=str, default = 'SCARF', help='project name')
    parser.add_argument('--vis_type', type=str, default = 'capture', help='visualizaiton type')
    parser.add_argument('--model_path', type=str, default = 'exps/mpiis/DSC_7157', help='trained model folder')
    parser.add_argument('--body_model_path', type=str, default = '', help='if specified, then will use this model for body part')
    parser.add_argument('--clothing_model_path', type=str, default = '', help='if specified, then will use this model for clothing part')
    parser.add_argument('--image_size', type=int, default = 512, help='cfg file path')
    parser.add_argument('--video_type', type=str, default ='mp4', help='video type, gif or mp4')
    parser.add_argument('--fps', type=int, default = 10, help='fps for video, suggest 10 for novel view, and 30 for animation')
    parser.add_argument('--saveImages', action="store_true", help='save each image')    
    parser.add_argument('--frame_id', type=int, default = 0, help='frame id for novel view and mesh extraction')
    parser.add_argument('--animation_file', type=str, default = 'data/pixie_radioactive.pkl', help='path for pose data')                        
    args = parser.parse_args()
    
    #-- load config
    model_dir = args.model_path
    data_cfg = os.path.join(model_dir, 'config.yml')
    cfg_file = 'configs/exp/stage_1_hybrid.yml'
    cfg = get_cfg_defaults()
    cfg = update_cfg(cfg, cfg_file)
    cfg = update_cfg(cfg, data_cfg)
    cfg.output_dir = model_dir
    cfg.dataset.path = os.path.abspath(args.model_path)
    args.body_model_path = os.path.join(args.body_model_path, 'model.tar')
    args.clothing_model_path = os.path.join(args.clothing_model_path, 'model.tar')

    # set exp  
    cfg.wandb_name = args.wandb_name
    cfg.cfg_file = cfg_file
    cfg.model_dir = model_dir
    cfg.ckpt_path = os.path.join(model_dir, 'model.tar')
    cfg.clean = False
    if cfg.exp_name is None:
        exp_name = model_dir.split('/')[-1]
        cfg.exp_name = exp_name

    # settings for visualization
    cfg.train.write_summary = False
    cfg.dataset.image_size = args.image_size
    cfg.dataset.white_bg = True
    cfg.use_valid = False
    cfg.dis_threshold = 10 #0.01
    cfg.depth_std = 0.01
    # 
    visualizer = Visualizer(cfg)
    visualizer.run(vistype = args.vis_type, args=args)