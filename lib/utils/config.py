from yacs.config import CfgNode as CN
import argparse
import yaml
import os

cfg = CN()

workdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# project settings
cfg.output_dir = ''
cfg.group = 'test'
cfg.exp_name = None
cfg.device = 'cuda:0'

# load models
cfg.ckpt_path = None
cfg.nerf_ckpt_path = 'exps/mpiis/DSC_7157/model.tar'
cfg.mesh_ckpt_path = ''
cfg.pose_ckpt_path = ''

# ---------------------------------------------------------------------------- #
# Options for SCARF model
# ---------------------------------------------------------------------------- #
cfg.depth_std = 0.
cfg.opt_nerf_pose = True
cfg.opt_beta = True
cfg.opt_mesh = True
cfg.sample_patch_rays = False
cfg.sample_patch_size = 32

# merf model
cfg.mesh_offset_scale = 0.01
cfg.exclude_hand = False
cfg.use_perspective = False
cfg.cam_mode = 'orth'
cfg.use_mesh = True
cfg.mesh_only_body = False
cfg.use_highres_smplx = True
cfg.freqs_tex = 10
cfg.tex_detach_verts = False
cfg.tex_network = 'siren' # siren
cfg.use_nerf = True
cfg.use_fine = True
cfg.share_fine = False
cfg.freqs_xyz = 10
cfg.freqs_dir = 4
cfg.use_view = False
# skinning
cfg.use_outer_mesh = False
cfg.lbs_map = False
cfg.k_neigh = 1
cfg.weighted_neigh = False
cfg.use_valid = False
cfg.dis_threshold = 0.01
## deformation code
cfg.use_deformation = False #'opt_code'
cfg.deformation_type = '' #'opt_code'
cfg.deformation_dim = 0
cfg.latent_dim = 0
cfg.pose_dim = 69
## appearance code
cfg.use_appearance = False
cfg.appearance_type = '' #'opt_code' # or pose
cfg.appearance_dim = 0 # if pose, should be 3
## mesh tex code
cfg.use_texcond = False
cfg.texcond_type = '' #'opt_code' # or pose
cfg.texcond_dim = 0 # if pose, should be 3

# nerf rendering
cfg.n_samples = 64
cfg.n_importance = 32
cfg.n_depth = 0
cfg.chunk = 32*32 # chunk size to split the input to avoid OOM
cfg.query_inside = False
cfg.white_bkgd = True

####--------pose model
cfg.opt_pose = False
cfg.opt_exp = False
cfg.opt_cam = False
cfg.opt_appearance = False
cfg.opt_focal = False

# ---------------------------------------------------------------------------- #
# Options for Training
# ---------------------------------------------------------------------------- #
cfg.train = CN()
cfg.train.optimizer = 'adam'
cfg.train.resume = True
cfg.train.batch_size = 1
cfg.train.max_epochs = 100000
cfg.train.max_steps = 100*1000
cfg.train.lr = 5e-4
cfg.train.pose_lr = 5e-4
cfg.train.tex_lr = 5e-4
cfg.train.geo_lr = 5e-4
cfg.train.decay_steps = [50]
cfg.train.decay_gamma = 0.5
cfg.train.precrop_iters = 400
cfg.train.precrop_frac = 0.5
cfg.train.coarse_iters = 0 # in the begining, only train coarse
# logger
cfg.train.log_dir = 'logs'
cfg.train.log_steps = 10
cfg.train.vis_dir = 'train_images'
cfg.train.vis_steps = 200
cfg.train.write_summary = True
cfg.train.checkpoint_steps = 500
cfg.train.val_steps = 200
cfg.train.val_vis_dir = 'val_images'
cfg.train.eval_steps = 5000

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.type = 'scarf'
cfg.dataset.path = ''
cfg.dataset.white_bg = True
cfg.dataset.subjects = None
cfg.dataset.n_subjects = 1
cfg.dataset.image_size = 512
cfg.dataset.num_workers = 4
cfg.dataset.n_images = 1000000
cfg.dataset.updated_params_path = ''
cfg.dataset.load_gt_pose = True
cfg.dataset.load_normal = False
cfg.dataset.load_perspective = False

# training setting
cfg.dataset.train = CN()
cfg.dataset.train.cam_list = []
cfg.dataset.train.frame_start = 0
cfg.dataset.train.frame_end = 10000
cfg.dataset.train.frame_step = 4
cfg.dataset.val = CN()
cfg.dataset.val.cam_list = []
cfg.dataset.val.frame_start = 400
cfg.dataset.val.frame_end = 500
cfg.dataset.val.frame_step = 4
cfg.dataset.test = CN()
cfg.dataset.test.cam_list = []
cfg.dataset.test.frame_start = 400
cfg.dataset.test.frame_end = 500
cfg.dataset.test.frame_step = 4

# ---------------------------------------------------------------------------- #
# Options for losses
# ---------------------------------------------------------------------------- #
cfg.loss = CN()
cfg.loss.w_rgb = 1.
cfg.loss.w_patch_mrf = 0.#0005
cfg.loss.w_patch_perceptual = 0.#0005
cfg.loss.w_alpha = 0.
cfg.loss.w_xyz = 1.
cfg.loss.w_depth = 0.
cfg.loss.w_depth_close = 0.
cfg.loss.reg_depth = 0.
cfg.loss.use_mse = False
cfg.loss.mesh_w_rgb = 1.
cfg.loss.mesh_w_normal = 0.1
cfg.loss.mesh_w_mrf = 0.
cfg.loss.mesh_w_perceptual = 0.
cfg.loss.mesh_w_alpha = 0.
cfg.loss.mesh_w_alpha_skin = 0.
cfg.loss.skin_consistency_type = 'verts_all_mean'
cfg.loss.mesh_skin_consistency = 0.001
cfg.loss.mesh_inside_mask = 100.
cfg.loss.nerf_hard = 0.
cfg.loss.nerf_hard_scale = 1.
cfg.loss.mesh_reg_wdecay = 0.
# regs
cfg.loss.geo_reg = True
cfg.loss.reg_beta_l1 = 1e-4
cfg.loss.reg_cam_l1 = 1e-4
cfg.loss.reg_pose_l1 = 1e-4
cfg.loss.reg_a_norm = 1e-4
cfg.loss.reg_beta_temp = 1e-4
cfg.loss.reg_cam_temp = 1e-4
cfg.loss.reg_pose_temp = 1e-4
cfg.loss.nerf_reg_dxyz_w = 1e-4
##
cfg.loss.reg_lap_w = 1.0
cfg.loss.reg_edge_w = 10.0
cfg.loss.reg_normal_w = 0.01
cfg.loss.reg_offset_w = 100.
cfg.loss.reg_offset_w_face = 500.
cfg.loss.reg_offset_w_body = 0.
cfg.loss.use_new_edge_loss = False
## new
cfg.loss.pose_reg = False
cfg.loss.background_reg = False
cfg.loss.nerf_reg_normal_w = 0. #0.01

# ---------------------------------------------------------------------------- #
# Options for Body model
# ---------------------------------------------------------------------------- #
cfg.data_dir = os.path.join(workdir, 'data')
cfg.model = CN()
cfg.model.highres_path = os.path.join(cfg.data_dir, 'subdiv_level_1') 
cfg.model.topology_path = os.path.join(cfg.data_dir, 'SMPL_X_template_FLAME_uv.obj') 
cfg.model.smplx_model_path = os.path.join(cfg.data_dir, 'SMPLX_NEUTRAL_2020.npz')
cfg.model.extra_joint_path = os.path.join(cfg.data_dir, 'smplx_extra_joints.yaml')
cfg.model.j14_regressor_path = os.path.join(cfg.data_dir, 'SMPLX_to_J14.pkl')
cfg.model.mano_ids_path = os.path.join(cfg.data_dir, 'MANO_SMPLX_vertex_ids.pkl')
cfg.model.flame_vertex_masks_path = os.path.join(cfg.data_dir, 'FLAME_masks.pkl')
cfg.model.flame_ids_path = os.path.join(cfg.data_dir, 'SMPL-X__FLAME_vertex_ids.npy')
cfg.model.n_shape = 10
cfg.model.n_exp = 10 

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default = os.path.join(os.path.join(root_dir, 'configs/nerf_pl'), 'test.yaml'), help='cfg file path', )
    parser.add_argument('--mode', type=str, default = 'train', help='mode: train, test')
    parser.add_argument('--random_beta', action="store_true", default = False, help='delete folders')
    parser.add_argument('--clean', action="store_true", default = False, help='delete folders')
    parser.add_argument('--debug', action="store_true", default = False, help='debug model')

    args = parser.parse_args()
    print(args, end='\n\n')
    
    cfg = get_cfg_defaults()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file
    cfg.mode = args.mode
    cfg.clean = args.clean
    cfg.debug = args.debug
    cfg.random_beta = args.random_beta
    return cfg
