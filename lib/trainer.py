import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import wandb
from pytorch3d.loss import (
    mesh_edge_loss, 
)
from .models.scarf import SCARF
from .utils import util, rotation_converter, lossfunc
from .utils.config import cfg
from .datasets import build_datasets

def cache_data(dataset):
    '''read pose data from original path, save cache data for fast loading
    '''
    n_frames = len(dataset)
    logger.info(f'caching data...')

    pose_dict = {}
    cam_dict = {}
    exp_dict = {}
    beta_dict = {}
    init_beta = []
    for i in tqdm(range(n_frames)):
        sample = dataset[i]
        frame_id = sample['frame_id']
        name = sample['name']
        if 'cam_id' in sample.keys():
            cam_id = sample['cam_id']
            name = f'{name}_{cam_id}'
        exp_dict[f'{name}_exp_{frame_id}'] = sample['exp'][:10][None,...]
        cam_dict[f'{name}_cam_{frame_id}'] = sample['cam'][None,...]
        init_beta.append(sample['beta'][:10])
        pose_matrix = sample['full_pose']
        pose_axis = rotation_converter.batch_matrix2axis(pose_matrix) + 1e-8
        pose_dict[f'{name}_pose_{frame_id}'] = pose_axis.clone()[None,...]
    init_beta = torch.stack(init_beta)
    init_beta = init_beta.mean(0)[None,...]
    beta_dict[f'{name}_beta'] = init_beta
    beta_dict['beta'] = init_beta
    torch.save(pose_dict, dataset.pose_cache_path)
    torch.save(cam_dict, dataset.cam_cache_path)
    torch.save(exp_dict, dataset.exp_cache_path)
    torch.save(beta_dict, dataset.beta_cache_path)

class PoseModel(nn.Module):
    def __init__(self, dataset, optimize_cam=False, use_perspective=False, 
                    use_deformation=False, deformation_dim=0):
        super(PoseModel, self).__init__()
        self.subject_id = dataset.subject_id        
        # first load cache data
        pose_dict = torch.load(dataset.pose_cache_path)
        for key in pose_dict.keys():
            self.register_parameter(key, torch.nn.Parameter(pose_dict[key]))
        self.pose_dict = pose_dict
        
        self.optimize_cam = optimize_cam
        self.use_deformation = use_deformation
        if self.optimize_cam:
            cam_dict = torch.load(dataset.cam_cache_path)
            for key in cam_dict.keys():
                self.register_parameter(key, torch.nn.Parameter(cam_dict[key]))
            self.cam_dict = cam_dict
            if use_perspective:
                self.register_parameter('focal', torch.nn.Parameter(self.cam_dict[key][:,0]))
        if self.use_deformation:
            deformation_dict = torch.load(dataset.cam_cache_path)
            for key in deformation_dict.keys():
                self.register_parameter(key.replace('cam', 'deformation'), torch.nn.Parameter(torch.zeros([1,deformation_dim])))
            self.deformation_dict = deformation_dict
        self.use_perspective = use_perspective

    def forward(self, batch):
        # return poses of given frame_ids
        name = self.subject_id
        if 'cam_id' in batch.keys():
            cam_id = batch['cam_id']
            names = [f'{name}_{cam}' for cam in cam_id]
        else:
            names = [name]*len(batch['frame_id'])
        frame_ids = batch['frame_id']
        batch_size = len(frame_ids)
        batch_pose = torch.cat([getattr(self, f'{names[i]}_pose_{frame_ids[i]}') for i in range(batch_size)])
        batch_pose = rotation_converter.batch_axis2matrix(batch_pose.reshape(-1, 3)).reshape(batch_size, 55, 3, 3)
        batch['init_full_pose'] = batch['full_pose'].clone()
        batch['full_pose'] = batch_pose
        batch['full_pose'][:,22] = torch.eye(3).to(batch_pose.device)[None,...].expand(batch_size, -1, -1)
        if self.optimize_cam:
            batch['init_cam'] = batch['cam'].clone()
            batch['cam'] = torch.cat([getattr(self, f'{names[i]}_cam_{frame_ids[i]}') for i in range(batch_size)])
            if self.use_perspective:
                batch['cam'][:,0] = self.focal
        if self.use_deformation:
            batch['deformation_code'] = torch.cat([getattr(self, f'{names[i]}_deformation_{frame_ids[i]}') for i in range(batch_size)])
        return batch
    
class Trainer(torch.nn.Module):
    def __init__(self, config=None):
        super(Trainer, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        
        self.device = cfg.device
        self.batch_size = self.cfg.train.batch_size
        self.image_size = self.cfg.dataset.image_size

        # load model
        self.prepare_data()
        init_beta = self.train_dataset[0]['beta']
        self.model = SCARF(self.cfg, device=self.device, init_beta=init_beta).to(self.device)
        self.posemodel = PoseModel(dataset=self.train_dataset, optimize_cam=self.cfg.opt_cam, use_perspective=self.cfg.use_perspective,
                                    use_deformation = self.cfg.use_deformation & (self.cfg.deformation_type=='opt_code'), deformation_dim=self.cfg.deformation_dim,
                                    ).to(self.device)
        self.configure_optimizers()
        self.load_checkpoint() 
        ### loss
        if self.cfg.loss.mesh_w_mrf > 0. or self.cfg.loss.w_patch_mrf:
            self.mrf_loss = lossfunc.IDMRFLoss().to(self.device)
        ### logger
        self.savefolder = self.cfg.output_dir
        logger.add(os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, 'train.log'))
        logfolder = os.path.join(self.cfg.output_dir, self.cfg.train.log_dir)
        group = self.cfg.group
        resume = None if self.cfg.clean else "allow"
        wandb.init(
            id = f'{self.cfg.group}_{self.cfg.exp_name}',
            resume = resume,
            project=self.cfg.wandb_name, 
            name = self.cfg.exp_name,
            save_code = True,
            group = group,
            dir = logfolder)

    def configure_optimizers(self):
        if self.cfg.opt_beta:
            parameters = [{'params': [self.model.beta], 'lr': self.cfg.train.lr}]
        else:
            parameters = []
        if self.cfg.use_nerf:
            parameters.append({'params': self.model.nerf.parameters(), 'lr': self.cfg.train.lr})
            if self.cfg.use_fine and not self.cfg.share_fine:
                parameters.append({'params': self.model.nerf_fine.parameters(), 'lr': self.cfg.train.lr})
            if self.cfg.use_deformation:
                parameters.append({'params': self.model.derf.parameters(), 'lr': self.cfg.train.lr})
        if self.cfg.use_mesh and self.cfg.opt_mesh:
            parameters.append({'params': self.model.mlp_tex.parameters(), 'lr': self.cfg.train.tex_lr})
            parameters.append({'params': self.model.mlp_geo.parameters(), 'lr': self.cfg.train.geo_lr})
            if self.cfg.use_outer_mesh:
                parameters.append({'params': self.model.mlp_geo_outer.parameters(), 'lr': self.cfg.train.geo_lr*0.1 })
        if self.cfg.opt_pose:
            parameters.append({'params': self.posemodel.parameters(), 'lr': self.cfg.train.pose_lr})
        self.optimizer = torch.optim.Adam(params=parameters)   
        self.decay_steps = [1000, 5000, 10000, 50000]; self.decay_gamma = 0.5
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.decay_steps, gamma=self.decay_gamma)
       
    def model_dict(self):
        current_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step
        }
        if self.cfg.opt_pose:
            current_dict['pose'] = self.posemodel.state_dict()
        return current_dict

    def load_checkpoint(self):
        self.global_step = 0
        model_dict = self.model_dict()
        # resume training, including model weight, opt, steps
        if self.cfg.train.resume and os.path.exists(os.path.join(self.cfg.output_dir, 'model.tar')):
            checkpoint = torch.load(os.path.join(self.cfg.output_dir, 'model.tar'))
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    if isinstance(checkpoint[key], dict):
                        util.copy_state_dict(model_dict[key], checkpoint[key])
            self.global_step = checkpoint['global_step']
            logger.info(f"resume training from {os.path.join(self.cfg.output_dir, 'model.tar')}")
            logger.info(f"training start from step {self.global_step}")
            
        elif os.path.exists(self.cfg.ckpt_path):
            checkpoint = torch.load(self.cfg.ckpt_path)
            key = 'model'
            util.copy_state_dict(model_dict[key], checkpoint[key])
            key = 'pose'
            util.copy_state_dict(model_dict[key], checkpoint[key])
            # if specify nerf ckpt, load and overwrite nerf weight
            if os.path.exists(self.cfg.nerf_ckpt_path):
                checkpoint = torch.load(self.cfg.nerf_ckpt_path)
                for param_name in model_dict['model'].keys():
                    if 'nerf' in param_name:
                        model_dict['model'][param_name].copy_(checkpoint['model'][param_name])
            # if specify mesh ckpt, load and overwrite mesh weight
            if os.path.exists(self.cfg.mesh_ckpt_path):
                checkpoint = torch.load(self.cfg.mesh_ckpt_path)
                for param_name in model_dict['model'].keys():
                    if 'mlp_geo' in param_name or 'mlp_tex' in param_name or 'beta' in param_name:
                        if param_name in checkpoint['model']:
                            model_dict['model'][param_name].copy_(checkpoint['model'][param_name])
            # if specify pose ckpt, load and overwrite pose weight
            if os.path.exists(self.cfg.pose_ckpt_path):
                checkpoint = torch.load(self.cfg.pose_path)
                util.copy_state_dict(model_dict['pose'], checkpoint['pose'])
        else:
            logger.info('model path not found, start training from scratch')

    def training_step(self, batch, batch_nb, test=False):
        self.model.train()
        util.move_dict_to_device(batch, self.device)
        #-- update pose
        if not test:
            batch = self.posemodel(batch)
        #-- model: smplx parameters, rendering
        batch['global_step'] = self.global_step
        opdict = self.model(batch, train=True)
        #-- loss
        #### ----------------------- Losses
        losses = {}
        if self.cfg.use_nerf:
            ## regs for deformation
            if self.cfg.use_deformation and 'd_xyz' in batch.keys():
                losses['nerf_reg_dxyz'] = (batch['d_xyz']**2).sum(-1).mean()*self.cfg.loss.nerf_reg_dxyz_w
            gt_mask = batch['mask_sampled']
            if self.cfg.use_mesh:
                gt_mask = batch['cloth_mask_sampled']
            if self.cfg.loss.nerf_reg_normal_w > 0.:
                points_normal, points_neighbs_normal = self.model.canonical_normal(use_fine=False)
                losses['nerf_reg_normal'] = self.cfg.loss.nerf_reg_normal_w * F.mse_loss(points_normal, points_neighbs_normal)
                if self.cfg.use_fine and self.cfg.n_importance > 0 and not self.cfg.share_fine:
                    points_normal_fine, points_neighbs_normal_fine = self.model.canonical_normal(use_fine=True)
                    losses['nerf_reg_normal'] += self.cfg.loss.nerf_reg_normal_w * F.mse_loss(points_normal_fine, points_neighbs_normal_fine)
            losses['rgb'] = lossfunc.huber(opdict['rgbs']*batch['mask_sampled'], batch['image_sampled']*batch['mask_sampled'])*self.cfg.loss.w_rgb
            losses['alpha'] = lossfunc.huber(opdict['alphas'], gt_mask)*self.cfg.loss.w_alpha
            if self.cfg.use_fine and self.cfg.n_importance > 0 and not self.cfg.share_fine:
                losses['rgb_fine'] = lossfunc.huber(opdict['rgbs_fine']*batch['mask_sampled'], batch['image_sampled']*batch['mask_sampled'])*self.cfg.loss.w_rgb
                losses['alpha_fine'] = lossfunc.huber(opdict['alphas_fine'], gt_mask)*self.cfg.loss.w_alpha
                if self.cfg.sample_patch_rays:
                    patch_size = self.cfg.sample_patch_size
                    rgb_patch = opdict['rgbs_fine'][:,:patch_size**2] 
                    rgb_patch = opdict['rgbs_fine'][:,:patch_size**2] .reshape(-1, patch_size, patch_size, 3).permute(0,3,1,2)
                    rgb_patch_gt = batch['image_sampled'][:,:patch_size**2] .reshape(-1, patch_size, patch_size, 3).permute(0,3,1,2)
                    mask_patch = gt_mask[:,:patch_size**2].reshape(-1, patch_size, patch_size, 1).permute(0,3,1,2)
                    if self.cfg.loss.w_patch_mrf > 0.:
                        losses['nerf_patch_mrf'] = self.mrf_loss(rgb_patch*mask_patch, rgb_patch_gt*mask_patch)*self.cfg.loss.w_patch_mrf
                             
        if self.cfg.use_mesh and self.cfg.opt_mesh:
            if self.cfg.loss.geo_reg:
                mesh = opdict['mesh']
                offset = opdict['mesh_offset']
                new_verts = self.model.verts[None,...].expand(offset.shape[0], -1, -1) + offset
                batch_size = offset.shape[0] 
                losses["reg_offset"] = (opdict['offset']**2).sum(-1).mean()*self.cfg.loss.reg_offset_w
                exclude_idx = self.model.part_idx_dict['exclude']
                losses["reg_offset_fh"] = (opdict['offset'][:,exclude_idx]**2).sum(-1).mean()*self.cfg.loss.reg_offset_w_face
                losses["reg_offset_hand"] = (opdict['offset'][:,self.model.part_idx_dict['hand']]**2).sum(-1).mean()*self.cfg.loss.reg_offset_w_face*9.
                if self.cfg.loss.use_new_edge_loss:
                    offset = opdict['mesh_offset']
                    new_verts = self.model.verts[None,...].expand(offset.shape[0], -1, -1) + offset
                    losses["reg_edge"] = lossfunc.relative_edge_loss(new_verts, self.model.verts[None,...].expand(new_verts.shape[0], -1, -1), vertices_per_edge=self.model.verts_per_edge)*self.cfg.loss.reg_edge_w*100.
                else:
                    losses["reg_edge"] = mesh_edge_loss(mesh)*self.cfg.loss.reg_edge_w
                  
            if self.cfg.use_nerf:
                losses['mesh_skin_mask'] =lossfunc.huber(batch['skin_mask']*opdict['mesh_mask'], batch['skin_mask'])*self.cfg.loss.mesh_w_alpha_skin
                losses['mesh_inside_mask'] = (torch.relu(opdict['mesh_mask'] - batch['mask'])).abs().mean()*self.cfg.loss.mesh_inside_mask
                losses['mesh_image'] = lossfunc.huber(batch['skin_mask']*opdict['mesh_image'], batch['skin_mask']*batch['image'])*self.cfg.loss.mesh_w_rgb    
                losses['mesh_mask'] = lossfunc.huber(opdict['mesh_mask'], batch['mask'])*self.cfg.loss.mesh_w_alpha
                # skin color consistency
                tex = opdict['mesh_tex']
                if self.cfg.loss.skin_consistency_type == 'verts_all_mean':
                    losses['mesh_skin_consistency'] = (tex - tex.detach().mean(1)[:,None,:]).abs().mean()*self.cfg.loss.mesh_skin_consistency
                if self.cfg.loss.skin_consistency_type == 'verts_hand_mean':
                    all_idx = list(range(tex.shape[1]))
                    hand_idx = self.model.part_idx_dict['hand']
                    non_hand_idx = [i for i in all_idx if i not in self.model.part_idx_dict['exclude']]
                    losses['mesh_skin_consistency'] = (tex[:,non_hand_idx] - tex[:,hand_idx].detach().mean(1)[:,None,:]).abs().mean()*self.cfg.loss.mesh_skin_consistency
                if self.cfg.loss.skin_consistency_type == 'render_nonskin_mean':
                    losses['mesh_skin_consistency'] = (batch['cloth_mask']*(opdict['mesh_image'] - tex.detach().mean(1)[:,:,None,None]).abs()).mean()*self.cfg.loss.mesh_skin_consistency                
                if self.cfg.loss.skin_consistency_type == 'render_hand_mean':
                    hand_idx = self.model.part_idx_dict['hand']
                    losses['mesh_skin_consistency'] = (batch['cloth_mask']*(opdict['mesh_image'] - tex[:,hand_idx].detach().mean(1)[:,:,None,None]).abs()).mean()*self.cfg.loss.mesh_skin_consistency           
                if self.cfg.loss.mesh_w_mrf > 0.:
                    losses['mesh_image_mrf'] = self.mrf_loss(opdict['mesh_image']*batch['skin_mask'], batch['image']*batch['skin_mask'])*self.cfg.loss.mesh_w_mrf
            else:
                losses['mesh_image'] = lossfunc.huber(opdict['mesh_image'], batch['image'])*self.cfg.loss.mesh_w_rgb
                losses['mesh_mask'] = lossfunc.huber(opdict['mesh_mask'], batch['mask'])*self.cfg.loss.mesh_w_alpha
                if self.cfg.loss.mesh_w_mrf > 0.:
                    losses['mesh_image_mrf'] = self.mrf_loss(opdict['mesh_image'], batch['image'])*self.cfg.loss.mesh_w_mrf
                
        #########################################################d
        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        return losses, opdict

    def prepare_data(self):
        self.train_dataset = build_datasets.build_train(self.cfg.dataset, mode='train')
        self.val_dataset = build_datasets.build_train(self.cfg.dataset, mode='val')
        logger.info('---- training data numbers: ', len(self.train_dataset))
        if os.path.exists(self.train_dataset.pose_cache_path) is False:
            cache_data(self.train_dataset)            
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.cfg.dataset.num_workers,
                            pin_memory=True,
                            drop_last=True)
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=4, shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)

    @torch.no_grad()
    def validation_step(self, data='train', batch=None, returnVis=False):
        self.model.eval()
        if batch is None:
            if data == 'val':
                val_iter = iter(self.val_dataloader)
                batch = next(val_iter)
            else:
                val_iter = iter(self.train_dataloader)
                batch = next(val_iter)
            util.move_dict_to_device(batch, self.device)
            if data == 'train':
                batch = self.posemodel(batch)
            else:
                if self.cfg.use_deformation and self.cfg.deformation_type == 'opt_code':
                    frame_id = '000000'
                    deformation_code = getattr(self.posemodel, f'{self.posemodel.subject_id}_deformation_{frame_id}')
                    batch['deformation_code'] = deformation_code.expand(batch['image'].shape[0], -1)
        # first batch to show canonical
        # if data == 'val':
        #     batch['full_pose'][:1] = self.model.canonical_pose
        #     batch['cam'][:1] = self.model.canonical_cam                  
        batch['global_step'] = self.global_step
        opdict = self.model(batch, train=False)
        visdict = {}
        datadict = {**batch, **opdict}
        for key in datadict.keys():
            if not (self.cfg.use_mesh and self.cfg.use_nerf):
                if 'cloth' in key or 'skin' in key:
                    continue
            if 'path' in key or 'depth' in key or 'vis' in key or 'sampled' in key:
                continue
            if 'image' in key:
                visdict[key] = datadict[key]
            if 'mask' in key:
                visdict[key] = datadict[key].expand(-1, 3, -1, -1)        
        visdict['shape_image'] = self.model.forward_mesh(batch, renderShape=True, background=batch['image'])['shape_image']
        if self.cfg.use_outer_mesh:
            visdict['shape_outer_image'] = self.model.forward_mesh(batch, renderShape=True, background=batch['image'])['shape_outer_image']

        if returnVis:
            return visdict
        if self.cfg.opt_pose and data=='train':
            batch['full_pose'] = batch['init_full_pose']
            if self.cfg.opt_cam:
                batch['cam'] = batch['init_cam']
            rendering = self.model.forward_mesh(batch, renderShape=True, background=batch['image'])
            visdict['init_pose'] = rendering['shape_image']
        savepath = os.path.join(self.cfg.output_dir, self.cfg.train.val_vis_dir, f'{self.global_step:06}.jpg')
        grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=self.image_size)
        images = wandb.Image(grid_image[:,:,[2,1,0]], caption="validation")
        wandb.log({data: images}, step=self.global_step)
        logger.info(f'---- validation {data} step: {self.global_step}, save to {savepath}')

    def fit(self):
        self.prepare_data()
        iters_every_epoch = int(len(self.train_dataset)/self.batch_size)
        start_epoch = self.global_step//iters_every_epoch
        for epoch in range(start_epoch, self.cfg.train.max_epochs):
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{self.cfg.train.max_epochs}]"):
                if epoch*iters_every_epoch + step < self.global_step:
                    continue
                if self.global_step % self.cfg.train.val_steps == 0:
                    self.validation_step(data='train')
                    self.validation_step(data='val')
                ## train
                try:
                    batch = next(self.train_iter)
                except:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)
                losses, _ = self.training_step(batch, self.global_step)
                all_loss = losses['all_loss']

                self.optimizer.zero_grad()
                all_loss.backward()
                self.optimizer.step()

                ### logger
                if self.global_step % self.cfg.train.log_steps == 0:
                    loss_info = f"ExpName: {self.cfg.exp_name} \n Step: {self.global_step}, Epoch: {epoch}, Iter: {step}/{iters_every_epoch}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
                    for k, v in losses.items():
                        loss_info = loss_info + f'{k}: {v:.6f}, '
                    wandb.log(losses, step=self.global_step)                                                  
                    logger.info(loss_info)
                    
                if self.global_step % self.cfg.train.checkpoint_steps == 0:
                    torch.save(self.model_dict(), os.path.join(self.cfg.output_dir, 'model.tar'))   
                
                self.global_step += 1
                if self.global_step > self.cfg.train.max_steps:
                    print('training done')
                    return 0
                self.scheduler.step()
