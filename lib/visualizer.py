import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import wandb
import yaml
import shutil
import mcubes
import trimesh
from glob import glob

from copy import deepcopy

from .utils import util, rotation_converter, lossfunc
from .utils.config import cfg
from .datasets import build_datasets
from .trainer import Trainer
from .utils.rasterize_rendering import render_shape

def mcubes_to_world(vertices, N, x_range, y_range, z_range):
    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range
    vertices_ = vertices / N
    x_ = (ymax-ymin) * vertices_[:, 1] + ymin
    y_ = (xmax-xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax-zmin) * vertices_[:, 2] + zmin
    return vertices_

def create_grid(N, x_range, y_range, z_range):
    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)
    grid = np.stack(np.meshgrid(x, y, z), -1)
    return grid

def load_pixie_smplx(actions_file='/home/yfeng/github/SCARF/data/pixie_radioactive.pkl'):
    # load pixie animation poses
    assert os.path.exists(actions_file), f'{actions_file} does not exist'
    with open(actions_file, 'rb') as f:
        codedict = pickle.load(f)
    full_pose = torch.from_numpy(codedict['full_pose'])
    cam = torch.from_numpy(codedict['cam'])
    cam[:,0] = cam[:,0] * 1.15
    cam[:,2] = cam[:,2] + 0.3
    exp = torch.from_numpy(codedict['exp'])
    return full_pose, exp, cam

class Visualizer(Trainer):
    def __init__(self, config=None):
        super(Visualizer, self).__init__(config=config)

    def save_video(self, savepath, image_list, fps=10):
        video_type = savepath.split('.')[-1]
        if video_type == 'mp4' or video_type == 'gif':
            import imageio
            if video_type == 'mp4':
                writer = imageio.get_writer(savepath, mode='I', fps=fps)
            elif video_type == 'gif':
                writer = imageio.get_writer(savepath, mode='I', duration=1/fps)
            for image in image_list:
                writer.append_data(image[:,:,::-1])
            writer.close()
            logger.info(f'{video_type} saving to {savepath}')
        
    @torch.no_grad()
    def capture(self, savefolder, saveImages=False, video_type='mp4', fps=10):
        """ show color and hybrid rendering of training frames

        Args:
            savefolder (_type_): _description_
        """
        # load data
        self.train_dataset = build_datasets.build_train(self.cfg.dataset, mode='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)
        image_list = []
        for batch_nb, batch in enumerate(tqdm(self.train_dataloader)):
            util.move_dict_to_device(batch, self.device)
            frame_id = batch['frame_id'][0]
            visdict = {'image': batch['image']}
                
            # run model
            if self.cfg.opt_pose:
                batch = self.posemodel(batch)
            opdict = self.model(batch, train=False)
            # visualization
            visdict['render'] = opdict['nerf_fine_image']
            visdict['render_hybrid'] = opdict['nerf_fine_hybrid_image']
            savepath = os.path.join(savefolder, f'{self.cfg.exp_name}_f{frame_id}.jpg')
            grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=512, print_key=False)
            image_list.append(grid_image)
            print(f'saving to {savepath}')
            if saveImages:
                os.makedirs(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}'), exist_ok=True)
                for key in visdict.keys():
                    image = visdict[key]
                    cv2.imwrite(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}', f'{self.cfg.exp_name}_f{frame_id}_{key}.jpg'),util.tensor2image(visdict[key][0]))
        videopath = os.path.join(savefolder, f'{self.cfg.exp_name}_capture.{video_type}')
        self.save_video(videopath, image_list, fps=fps)
            
    @torch.no_grad()
    def novel_view(self, savefolder, frame_id=0, saveImages=False, video_type='mp4', fps=10):
        """ show novel view of given frames
        Args:
            savefolder (_type_): _description_
        """
        # load data
        self.cfg.dataset.train.frame_start= frame_id
        self.cfg.dataset.train.frame_step = 1
        self.cfg.dataset.train.frame_end = frame_id + 1
        self.train_dataset = build_datasets.build_train(self.cfg.dataset, mode='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)
        
        image_list = []
        for batch_nb, batch in enumerate(tqdm(self.train_dataloader)):
            util.move_dict_to_device(batch, device=self.device)
            frame_id = batch['frame_id'][0]
            # visdict = {'image': batch['image']}
            visdict = {}
            if self.cfg.opt_pose:
                batch = self.posemodel(batch)
                
            # change the global pose (pelvis) for novel view
            yaws = np.arange(0, 361, 20)
            init_pose = batch['full_pose']
            for yaw in tqdm(yaws):
                euler_pose = torch.zeros((1, 3), device=self.device, dtype=torch.float32)
                euler_pose[:,1] = yaw
                global_pose = rotation_converter.batch_euler2matrix(rotation_converter.deg2rad(euler_pose))
                pose = init_pose.clone()
                pose[:,0,:,:] = torch.matmul(pose[:,0,:,:], global_pose)
                batch['full_pose'] = pose
                opdict = self.model(batch, train=False)
                # visualization
                visdict['render'] = opdict['nerf_fine_image']
                visdict['render_hybrid'] = opdict['nerf_fine_hybrid_image']
                savepath = os.path.join(savefolder, f'{self.cfg.exp_name}_f{frame_id}_{yaw:03}.jpg')
                grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=512, print_key=False)
                image_list.append(grid_image)
                print(f'saving to {savepath}')
                if saveImages:
                    os.makedirs(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}_{yaw:03}'), exist_ok=True)
                    for key in visdict.keys():
                        image = visdict[key]
                        cv2.imwrite(os.path.join(savefolder,  f'{self.cfg.exp_name}_f{frame_id}_{yaw:03}', f'{self.cfg.exp_name}_f{frame_id}_{key}.jpg'),util.tensor2image(visdict[key][0]))
            videopath = os.path.join(savefolder, f'{self.cfg.exp_name}_{frame_id}_novel_view.{video_type}')
            self.save_video(videopath, image_list, fps=fps)
                            
    @torch.no_grad()
    def extract_mesh(self, savefolder, frame_id=0):
        logger.info(f'extracting mesh from frame {frame_id}')
        # load data
        self.cfg.dataset.train.frame_start= frame_id
        self.cfg.dataset.train.frame_step = 1
        self.cfg.dataset.train.frame_end = frame_id + 1
        self.train_dataset = build_datasets.build_train(self.cfg.dataset, mode='train')
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False)
        
        for batch_nb, batch in enumerate(tqdm(self.train_dataloader)):
            util.move_dict_to_device(batch, device=self.device)
            frame_id = batch['frame_id'][0]
            visdict = {'image': batch['image']}
            
            # extract mesh
            N_grid = 256
            sigma_threshold = 50.
            if self.cfg.use_nerf:
                ## points from rays, rays, xyz_fine, viewdir, z_vals
                x_range = [-1, 1]
                y_range = [-1, 1]
                z_range = [self.model.near, self.model.far]
                grid = create_grid(N_grid, x_range, y_range, z_range)
                xyz = torch.from_numpy(grid.reshape(-1, 3)).unsqueeze(0).float().to(self.device) # (1, N*N*N, 3)

                sigmas = []
                chunk = 32*32*4*2
                mesh_out = self.model.forward_mesh(batch, returnVerts=True)
                batch.update(mesh_out)
                for i in tqdm(range(0, xyz.shape[1], chunk)):
                    xyz_chunk = xyz[:, i:i+chunk, :]
                    dir_chunk = torch.zeros_like(xyz_chunk)
                    ## backward skinning
                    xyz_chunk, dir_chunk, _ = self.model.backward_skinning(batch, xyz_chunk, dir_chunk)
                    ## query canonical model
                    _, sigma_chunk = self.model.query_canonical_space(xyz_chunk, dir_chunk, use_fine=True)
                    sigmas.append(torch.relu(sigma_chunk))
                sigmas = torch.cat(sigmas, 1)

                sigmas = sigmas.cpu().numpy()
                sigmas = np.maximum(sigmas, 0).reshape(N_grid, N_grid, N_grid)
                sigmas = sigmas - sigma_threshold
                # smooth
                sigmas = mcubes.smooth(sigmas)
                # sigmas = mcubes.smooth(sigmas)
                vertices, faces = mcubes.marching_cubes(-sigmas, 0.)
                vertices = mcubes_to_world(vertices, N_grid, x_range, y_range, z_range)
                cloth_verts = vertices
                cloth_faces = faces
                
            ### add body shape
            if self.cfg.use_mesh:
                mesh_out = self.model.forward_mesh(batch, renderShape=True)
                body_verts = mesh_out['trans_verts'].cpu().numpy().squeeze()
                body_faces = self.model.faces.cpu().numpy().squeeze()
                if self.cfg.use_nerf:
                    # combine two mesh
                    faces = np.concatenate([faces, body_faces+vertices.shape[0]]).astype(np.int32)
                    vertices = np.concatenate([vertices, body_verts], axis=0).astype(np.float32)
                else:
                    faces = body_faces
                    vertices = body_verts
                    
            ### visualize
            batch_size = 1
            faces = torch.from_numpy(faces.astype(np.int32)).long().to(self.device)[None,...]
            vertices = torch.from_numpy(vertices).float().to(self.device)[None,...]
            if self.cfg.use_mesh and self.cfg.use_nerf:
                colors = torch.ones_like(vertices)*180/255.
                colors[:,:cloth_verts.shape[0], [0,2]] = 180/255.
                colors[:,:cloth_verts.shape[0], 1] = 220/255.
            else:
                colors = None
            # import ipdb; ipdb.set_trace()
            shape_image = render_shape(vertices = vertices, faces = faces, 
                                    image_size=512, blur_radius=1e-8,
                                    colors=colors)
                                    # background=batch['image'])
            visdict['shape_image'] = shape_image
                
            savepath = os.path.join(savefolder, f'{self.cfg.exp_name}_f{frame_id}_vis.jpg')
            grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=512, print_key=False)
            # save obj
            util.write_obj(os.path.join(savefolder, f'{self.cfg.exp_name}_f{frame_id}.obj'), 
                           vertices[0].cpu().numpy(), 
                           faces[0].cpu().numpy()[:,[0,2,1]],
                           colors=colors[0].cpu().numpy())
        logger.info(f'Visualize results saved to {savefolder}')
    
    @torch.no_grad()
    def animate(self, savefolder, animation_file, saveImages=False, video_type='mp4', fps=10):
        # load animation poses        
        full_pose, exp, cam = load_pixie_smplx(animation_file)
        cam = cam.to(self.device)
        full_pose = full_pose.to(self.device)
        exp = exp.to(self.device)
        image_list = []
        for i in tqdm(range(full_pose.shape[0])):
            batch = {'full_pose': full_pose[i:i+1], 
                     'exp': exp[i:i+1], 
                     'cam': cam[i:i+1]}
            opdict = self.model(batch, train=False)
            visdict = {
                'render': opdict['nerf_fine_image'],
                'render_hybrid': opdict['nerf_fine_hybrid_image'],
            }
            savepath = os.path.join(savefolder, f'{self.cfg.exp_name}_animation_{i:03}.jpg')
            grid_image = util.visualize_grid(visdict, savepath, return_gird=True, size=512, print_key=False)
            image_list.append(grid_image)
        
        video_path = os.path.join(savefolder, f'{self.cfg.exp_name}_animation.{video_type}')
        self.save_video(video_path, image_list, fps=fps)
        
    @torch.no_grad()
    def run(self, vistype, args=None):
        # check if body or clothing model are specified
        model_dict = self.model_dict()
        if os.path.exists(args.body_model_path):
            body_name = args.body_model_path.split('/')[-2]
            savefolder = os.path.join(self.cfg.output_dir, f'visualization_body_{body_name}', vistype)
            if os.path.exists(args.body_model_path):
                checkpoint = torch.load(args.body_model_path)
                for param_name in model_dict['model'].keys():
                    if 'mlp_geo' in param_name or 'mlp_tex' in param_name or 'beta' in param_name:
                        if param_name in checkpoint['model']:
                            model_dict['model'][param_name].copy_(checkpoint['model'][param_name])
        elif os.path.exists(args.clothing_model_path):
            clothing_name = args.clothing_model_path.split('/')[-2]
            savefolder = os.path.join(self.cfg.output_dir, f'visualization_clothing_{clothing_name}', vistype)
            if os.path.exists(args.clothing_model_path):
                checkpoint = torch.load(args.clothing_model_path)
                for param_name in model_dict['model'].keys():
                    if 'nerf' in param_name:
                        model_dict['model'][param_name].copy_(checkpoint['model'][param_name])
        else:
            savefolder = os.path.join(self.cfg.output_dir, 'visualization', vistype)
            
        os.makedirs(savefolder, exist_ok=True)
        if vistype == 'capture':
            self.capture(savefolder, saveImages=args.saveImages, video_type=args.video_type, fps=args.fps)
        elif vistype == 'novel_view':
            self.novel_view(savefolder, frame_id=args.frame_id, saveImages=args.saveImages, video_type=args.video_type, fps=args.fps)
        elif vistype == 'extract_mesh':
            self.extract_mesh(savefolder, frame_id=args.frame_id)
        elif vistype == 'animate':
            self.animate(savefolder, animation_file=args.animation_file, saveImages=args.saveImages, video_type=args.video_type, fps=args.fps)