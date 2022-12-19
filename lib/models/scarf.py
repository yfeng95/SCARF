import torch
from torch import nn
import numpy as np
import pickle
import os
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, TexturesVertex,
)
from pytorch3d.io import load_obj
from pytorch3d.ops.knn import knn_points
from .smplx import SMPLX
from ..utils.volumetric_rendering import fancy_integration, perturb_points, sample_pdf
from ..utils.rasterize_rendering import pytorch3d_rasterize, render_shape
from ..utils import rotation_converter, camera_util
from ..utils import util
from .nerf import NeRF, DeRF
from .siren import GeoSIREN

def smplx_lbsmap_top_k(lbs_weights, verts_transform, points, template_points, source_points=None, K=1, addition_info=None):
    '''ref: https://github.com/JanaldoChen/Anim-NeRF
    Args:  
    '''
    bz, np, _ = points.shape
    with torch.no_grad():
        results = knn_points(points, template_points, K=K)
        dists, idxs = results.dists, results.idx
    neighbs_dist = dists
    neighbs = idxs
    weight_std = 0.1
    weight_std2 = 2. * weight_std ** 2
    xyz_neighbs_lbs_weight = lbs_weights[neighbs] # (bs, n_rays*K, k_neigh, 24)
    xyz_neighbs_weight_conf = torch.exp(-torch.sum(torch.abs(xyz_neighbs_lbs_weight - xyz_neighbs_lbs_weight[..., 0:1, :]), dim=-1)/weight_std2) # (bs, n_rays*K, k_neigh)
    xyz_neighbs_weight_conf = torch.gt(xyz_neighbs_weight_conf, 0.9).float()
    xyz_neighbs_weight = torch.exp(-neighbs_dist) # (bs, n_rays*K, k_neigh)
    xyz_neighbs_weight *= xyz_neighbs_weight_conf
    xyz_neighbs_weight = xyz_neighbs_weight / xyz_neighbs_weight.sum(-1, keepdim=True) # (bs, n_rays*K, k_neigh)

    xyz_neighbs_transform = util.batch_index_select(verts_transform, neighbs) # (bs, n_rays*K, k_neigh, 4, 4)
    xyz_transform = torch.sum(xyz_neighbs_weight.unsqueeze(-1).unsqueeze(-1) * xyz_neighbs_transform, dim=2) # (bs, n_rays*K, 4, 4)
    xyz_dist = torch.sum(xyz_neighbs_weight * neighbs_dist, dim=2, keepdim=True) # (bs, n_rays*K, 1)

    if addition_info is not None: #[bz, nv, 3]
        xyz_neighbs_info = util.batch_index_select(addition_info, neighbs)
        xyz_info = torch.sum(xyz_neighbs_weight.unsqueeze(-1)* xyz_neighbs_info, dim=2) 
        return xyz_dist, xyz_transform, xyz_info
    else:
        return xyz_dist, xyz_transform

class SCARF(nn.Module):
    def __init__(self, cfg, init_beta=None, device='cuda:0'):
        super(SCARF, self).__init__()
        self.cfg = cfg
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        
        ## set smplx model
        self.set_smplx()
        if init_beta is None:
            beta = torch.zeros([1,10]).float()
        else:
            beta = init_beta.view(1,10)
        self.register_parameter('beta', torch.nn.Parameter(beta.to(self.device)))

        ## scarf model and setup corresponding rendering
        if self.cfg.use_mesh: # hybrid: mesh as inside
            self.cond_dim = 1
            self.mlp_geo = GeoSIREN(input_dim=3, z_dim=self.cond_dim, hidden_dim=128, output_dim=3, device=self.device, last_op=torch.tanh, scale=self.cfg.mesh_offset_scale)
            self.mlp_tex = GeoSIREN(input_dim=3, z_dim=self.cond_dim, hidden_dim=128, output_dim=3, device=self.device, last_op=torch.sigmoid, scale=1)
            # setup renderer for mesh
            self._setup_render()

        if self.cfg.use_nerf:
            if self.cfg.use_deformation:
                self.derf = DeRF(freqs_xyz=cfg.freqs_xyz, deformation_dim=cfg.deformation_dim, out_channels=3)
            self.nerf = NeRF(freqs_xyz=cfg.freqs_xyz, freqs_dir=cfg.freqs_dir, use_view=cfg.use_view, deformation_dim=0)
            if self.cfg.use_fine:
                if cfg.share_fine:
                    self.nerf_fine = self.nerf
                else:
                    self.nerf_fine = NeRF(freqs_xyz=cfg.freqs_xyz, freqs_dir=cfg.freqs_dir, use_view=cfg.use_view, deformation_dim=0)
                ## rays for volume rendering
                self.pixel_rays = self.fixed_rays()

    def set_smplx(self):
        # set canonical space
        self.smplx = SMPLX(self.cfg.model).to(self.device)
        pose = torch.zeros([55,3], dtype=torch.float32, device=self.device) # 55
        angle = 30*np.pi/180.
        pose[1, 2] = angle
        pose[2, 2] = -angle
        pose_euler = pose.clone()
        pose = rotation_converter.batch_euler2matrix(pose)
        pose = pose[None,...]
        xyz_c, _, _, A, T, shape_offsets, pose_offsets = self.smplx(full_pose = pose, return_T=True)
        self.v_template = xyz_c.squeeze().to(self.device)
        self.canonical_transform = T
        self.canonical_offsets = shape_offsets + pose_offsets
        ##--- load vertex mask
        with open(self.cfg.model.mano_ids_path, 'rb') as f:
            hand_idx = pickle.load(f)
        flame_idx = np.load(self.cfg.model.flame_ids_path)
        with open(self.cfg.model.flame_vertex_masks_path, 'rb') as f:
            flame_vertex_mask = pickle.load(f, encoding='latin1')
        # verts = torch.nn.Parameter(self.v_template, requires_grad=True)
        exclude_idx = []
        exclude_idx += list(hand_idx['left_hand'])
        exclude_idx += list(hand_idx['right_hand'])
        exclude_idx += list(flame_vertex_mask['face'])
        exclude_idx += list(flame_vertex_mask['left_eyeball'])
        exclude_idx += list(flame_vertex_mask['right_eyeball'])
        exclude_idx += list(flame_vertex_mask['left_ear'])
        exclude_idx += list(flame_vertex_mask['right_ear'])
        all_idx = range(xyz_c.shape[1])
        face_idx = list(flame_vertex_mask['face'])
        body_idx = [i for i in all_idx if i not in face_idx]
                
        self.part_idx_dict = {
            'face': flame_vertex_mask['face'],
            'hand': list(hand_idx['left_hand']) + list(hand_idx['right_hand']), 
            'exclude': exclude_idx,
            'body': body_idx
        }
        ## smplx topology
        _, faces, _ = load_obj(self.cfg.model.topology_path)
        self.faces = faces.verts_idx[None,...].to(self.device)

        self.smplx_verts = self.verts = self.v_template
        self.smplx_faces = self.faces = self.smplx.faces_tensor

        # higher resolution smplx
        if self.cfg.use_highres_smplx:
            highres_path = self.cfg.model.highres_path
            embedding_path = os.path.join(highres_path, 'embedding.npz')
            mesh_path = os.path.join(highres_path, 'quad_mesh.obj')
            # Faces of the upsampled SMPL-X triangle mesh
            _, faces_idx, _ = load_obj(mesh_path)
            # Subdivision embedding
            subdiv_embedding = np.load(embedding_path, allow_pickle=True)
            # Faces of the low_resolution mesh that each upsampled vertex is embedded in
            nearest_faces = subdiv_embedding['nearest_faces']
            # Barycentric coordinates of each upsampled vertices
            b_coords = subdiv_embedding['b_coords']
            self.nearest_faces = torch.from_numpy(nearest_faces).to(self.device).long()
            # Get subdivided vertices using the Barycentric coordinates
            b_coords = torch.from_numpy(b_coords).to(self.device)
            subdiv_verts = self.smplx_verts[self.smplx_faces[nearest_faces,0]]*b_coords[:,0:1] + self.smplx_verts[self.smplx_faces[nearest_faces,1]]*b_coords[:,1:2] + self.smplx_verts[self.smplx_faces[nearest_faces,2]]*b_coords[:,2:]
            self.verts = subdiv_verts.squeeze().float()
            self.faces = faces_idx.verts_idx.squeeze().to(self.device)            
            self.b_coords = b_coords.float()
            ## face and hands index of high res version
            for key in self.part_idx_dict.keys():
                part_idx = self.part_idx_dict[key]
                smplx_color = torch.zeros([self.smplx_verts.shape[0]], device=self.device)
                smplx_color[part_idx] = 1.
                highres_color = smplx_color[self.smplx_faces[nearest_faces,0]]*b_coords[:,0] + smplx_color[self.smplx_faces[nearest_faces,1]]*b_coords[:,1] + smplx_color[self.smplx_faces[nearest_faces,2]]*b_coords[:,2]
                highres_idx = torch.nonzero(highres_color).squeeze()
                self.part_idx_dict[key] = highres_idx.cpu().numpy().astype(np.int32)
            ## highres to lowres index
            if self.cfg.loss.use_new_edge_loss:
                from ..utils import lossfunc
                reg_verts = self.verts.cpu().numpy().squeeze()
                reg_faces = self.faces.cpu().numpy().squeeze()
                verts_per_edge = lossfunc.get_vertices_per_edge(len(reg_verts), reg_faces)
                self.verts_per_edge = torch.from_numpy(verts_per_edge).float().to(self.device).long()
        ### cam and pose to visualize canonical pose
        pose_euler[0, 2] = np.pi
        pose_euler[0, 1] = np.pi
        pose = rotation_converter.batch_euler2matrix(pose_euler)
        self.canonical_pose = pose[None,...].to(self.device)
        self.canonical_cam = torch.tensor([[1., 0., 0.28]]).float().to(self.device)
        if self.cfg.use_perspective:
            self.canonical_cam = torch.tensor([[46.6029,  0.0747,  0.3319, 47.1121]]).float().to(self.device)
            
    def _setup_render(self):
        ## camera
        R = torch.eye(3).unsqueeze(0)
        T = torch.zeros([1,3])
        batch_size = 1
        self.cameras = pytorch3d.renderer.cameras.FoVOrthographicCameras(
                    R=R.expand(batch_size, -1, -1), T=T.expand(batch_size, -1), znear=0.0).to(self.device)
        
        blend_params = BlendParams(sigma=1e-7, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=50, 
            bin_size = 0
            )
        # Create a silhouette mesh renderer by composing a rasterizer and a shader. 
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )
    
    def fixed_rays(self):
        # generat rays
        height, width = self.image_size, self.image_size
        if self.cfg.use_perspective:
            near = 43; far = 48.5
        else:
            near = -0.6; far = 0.6
        self.near = near
        self.far = far
        Y, X = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=self.device)/height*2. - 1,
            torch.arange(width, dtype=torch.float32, device=self.device)/width*2. - 1,
            )
        Z = torch.zeros_like(X)*near
        center = torch.stack((X,Y,Z), dim=-1)[None,...]
        direction = torch.tensor([0,0,1.], device=self.device)[None,None,None,:].repeat(1,height, width,1)
        near = torch.ones_like(X)[None,:,:,None]*near
        far = torch.ones_like(X)[None,:,:,None]*far 
        pixel_rays = torch.cat([center, direction, near, far], dim=-1) 
        return pixel_rays

    @torch.no_grad()
    def sample_rays(self, batch, train=False):
        ''' sample points in view space
        '''
        batch_size = batch['cam'].shape[0]
        with torch.no_grad():
            ## random rays for each image
            rays = self.pixel_rays.expand(batch_size, -1, -1, -1)            
            if self.cfg.use_mesh:
                vis_image = batch['vis_image'].detach()
                depth_image = batch['depth_image'].detach().squeeze() 
                far = depth_image + self.cfg.depth_std
                vis = vis_image.clone().squeeze(1)
                rays = rays.clone()
                rays_far = far.clone()*vis + rays.clone()[:,:,:,7]*(1-vis)
                rays[:,:,:,7] = rays_far
            if train:
                if self.cfg.train.precrop_iters > batch['global_step'] and not self.cfg.sample_patch_rays:
                    H, W = self.image_size, self.image_size
                    dH = int(H//2 * self.cfg.train.precrop_frac)
                    dW = int(W//2 * self.cfg.train.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, self.image_size-1, self.image_size), torch.linspace(0, self.image_size-1, self.image_size)), -1)  
                n_rays = self.cfg.chunk
                coords = torch.reshape(coords, [-1,2]).long().to(self.device)
                # 
                select_inds = torch.randperm(coords.shape[0])[:n_rays]
                coords = coords[select_inds]
                if self.cfg.sample_patch_rays:
                    half_size = self.cfg.sample_patch_size//2
                    while True:
                        tmp_mask = torch.zeros([self.image_size, self.image_size]).to(self.device)
                        center = torch.randint(low=half_size*2, high=self.image_size-half_size*2, size=(2,))
                        # center = [self.image_size//2, self.image_size//2]
                        tmp_mask[center[0]-half_size:center[0]+half_size, center[1]-half_size:center[1]+half_size] = 1.
                        inds = torch.nonzero(tmp_mask)
                        gt_mask = batch['mask']
                        patch_mask = gt_mask[:,:,inds[:,0], inds[:,1]]
                        if patch_mask.sum() > self.cfg.sample_patch_size**2*batch_size/3.:
                            break
                    coords[:self.cfg.sample_patch_size**2] = inds
                    #
                rays = rays[:,coords[:,0], coords[:,1]]
                ## train, sample corresponding gt 
                for key in ['image', 'mesh_mask', 'cloth_mask', 'mesh_image', 'mask', 'shape_image']:
                    if key in batch.keys():
                        gts = batch[key].permute(0,2,3,1)
                        gts_sampled = gts[:,coords[:,0], coords[:,1]].reshape(batch_size, -1, gts.shape[-1])
                        batch[f'{key}_sampled'] = gts_sampled
            else:
                rays = rays.reshape(batch_size,-1,rays.shape[-1])
                for key in ['mesh_mask', 'mesh_image', 'shape_image']:
                    if key in batch.keys():
                        gts = batch[key].permute(0,2,3,1)
                        gts_sampled = gts.reshape(batch_size, -1, gts.shape[-1])
                        batch[f'{key}_sampled'] = gts_sampled

        return rays
    
    @torch.no_grad()
    def sample_points(self, rays, perturb=False, z_coarse=None, weights=None, combine=True):
        batch_size = rays.shape[0]
        center = rays[..., :3]; direction = rays[..., 3:6]; near = rays[...,6]; far = rays[...,7]
        # sample coarse points
        if z_coarse is None:
            z_steps = torch.linspace(0, 1, self.cfg.n_samples, device=self.device)
            z_steps = z_steps.unsqueeze(0).expand(batch_size, -1)  # (B, Kc)
            z_steps = z_steps[:,None,:, None] #[batch_size, 1, n_points, 1]
            z_vals = near[:,:,None,None] * (1-z_steps) + far[:,:,None,None] * z_steps
            xyz = center[:,:,None,:] + z_vals * direction[:,:,None,:]
            ## pertub_points
            if perturb:
                xyz, z_vals = perturb_points(xyz, z_vals, direction, device=xyz.device)
        # sample fine points depends on weights
        else:
            z_vals_coarse = z_coarse.reshape(-1, self.cfg.n_samples)
            z_vals_mid = 0.5 * (z_vals_coarse[..., :-1] + z_vals_coarse[..., 1:])
            z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1].detach(),
                                 self.cfg.n_importance, det=True).detach()
            if combine:
                z_vals = torch.cat([z_coarse, z_vals], dim=-1)
                z_vals, _ = torch.sort(z_vals, dim=-1)
            z_vals = z_vals.reshape(rays.shape[0], rays.shape[1], -1, 1)
            xyz = center[:,:,None,:] + z_vals * direction[:,:,None,:]
        
        xyz = xyz.view(batch_size, -1, 3)
        ## note: viewdir not used
        viewdir = direction.view(batch_size, -1, 3)
        return xyz, viewdir, z_vals
    
    def cam_project(self, points, cam, inv=False):
        if self.cfg.cam_mode == 'orth':
            if inv:
                proj_points = util.batch_orth_proj_inv(points, cam.squeeze(-1))
            else:
                proj_points = util.batch_orth_proj(points, cam.squeeze(-1))
        else:
            if inv:
                proj_points = camera_util.perspective_project_inv(points, focal=cam[:,0].mean(), transl=cam[:,1:])
            else:
                proj_points = camera_util.perspective_project(points, focal=cam[:,0].mean(), transl=cam[:,1:])
        return proj_points

    def deformation(self, xyz, deformation_code):
        ''' xyz: canonical points (after backward skinning)
        '''
        _, nv = xyz.shape[:2]
        d_xyz = self.derf(xyz, deformation_code)
        d_xyz = d_xyz*0.001
        xyz_deformed = xyz + d_xyz
        return xyz_deformed, d_xyz
    
    def backward_skinning(self, batch, xyz, viewdir):
        '''
            xyz: [B, n_ray, n_point, 3]
        '''
        batch_size, n_sample,  _ = xyz.shape
        cam = batch['cam']

        #-- backward skinning based on knn 
        posed_verts, _, _, joints_transform, curr_vertices_transform, shape_offsets, pose_offsets = \
                self.smplx(full_pose=batch['full_pose'], shape_params=self.beta.expand(batch_size, -1), transl=batch.get('transl', torch.zeros([batch_size, 3], device=xyz.device)), return_T=True)
        curr_offsets = shape_offsets + pose_offsets
        ## inverse transofrm: 
        # posed -> mean shape -> canonical
        xyz = self.cam_project(xyz, cam, inv=True)
        vertices_transform = torch.inverse(curr_vertices_transform)
        vertices_transform_3x1 = vertices_transform[...,:3,3] - curr_offsets + self.canonical_offsets
        vertices_transform_3x3 = torch.cat([vertices_transform[...,:3,:3], vertices_transform_3x1[...,None]], dim=-1) #3x4
        vertices_transform = torch.cat([vertices_transform_3x3, vertices_transform[...,3:,:]], dim=-2)
        vertices_transform = torch.matmul(self.canonical_transform.clone().repeat(batch_size, 1, 1, 1), vertices_transform)
        ## smplx to highres smplx
        if self.cfg.use_highres_smplx:
            verts_index = self.smplx_faces[self.nearest_faces]
            nearest_verts_transform = util.batch_index_select(vertices_transform, verts_index) #[bz, nv, 3, 4, 4]
            vertices_transform = (nearest_verts_transform*self.b_coords[None, :,:,None,None]).sum(2)
            posed_verts = batch['posed_verts']
            if self.cfg.use_mesh:
                offset = batch['offset']
                vertices_transform[:,:,:3,3] = vertices_transform[:,:,:3,3] - offset
            lbs_weights = (util.batch_index_select(self.smplx.lbs_weights[None,...], verts_index)*self.b_coords[None, :,:,None]).sum(2).squeeze()
        else:
            lbs_weights = self.smplx.lbs_weights   
        
        surface_verts = posed_verts
        if (self.cfg.use_deformation and self.cfg.deformation_type == 'posed_verts'):
            xyz_dist, xyz_transform, xyz_posed_verts = smplx_lbsmap_top_k(lbs_weights, vertices_transform, 
                                    xyz, surface_verts, K=self.cfg.k_neigh,
                                    addition_info=posed_verts)
            batch['deformation_code'] = xyz_posed_verts.detach()
        else:
            xyz_dist, xyz_transform = smplx_lbsmap_top_k(lbs_weights, vertices_transform, 
                                    xyz, surface_verts, K=self.cfg.k_neigh)
        
        xyz = util.batch_transform(xyz_transform, xyz)
        xyz_valid = torch.lt(xyz_dist, self.cfg.dis_threshold).float().squeeze()
        valid = xyz_valid.reshape(batch_size, self.cfg.chunk, -1, 1)
        if self.cfg.use_deformation:
            xyz, d_xyz = self.deformation(xyz, batch['deformation_code'])
            batch['d_xyz'] = d_xyz
        return xyz, viewdir, valid

    def query_canonical_space(self, xyz, viewdir=None, use_fine=False, only_sigma=False, only_normal=False, 
                                deformation_code=None):
        deformation_code = None
        if only_sigma:
            if not use_fine:
                sigma = self.nerf.get_sigma(xyz, deformation_code=deformation_code, only_sigma=only_sigma)
            else:
                sigma = self.nerf_fine.get_sigma(xyz, deformation_code=deformation_code, only_sigma=only_sigma)
            return sigma
        if only_normal:
            if not use_fine:
                normal = self.nerf.get_normal(xyz, deformation_code=deformation_code)
            else:
                normal = self.nerf_fine.get_normal(xyz, deformation_code=deformation_code)
            return normal
        if not use_fine:
            rgb, sigma = self.nerf(xyz, viewdir=viewdir, deformation_code=deformation_code)
        else:
            rgb, sigma = self.nerf_fine(xyz, viewdir=viewdir, deformation_code=deformation_code)

        return rgb, sigma
    
    def query_model(self, batch, rays, xyz, viewdir, z_vals, noise_std, use_fine=False, render_cloth=False, with_shape=False, get_normal=False):
        ## backward skinning
        xyz, viewdir, valid = self.backward_skinning(batch, xyz, viewdir)
        if get_normal:
            normal =  self.query_canonical_space(xyz, use_fine=True, only_normal=True)
            sigma =  self.query_canonical_space(xyz, use_fine=True, only_sigma=True)
            sigma = sigma.reshape(rays.shape[0], rays.shape[1], -1, 1)
            normal = normal.reshape(rays.shape[0], rays.shape[1], -1, 3)
            normals, _, _ = fancy_integration(torch.cat([normal, sigma], dim=-1), z_vals, device=self.device, white_back=self.cfg.dataset.white_bg, 
                        last_back=self.cfg.use_mesh, clamp_mode='relu', noise_std=noise_std)     
            return normals   
        ## query nerf mlp
        rgb, sigma = self.query_canonical_space(xyz, viewdir, use_fine=use_fine)
        rgb = rgb.reshape(rays.shape[0], rays.shape[1], -1, 3)
        sigma = sigma.reshape(rays.shape[0], rays.shape[1], -1, 1)
        if self.cfg.use_valid:
            sigma[valid < 1] = -1e6
        if self.cfg.use_mesh:
            rgb = torch.cat([rgb[:,:,:-1,:3].clone(), batch['mesh_image_sampled'][:,:,None,:]], dim=2)
            if render_cloth:
                if with_shape:
                    rgb = torch.cat([rgb[:,:,:-1,:3].clone(), batch['shape_image_sampled'][:,:,None,:]], dim=2)
                else:
                    rgb[:,:,-1,:3] = 1.
        rgbs, depths, weights = fancy_integration(torch.cat([rgb, sigma], dim=-1), z_vals, device=self.device, white_back=self.cfg.dataset.white_bg, 
                        last_back=self.cfg.use_mesh, clamp_mode='relu', noise_std=noise_std)        
        alphas = torch.sum(weights[...,:-1], dim=-1, keepdim=True)
        output = {
            'rgbs': rgbs,
            'weights': weights,
            'depths': depths,
            'alphas': alphas
            }
        return output

    def forward_nerf(self, batch, rays, train=False, render_cloth=False, with_shape=False):
        if train:
            noise_std = max(0, 1. - batch['global_step']/5000.)
        else:
            noise_std = 0.
        # 1. sample points from rays, coarse sample
        xyz, viewdir, z_vals = self.sample_points(rays, perturb=train)
        if self.cfg.n_importance > 0 and self.cfg.share_fine:
            with torch.no_grad():
                output = self.query_model(batch, rays, xyz, viewdir, z_vals, noise_std=noise_std, render_cloth=render_cloth, with_shape=with_shape)
        else:
            output = self.query_model(batch, rays, xyz, viewdir, z_vals, noise_std=noise_std, render_cloth=render_cloth, with_shape=with_shape)

        ### ----- fine model
        if self.cfg.use_fine and self.cfg.n_importance > 0:
            z_coarse = z_vals
            weights = output['weights']
            xyz_fine, viewdir_fine, z_vals_fine = self.sample_points(rays, z_coarse=z_coarse.reshape(-1, self.cfg.n_samples), \
                                                weights=weights.reshape(-1, self.cfg.n_samples)[..., 1:-1].detach()) # (bs, n_rays, Kf)
            fine_output = self.query_model(batch, rays, xyz_fine, viewdir_fine, z_vals_fine, noise_std=noise_std, use_fine=True, render_cloth=render_cloth, with_shape=with_shape)
            if self.cfg.share_fine:
                output = fine_output
            else:
                for key in fine_output:
                    output[f'{key}_fine'] = fine_output[key]
        return output
    
    def forward_mesh(self, batch, returnVerts=False, renderShape=False, renderDepth=False, background=None, clean_offset=False):
        batch_size = batch['cam'].shape[0]
        output = {}

        ## implicit surface function
        if self.cfg.use_mesh:
            cond = torch.ones([batch_size, self.cond_dim], device=self.device)
            offset = self.mlp_geo(self.verts[None,...].expand(batch_size, -1, -1), cond)
            if not self.cfg.use_texcond:
                tex = self.mlp_tex(self.verts[None,...].expand(batch_size, -1, -1), cond.detach())
            if clean_offset:
                offset[:] = 0.
        else:
            offset = torch.zeros_like(self.verts[None,...].expand(batch_size, -1, -1))
            tex = offset.clone()
        if self.cfg.exclude_hand:
            hand_mask = torch.ones_like(offset)
            hand_mask[:, self.part_idx_dict['hand']] = 0.
            offset = offset*hand_mask
        if self.cfg.use_highres_smplx:
            canonical_vert_pos = self.verts[None,...]+offset
            posed_verts, _, _, joints_transform, curr_vertices_transform, shape_offsets, pose_offsets = \
                    self.smplx(full_pose=batch['full_pose'], shape_params=self.beta.expand(batch_size, -1), 
                                transl=batch.get('transl', torch.zeros([batch_size, 3], device=self.device)), 
                                expression_params=batch.get('exp', torch.zeros([batch_size, 10], device=self.device)), 
                                return_T=True)
            curr_offsets = shape_offsets + pose_offsets
            # canonical -> mean shape -> posed
            vertices_transform = torch.inverse(self.canonical_transform.clone().repeat(batch_size, 1, 1, 1))
            vertices_transform_3x1 = vertices_transform[...,:3,3] + curr_offsets - self.canonical_offsets
            vertices_transform_3x3 = torch.cat([vertices_transform[...,:3,:3], vertices_transform_3x1[...,None]], dim=-1) #3x4
            vertices_transform = torch.cat([vertices_transform_3x3, vertices_transform[...,3:,:]], dim=-2)
            vertices_transform = torch.matmul(curr_vertices_transform, vertices_transform) #[bz, n_smplx_v, 4, 4]
            verts_index = self.smplx_faces[self.nearest_faces]
            nearest_verts_transform = util.batch_index_select(vertices_transform, verts_index) #[bz, nv, 3, 4, 4]
            verts_transform = (nearest_verts_transform*self.b_coords[None, :,:,None,None]).sum(2)
            posed_verts = util.batch_transform(verts_transform, canonical_vert_pos)

            if self.cfg.use_outer_mesh:
                offset_outer = self.mlp_geo_outer(self.verts[None,...].expand(batch_size, -1, -1), cond)
                canonical_vert_pos_outer = self.verts[None,...]+offset_outer
                posed_verts_outer = util.batch_transform(verts_transform, canonical_vert_pos_outer)
                output['posed_verts_outer'] = posed_verts_outer
                trans_verts_outer = self.cam_project(posed_verts_outer, batch['cam'])
                output['offset_outer'] = offset_outer
        else:        
            posed_verts, _, _, = \
                    self.smplx(full_pose=batch['full_pose'], shape_params=self.beta.expand(batch_size, -1), transl=batch.get('transl', torch.zeros([batch_size, 3], device=self.device)), offset=offset)
        trans_verts = self.cam_project(posed_verts, batch['cam'])

        output['offset'] = offset
        output['posed_verts'] = posed_verts
        output['trans_verts'] = trans_verts

        if returnVerts:
            return output
        if renderShape:
            shape_image = render_shape(vertices = trans_verts.detach(), faces = self.faces.expand(batch_size, -1, -1), 
                                    image_size=self.image_size, background=background)
            output['shape_image'] = shape_image
            if self.cfg.use_outer_mesh:
                shape_image_outer = render_shape(vertices = trans_verts_outer.detach(), faces = self.faces.expand(batch_size, -1, -1), 
                                    image_size=self.image_size, background=background)
                output['shape_outer_image'] = shape_image_outer
            return output
        if renderDepth:
            depth_verts = trans_verts.clone().detach()
            depth_verts[...,-1] = depth_verts[...,-1] + 10.
            depth_image, vis_image = pytorch3d_rasterize(vertices = depth_verts, faces = self.faces.expand(batch_size, -1, -1), image_size=self.image_size, 
                            blur_radius = 0.)
            depth_image = (depth_image - 10.)*vis_image
            output['depth_image'] = depth_image
            output['vis_image'] = vis_image

        ## render 
        trans_verts[:,:,:2] = -trans_verts[:,:,:2]
        trans_verts[:,:,2] = trans_verts[:,:,2] + 10
        faces = self.faces.unsqueeze(0).expand(batch_size,-1,-1)
        mesh = Meshes(
            verts = trans_verts,
            faces = faces,
            textures = TexturesVertex(verts_features=tex)
        )
        silhouette = self.silhouette_renderer(meshes_world=mesh).permute(0, 3, 1, 2)[:,3:]
        
        # image render
        trans_verts[:,:,:2] = -trans_verts[:,:,:2]
        trans_verts[:,:,2] = trans_verts[:,:,2] + 10
        attributes = util.face_vertices(tex, faces)
        if self.cfg.tex_detach_verts:
            trans_verts = trans_verts.detach()
        image = pytorch3d_rasterize(trans_verts, faces, image_size=self.image_size, attributes=attributes)
        alpha_image = image[:,[-1]]
        if self.cfg.dataset.white_bg:
            image = image[:,:3]*alpha_image + torch.ones_like(alpha_image)*(1-alpha_image)
        else:
            image = image[:,:3]*alpha_image 
        output['mesh_mask'] = silhouette
        output['mesh_image'] = image
        output['mesh'] = mesh
        output['mesh_offset'] = offset
        output['mesh_tex'] = tex

        return output
                
    def forward(self, batch, train=False, render_cloth=False, render_shape=False, render_background=False):
        batch_size = batch['cam'].shape[0]
        opdict = {}
        if train or render_background:
            background = batch['image']
        else:
            background = torch.ones([batch_size, 3, self.image_size, self.image_size], device=self.device)
        ## render inside body if using hybrid representation
        if self.cfg.use_mesh: 
            mesh_out = self.forward_mesh(batch, renderDepth=True, renderShape=render_shape, background=background)
        else:
            mesh_out = self.forward_mesh(batch, returnVerts=True)
            batch.update(mesh_out)
        opdict.update(mesh_out)
        if self.cfg.use_nerf:
            if self.cfg.use_mesh:
                batch.update(mesh_out)
            rays = self.sample_rays(batch, train=train)
            if train:
                nerf_out_image = self.forward_nerf(batch, rays, render_cloth=render_cloth)
            else:
                chunk = self.cfg.chunk
                normal_list = []
                depth_list = []
                rgbs_list = []
                alphas_list = []
                if self.cfg.use_fine and self.cfg.n_importance > 0:
                    rgbs_fine_list = []
                    alphas_fine_list = []
                if self.cfg.use_mesh:
                    mesh_image_sampled = batch['mesh_image_sampled'].clone()
                    if render_shape:
                        shape_image_sampled = batch['shape_image_sampled'].clone()
                for i in range(self.image_size**2//chunk):
                    chunk_idx = list(range(chunk*i, chunk*(i+1)))
                    if self.cfg.use_mesh:
                        batch['mesh_image_sampled'] = mesh_image_sampled[:,chunk_idx]
                        if render_shape:
                            batch['shape_image_sampled'] = shape_image_sampled[:,chunk_idx]
                    nerf_out = self.forward_nerf(batch, rays[:, chunk_idx], train=False, render_cloth=render_cloth, with_shape=render_shape)
                    rgbs_list.append(nerf_out['rgbs'])
                    alphas_list.append(nerf_out['alphas'])
                    if 'normal' in nerf_out.keys():
                        normal_list.append(nerf_out['normal'])
                    if self.cfg.use_fine and self.cfg.n_importance > 0 and not self.cfg.share_fine:
                        rgbs_fine_list.append(nerf_out['rgbs_fine'])
                        alphas_fine_list.append(nerf_out['alphas_fine'])
                    if 'depths_fine' in nerf_out.keys():
                        depth_list.append(nerf_out['depths_fine'])

                rgbs = torch.cat(rgbs_list, dim=1)
                alphas= torch.cat(alphas_list, dim=1)
                pixels = torch.cat([rgbs, alphas], -1) #[bz, n_rays, 4]
                if self.cfg.use_fine and self.cfg.n_importance > 0 and not self.cfg.share_fine:
                    rgbs_fine = torch.cat(rgbs_fine_list, dim=1)
                    alphas_fine = torch.cat(alphas_fine_list, dim=1)
                    pixels = torch.cat([rgbs, alphas, rgbs_fine, alphas_fine], -1) #[bz, n_rays, 4]

                pixel_image = pixels.reshape([batch_size, self.image_size, self.image_size, pixels.shape[-1]])
                pixel_image = pixel_image.permute(0,3,1,2).contiguous()
                nerf_out_image = {
                    'nerf_image': pixel_image[:,:3],
                    'nerf_mask': pixel_image[:,3:4]
                }
                if self.cfg.use_fine and self.cfg.n_importance > 0 and not self.cfg.share_fine:
                    nerf_out_image['nerf_fine_image'] = pixel_image[:,4:4+3]
                    nerf_out_image['nerf_fine_mask'] = pixel_image[:,4+3:]
                    nerf_image = nerf_out_image['nerf_fine_image']
                    nerf_mask = nerf_out_image['nerf_fine_mask']
                    shape_image = self.forward_mesh(batch, renderShape=True)['shape_image']
                    nerf_out_image['nerf_fine_hybrid_image'] = nerf_mask * nerf_image + (1-nerf_mask) * shape_image
            opdict.update(nerf_out_image)
        return opdict
    
    def canonical_normal(self, use_fine=False):
        # normal
        epsilon = 0.01
        points = self.verts.clone().detach()
        points += torch.randn_like(points) * self.cfg.dis_threshold * 0.5
        points_neighbs = points + torch.randn_like(points) * epsilon
        points_normal = self.query_canonical_space(points, use_fine=use_fine, only_normal=True)
        points_neighbs_normal = self.query_canonical_space(points_neighbs, use_fine=use_fine, only_normal=True)
        points_normal = points_normal / (torch.norm(points_normal, p=2, dim=-1, keepdim=True) + 1e-5)
        points_neighbs_normal = points_neighbs_normal / (torch.norm(points_neighbs_normal, p=2, dim=-1, keepdim=True) + 1e-5)
        return points_normal, points_neighbs_normal
        