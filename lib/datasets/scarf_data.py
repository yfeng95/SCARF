from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread
import cv2
import pickle
from tqdm import tqdm
import numpy as np
import torch
import os
from glob import glob
from ..utils import rotation_converter

class NerfDataset(torch.utils.data.Dataset):
    """SCARF Dataset"""

    def __init__(self, cfg, mode='train'):
        super().__init__()
        subject = cfg.subjects[0]
        imagepath_list = []
        self.dataset_path = os.path.join(cfg.path, subject)
        if not os.path.exists (self.dataset_path):
            print(f'{self.dataset_path} not exists, please check the data path')
            exit()
        imagepath_list = glob(os.path.join(self.dataset_path, 'image', f'{subject}_*.png'))
        root_dir = os.path.join(self.dataset_path, 'cache')
        os.makedirs(root_dir, exist_ok=True)
        self.pose_cache_path = os.path.join(root_dir, 'pose.pt')
        self.cam_cache_path = os.path.join(root_dir, 'cam.pt')
        self.exp_cache_path = os.path.join(root_dir, 'exp.pt')
        self.beta_cache_path = os.path.join(root_dir, 'beta.pt')
        self.subject_id = subject

        imagepath_list = sorted(imagepath_list) 
        frame_start = getattr(cfg, mode).frame_start
        frame_end = getattr(cfg, mode).frame_end
        frame_step = getattr(cfg, mode).frame_step
        imagepath_list = imagepath_list[frame_start:min(len(imagepath_list), frame_end):frame_step]

        self.data = imagepath_list
        if cfg.n_images < 10:
            self.data = self.data[:cfg.n_images]
        assert len(self.data) > 0, f"Can't find data; make sure data path {self.dataset_path} is correct"

        self.image_size = cfg.image_size
        self.white_bg = cfg.white_bg
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # load image
        imagepath = self.data[index]
        image = imread(imagepath) / 255.
        imagename = imagepath.split('/')[-1].split('.')[0]
        image = image[:, :, :3]
        frame_id = int(imagename.split('_f')[-1])
        frame_id = f'{frame_id:06d}'

        # load mask
        maskpath = os.path.join(self.dataset_path, 'matting', f'{imagename}.png')
        alpha_image = imread(maskpath) / 255.
        alpha_image = (alpha_image > 0.5).astype(np.float32)
        alpha_image = alpha_image[:, :, -1:]
        if self.white_bg:
            image = image[..., :3] * alpha_image + (1. - alpha_image)
        else:
            image = image[..., :3] * alpha_image
        # add alpha channel
        image = np.concatenate([image, alpha_image[:, :, :1]], axis=-1)
        image = resize(image, [self.image_size, self.image_size])
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = image[3:]
        image = image[:3]

        # load camera and pose
        frame_id = int(imagename.split('_f')[-1])
        name = self.subject_id

        # load pickle 
        pkl_file = os.path.join(self.dataset_path, 'pixie', f'{imagename}_param.pkl')
        with open(pkl_file, 'rb') as f:
            codedict = pickle.load(f)
        param_dict = {}
        for key in codedict.keys():
            if isinstance(codedict[key], str):
                param_dict[key] = codedict[key]
            else:
                param_dict[key] = torch.from_numpy(codedict[key])
        beta = param_dict['shape'].squeeze()[:10]
        # full_pose = param_dict['full_pose'].squeeze()
        jaw_pose = torch.eye(3, dtype=torch.float32).unsqueeze(0) #param_dict['jaw_pose']
        eye_pose = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2,1,1)
        # hand_pose = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(15,1,1)
        full_pose = torch.cat([param_dict['global_pose'], param_dict['body_pose'],
                            jaw_pose, eye_pose, 
                            # hand_pose, hand_pose], dim=0)        
                            param_dict['left_hand_pose'], param_dict['right_hand_pose']], dim=0)   
        cam = param_dict['body_cam'].squeeze()
        exp = torch.zeros_like(param_dict['exp'].squeeze()[:10])
        frame_id = f'{frame_id:06}'
        data = {
            'idx': index,
            'frame_id': frame_id,
            'name': name,
            'imagepath': imagepath,
            'image': image,
            'mask': mask,
            'full_pose': full_pose,
            'cam': cam,
            'beta': beta,
            'exp': exp
        }
       
        seg_image_path = os.path.join(self.dataset_path, 'cloth_segmentation', f"{imagename}.png")
        cloth_seg = imread(seg_image_path)/255.
        cloth_seg = resize(cloth_seg, [self.image_size, self.image_size])
        cloth_mask = torch.from_numpy(cloth_seg[:,:,:3].sum(-1))[None,...]
        cloth_mask = (cloth_mask > 0.1).float()
        cloth_mask = ((mask + cloth_mask) > 1.5).float()
        skin_mask = ((mask - cloth_mask) > 0).float()
        data['cloth_mask'] = cloth_mask
        data['skin_mask'] = skin_mask

        return data