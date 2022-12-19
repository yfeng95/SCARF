from ipaddress import ip_address
import os, sys
import argparse
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from glob import glob
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
import shutil
import torch
import cv2
import numpy as np
from PIL import Image

def get_palette(num_cls):
        """Returns the color map for visualizing the segmentation mask.
        Args:
            num_cls: Number of classes
        Returns:
            The color map
        """
        n = num_cls
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
                palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
                palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
                i += 1
                lab >>= 3
        return palette
        
def generate_frame(inputpath, savepath, subject_name=None, n_frames=2000, fps=30):
    ''' extract frames from video or copy frames from image folder
    '''
    os.makedirs(savepath, exist_ok=True)
    if subject_name is None:
        subject_name = Path(inputpath).stem
    ## video data
    if os.path.isfile(inputpath) and (os.path.splitext(inputpath)[-1] in ['.mp4', '.csv', '.MOV']):
        videopath = os.path.join(os.path.dirname(savepath), f'{subject_name}.mp4')
        logger.info(f'extract frames from video: {inputpath}..., then save to {videopath}')        
        vidcap = cv2.VideoCapture(inputpath)
        count = 0
        success, image = vidcap.read()
        cv2.imwrite(os.path.join(savepath, f'{subject_name}_f{count:06d}.png'), image) 
        h, w = image.shape[:2]
        print(h, w)
        # import imageio
        # savecap = imageio.get_writer(videopath, fps=fps)
        # savecap.append_data(image[:,:,::-1])
        while success:
            count += 1
            success,image = vidcap.read()
            if count > n_frames or image is None:
                break
            imagepath = os.path.join(savepath, f'{subject_name}_f{count:06d}.png')
            cv2.imwrite(imagepath, image)     # save frame as JPEG png
            # savecap.append_data(image[:,:,::-1])
        logger.info(f'extracted {count} frames')
    elif os.path.isdir(inputpath):
        logger.info(f'copy frames from folder: {inputpath}...')
        imagepath_list = glob(inputpath + '/*.jpg') +  glob(inputpath + '/*.png') + glob(inputpath + '/*.jpeg')
        imagepath_list = sorted(imagepath_list)
        for count, imagepath in enumerate(imagepath_list):
            shutil.copyfile(imagepath, os.path.join(savepath, f'{subject_name}_f{count:06d}.png'))
        print('frames are stored in {}'.format(savepath))
    else:
        logger.info(f'please check the input path: {inputpath}')
    logger.info(f'video frames are stored in {savepath}')

def generate_image(inputpath, savepath, subject_name=None, crop=False, crop_each=False, image_size=512, scale_bbox=1.1, device='cuda:0'):
    ''' generate image from given frame path. 
    '''
    logger.info(f'generae images, crop {crop}, image size {image_size}')
    os.makedirs(savepath, exist_ok=True)
    # load detection model
    from submodules.detector import FasterRCNN
    detector = FasterRCNN(device=device)
    if os.path.isdir(inputpath):
        imagepath_list = glob(inputpath + '/*.jpg') +  glob(inputpath + '/*.png') + glob(inputpath + '/*.jpeg')
        imagepath_list = sorted(imagepath_list)
        # if crop, detect the bbox of the first image and use the bbox for all frames
        if crop:
            imagepath = imagepath_list[0]
            logger.info(f'detect first image {imagepath}')
            imagename = os.path.splitext(os.path.basename(imagepath))[0]
            image = imread(imagepath)[:,:,:3]/255.
            h, w, _ = image.shape

            image_tensor = torch.tensor(image.transpose(2,0,1), dtype=torch.float32)[None, ...]
            bbox = detector.run(image_tensor)
            left = bbox[0]; right = bbox[2]; top = bbox[1]; bottom = bbox[3]
            np.savetxt(os.path.join(Path(inputpath).parent, 'image_bbox.txt'), bbox)
            
            ## calculate warping function for image cropping
            old_size = max(right - left, bottom - top)
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size*scale_bbox)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
            DST_PTS = np.array([[0,0], [0,image_size - 1], [image_size - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        for count, imagepath in enumerate(tqdm(imagepath_list)):
            if crop:
                image = imread(imagepath)
                dst_image = warp(image, tform.inverse, output_shape=(image_size, image_size))
                dst_image = (dst_image*255).astype(np.uint8)
                imsave(os.path.join(savepath, f'{subject_name}_f{count:06d}.png'), dst_image)
            else:
                shutil.copyfile(imagepath, os.path.join(savepath, f'{subject_name}_f{count:06d}.png'))
    logger.info(f'images are stored in {savepath}')

def generate_matting_rvm(inputpath, savepath, ckpt_path='assets/RobustVideoMatting/rvm_resnet50.pth', device='cuda:0'):
    sys.path.append('./submodules/RobustVideoMatting')
    from model import MattingNetwork
    EXTS = ['jpg', 'jpeg', 'png']
    segmentor = MattingNetwork(variant='resnet50').eval().to(device)
    segmentor.load_state_dict(torch.load(ckpt_path))

    images_folder = inputpath
    output_folder = savepath
    os.makedirs(output_folder, exist_ok=True)

    frame_IDs = os.listdir(images_folder)
    frame_IDs = [id.split('.')[0] for id in frame_IDs if id.split('.')[-1] in EXTS]
    frame_IDs.sort()
    frame_IDs = frame_IDs[:4][::-1] + frame_IDs

    rec = [None] * 4                                       # Initial recurrent 
    downsample_ratio = 1.0                                 # Adjust based on your video.   

    # bgr = torch.tensor([1, 1, 1.]).view(3, 1, 1).cuda() 
    for i in tqdm(range(len(frame_IDs))):
        frame_ID = frame_IDs[i]
        img_path = os.path.join(images_folder, '{}.png'.format(frame_ID))
        try:
            img_masked_path = os.path.join(output_folder, '{}.png'.format(frame_ID))
            img = cv2.imread(img_path)
            src = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            src = torch.from_numpy(src).float() / 255.
            src = src.permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                fgr, pha, *rec = segmentor(src.to(device), *rec, downsample_ratio)  # Cycle the recurrent states.
            pha = pha.permute(0, 2, 3, 1).cpu().numpy().squeeze(0)
            # check the difference of current 
            mask = (pha * 255).astype(np.uint8)
            img_masked = np.concatenate([img, mask], axis=-1)
            cv2.imwrite(img_masked_path, img_masked)
        except:
            os.remove(img_path)
            logger.info(f'matting failed for image {img_path}, delete it')
            
    sys.modules.pop('model')
    
def generate_cloth_segmentation(inputpath, savepath, ckpt_path='assets/cloth-segmentation/cloth_segm_u2net_latest.pth', device='cuda:0', vis=False):
    logger.info(f'generate cloth segmentation for {inputpath}')
    os.makedirs(savepath, exist_ok=True)
    # load model
    sys.path.insert(0, './submodules/cloth-segmentation')
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from data.base_dataset import Normalize_image
    from utils.saving_utils import load_checkpoint_mgpu
    from networks import U2NET

    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)

    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint_mgpu(net, ckpt_path)
    net = net.to(device)
    net = net.eval()

    palette = get_palette(4)

    images_list = sorted(os.listdir(inputpath))
    pbar = tqdm(total=len(images_list))
    for image_name in tqdm(images_list):
        img = Image.open(os.path.join(inputpath, image_name)).convert("RGB")
        image_tensor = transform_rgb(img)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

        output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
        output_img.putpalette(palette)
        name = Path(image_name).stem
        output_img.save(os.path.join(savepath, f'{name}.png'))
        pbar.update(1)
    pbar.close()
    
def generate_pixie(inputpath, savepath, ckpt_path='assets/face_normals/model.pth', device='cuda:0', image_size=512, vis=False):
    logger.info(f'generate pixie results')
    os.makedirs(savepath, exist_ok=True)
    # load model
    sys.path.insert(0, './submodules/PIXIE')
    from pixielib.pixie import PIXIE
    from pixielib.visualizer import Visualizer
    from pixielib.datasets.body_datasets import TestData
    from pixielib.utils import util
    from pixielib.utils.config import cfg as pixie_cfg
    from pixielib.utils.tensor_cropper import transform_points
    # run pixie
    testdata = TestData(inputpath, iscrop=False)
    pixie_cfg.model.use_tex = False
    pixie = PIXIE(config = pixie_cfg, device=device)
    visualizer = Visualizer(render_size=image_size, config = pixie_cfg, device=device, rasterizer_type='standard')
    testdata = TestData(inputpath, iscrop=False)
    for i, batch in enumerate(tqdm(testdata, dynamic_ncols=True)):
        batch['image'] = batch['image'].unsqueeze(0)
        batch['image_hd'] = batch['image_hd'].unsqueeze(0)
        name = batch['name']
        util.move_dict_to_device(batch, device)
        data = {
            'body': batch
        }
        param_dict = pixie.encode(data, threthold=True, keep_local=True, copy_and_paste=False)
        codedict = param_dict['body']
        opdict = pixie.decode(codedict, param_type='body')
        util.save_pkl(os.path.join(savepath, f'{name}_param.pkl'), codedict)
        if vis:
            opdict['albedo'] = visualizer.tex_flame2smplx(opdict['albedo'])
            visdict = visualizer.render_results(opdict, data['body']['image_hd'], overlay=True, use_deca=False)
            cv2.imwrite(os.path.join(savepath, f'{name}_vis.jpg'), visualizer.visualize_grid(visdict, size=image_size))       

def process_video(subjectpath, savepath=None, vis=False, crop=False, crop_each=False, ignore_existing=False, n_frames=2000):
    if savepath is None:
        savepath = Path(subjectpath).parent
    subject_name = Path(subjectpath).stem
    savepath = os.path.join(savepath, subject_name)
    os.makedirs(savepath, exist_ok=True)
    logger.info(f'processing {subject_name}')
    # 0. copy frames from video or image folder
    if ignore_existing or not os.path.exists(os.path.join(savepath, 'frame')):
        generate_frame(subjectpath, os.path.join(savepath, 'frame'), n_frames=n_frames)
        
    # 1. crop image from frames, use fasterrcnn for detection 
    if ignore_existing or not os.path.exists(os.path.join(savepath, 'image')):
        generate_image(os.path.join(savepath, 'frame'), os.path.join(savepath, 'image'), subject_name=subject_name,
                        crop=crop, crop_each=crop_each, image_size=512, scale_bbox=1.1, device='cuda:0')
    
    # 2. video matting
    if ignore_existing or not os.path.exists(os.path.join(savepath, 'matting')):
        generate_matting_rvm(os.path.join(savepath, 'image'), os.path.join(savepath, 'matting'))
        
    # 3. cloth segmentation
    if ignore_existing or not os.path.exists(os.path.join(savepath, 'cloth_segmentation')):
        generate_cloth_segmentation(os.path.join(savepath, 'image'), os.path.join(savepath, 'cloth_segmentation'), vis=vis)
    
    # 4. smplx estimation using PIXIE (https://github.com/yfeng95/PIXIE)
    if ignore_existing or not os.path.exists(os.path.join(savepath, 'pixie')):
        generate_pixie(os.path.join(savepath, 'image'), os.path.join(savepath, 'pixie'), vis=vis)
    logger.info(f'finish {subject_name}')
        
def main(args):    
    logger.add(args.logpath)
    
    with open(args.list, 'r') as f:
        lines = f.readlines()
    subject_list = [s.strip() for s in lines]
    if args.subject_idx is not None:
        if args.subject_idx > len(subject_list):
            print('idx error!')
        else:
            subject_list = [subject_list[args.subject_idx]]

    for subjectpath in tqdm(subject_list):
        process_video(subjectpath, savepath=args.savepath, vis=args.vis, crop=args.crop, crop_each=args.crop_each, ignore_existing=args.ignore_existing,
                      n_frames=args.n_frames)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate dataset from video or image folder')
    parser.add_argument('--list', default='lists/video_list.txt', type=str,
                        help='path to the subject data, can be image folder or video')
    parser.add_argument('--logpath', default='logs/generate_data.log', type=str,
                        help='path to save log')
    parser.add_argument('--savepath', default=None, type=str,
                        help='path to save processed data, if not specified, then save to the same folder as the subject data')
    parser.add_argument('--subject_idx', default=None, type=int,
                        help='specify subject idx, if None (default), then use all the subject data in the list') 
    parser.add_argument("--image_size", default=512, type=int,
                        help = 'image size')
    parser.add_argument("--crop", default=True, action="store_true",
                        help='whether to crop image according to the subject detection bbox')
    parser.add_argument("--crop_each", default=False, action="store_true",
                        help='TODO, whether to crop image according for each frame in the video')
    parser.add_argument("--vis", default=True, action="store_true",
                        help='whether to visualize labels (lmk, iris, face parsing)')
    parser.add_argument("--ignore_existing", default=False, action="store_true",
                        help='ignore existing data')
    parser.add_argument("--filter_data", default=False, action="store_true",
                        help='check labels, if it is not good, then delete image')
    parser.add_argument("--n_frames", default=400, type=int,
                        help='number of frames to be processed')
    args = parser.parse_args()

    main(args)

