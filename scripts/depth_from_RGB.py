import sys
sys.path.append('./')
import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
import time
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from PIL import Image
class GetDepth():
    def __inti__(self, model_pth = 'depth_anything_vitl14.pth', encoder = "vitl"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        }
        # encoder = 'vitl' # or 'vitb', 'vits'
        self.depth_anything = DepthAnything(model_configs[encoder]).to(self.device).eval()
        self.depth_anything.load_state_dict(torch.load('depth_anything_vitl14.pth'))

        self.transform = Compose([
            Resize(width=518, 
                height=518, 
                resize_target=False, 
                keep_aspect_ratio=True,
                ensure_multiple_of=14, 
                resize_method='lower_bound', 
                image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def get_depth_from_file(self, filename, grayscale = True):
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        self.get_depth_from_image(image)
    
    def get_depth_from_image(self, image, grayscale = False):
        h, w = image.shape[:2]
        image = self.transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth = self.depth_anything(image)
        depth = torch.nn.functional.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)

        if grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        return 255 - depth
    

if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, required=True, help='Directory containing the images')
    parser.add_argument('--outdir', type=str, default='./vis_depth', help='Directory to save depth images')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'], help='Model encoder type')
    parser.add_argument('--grayscale', action='store_true', help='Apply grayscale color map instead of inferno')
    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    depth_estimator = GetDepth()
    
    filenames = sorted([f for f in os.listdir(args.img_path) if f.startswith('frame') and f.endswith('.jpg')])
    for filename in tqdm(filenames):
        file_path = os.path.join(args.img_path, filename)

        depth = depth_estimator(file_path, grayscale = args.grayscale)

        new_filename = 'depth' + filename[5:-3] + 'png' 
        cv2.imwrite(os.path.join(args.outdir, new_filename), 255 - depth)

