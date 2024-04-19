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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, required=True, help='Directory containing the images')
    parser.add_argument('--outdir', type=str, default='./vis_depth', help='Directory to save depth images')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'], help='Model encoder type')
    parser.add_argument('--grayscale', action='store_true', help='Apply grayscale color map instead of inferno')
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(DEVICE)
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }

    encoder = 'vitl' # or 'vitb', 'vits'
    depth_anything = DepthAnything(model_configs[encoder]).to(DEVICE).eval()
    depth_anything.load_state_dict(torch.load('depth_anything_vitl14.pth'))

    transform = Compose([
        Resize(width=700, 
               height=700, 
               resize_target=False, 
               keep_aspect_ratio=True,
               ensure_multiple_of=14, 
               resize_method='lower_bound', 
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    filenames = sorted([f for f in os.listdir(args.img_path) if f.endswith('.png')])#f.startswith('frame') and f.endswith('.jpg')])

    for filename in tqdm(filenames):
        file_path = os.path.join(args.img_path, filename)
        raw_image = cv2.imread(file_path)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        h, w = image.shape[:2]
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth = depth_anything(image)


        depth = torch.nn.functional.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        new_filename = filename[:-3] + 'png' 
        # new_filename = 'depth' + filename[5:-3] + 'png' 
        cv2.imwrite(os.path.join(args.outdir, new_filename), 255 - depth)


        # depth = torch.nn.functional.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        # disp1 = depth.cpu().numpy()
        # range1 = np.minimum(disp1.max() / (disp1.min() + 0.001), 100.0)
        # max1 = disp1.max()
        # min1 = max1 / range1

        # depth1 = 1 / np.maximum(disp1, min1)
        # depth1 = (depth1 - depth1.min()) / (depth1.max() - depth1.min())
        # gamma = 255.0
        # depth1 = depth1 * gamma
        # depth1 = np.repeat(depth1[..., np.newaxis], 3, axis=-1)
        
        # # new_filename = 'depth' + filename[5:-3] + 'png' 
        # new_filename = filename[:-3] + 'png' 
        # # cv2.imwrite(os.path.join(args.outdir, new_filename), 255 - depth)
        # cv2.imwrite(os.path.join(args.outdir, new_filename), depth1)