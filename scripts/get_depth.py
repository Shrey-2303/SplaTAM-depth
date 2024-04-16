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
    model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{args.encoder}14').to(DEVICE).eval()

    transform = Compose([
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

    filenames = sorted([f for f in os.listdir(args.img_path) if f.startswith('frame') and f.endswith('.jpg')])

    for filename in tqdm(filenames):
        file_path = os.path.join(args.img_path, filename)
        raw_image = cv2.imread(file_path)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        h, w = image.shape[:2]
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth = model(image)
        depth = torch.nn.functional.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        new_filename = 'depth' + filename[5:-3] + 'png' 
        cv2.imwrite(os.path.join(args.outdir, new_filename), 255 - depth)