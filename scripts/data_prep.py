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
import random
from scipy.interpolate import griddata
import scipy
import matplotlib.pyplot as plt
import scipy.optimize
from tqdm import tqdm



def point_sampling(predicted_depth_image, actual_depth_image, num_points):
    height, width, channel = predicted_depth_image.shape
    random_indices = random.sample(range(0, height*width), num_points)
    # print(actual_depth_image.reshape(-1,3)[:,0].shape)
    # exit()
    reshaped_pred_depth = predicted_depth_image.reshape(-1,3)[:,0]
    reshaped_actual_depth = actual_depth_image.reshape(-1,3)[:,0]
    predicted_depth_values = [reshaped_pred_depth[i] for i in random_indices]
    actual_depth_values = [reshaped_actual_depth[i] for i in random_indices]


    return random_indices, predicted_depth_values, actual_depth_values

def ceneter_square_sampling(predicted_depth_image, actual_depth_image, box_size):
    height, width, channel = predicted_depth_image.shape
    center_x = height//2 
    center_y = width//2

    box_len = box_size
    predicted_depth_values = (predicted_depth_image[center_x - box_len:center_x + box_len,center_y - box_len:center_y+ box_len]).reshape(-1)
    actual_depth_values = (actual_depth_image[center_x - box_len:center_x + box_len,center_y - box_len:center_y+ box_len]).reshape(-1)
    # print(actual_depth_values)
    # print(actual_depth_values)
    return [], predicted_depth_values, actual_depth_values

class Scaling():
    def __init__(self):
        self.popt = []
        self.pcov = []
        self.initialized = False
    
    def func(sef,x,a,b,c):
        return a*np.log(b*x+c)
    
    def convert(self, x):
        if not self.initialized:
            return False
        else:
            return self.func(x, *self.popt)


    def get_optimal_conversion(self, predicted_depth_values, actual_depth_values):
        try:
            popt, pcov = scipy.optimize.curve_fit(self.func, np.array(predicted_depth_values).astype(float), np.array(actual_depth_values).astype(float))
            self.popt = popt
            self.pcov = pcov
            self.initialized = True
        except:
            print("optimization failed")
            

        return self.popt, self.pcov
    
    def eval_conversion(self, predicted_depth_image, actual_depth_image):
        x = np.linspace(0, 255, 255)
        y = self.func(x, *self.popt)

        predicted_depth_image = predicted_depth_image.reshape(-1,3)
        actual_depth_image = actual_depth_image.reshape(-1,3)

        plt.scatter(x, y, label = "conversion")
        plt.scatter(predicted_depth_image[:,0],actual_depth_image[:,0], label = "data")
        plt.legend()
        plt.show()

        # show image
        new_image = self.func(predicted_depth_image, *self.popt)
        new_image[np.isnan(new_image)] = 0
        cv2.imshow("image",new_image)
        cv2.waitkey(0)


        
        

def smoothening():
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    # Load the image
    image_path = '../SplaTAM/experiments/iPhone_Captures/offline_demo/depth/0.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(image)
    # Check if the image was loaded correctly
    if image is None:
        raise ValueError("Could not load the image. Check the path.")

    # Apply dithering using Floyd–Steinberg dithering to reduce banding
    def floyd_steinberg_dithering(image):
        img_float = image.astype(np.float32) / 255.0
        fs_filter = np.array([[0, 0, 7],
                            [3, 5, 1]]) / 16
        for y in range(image.shape[0] - 1):
            for x in range(1, image.shape[1] - 1):
                old_pixel = img_float[y, x]
                new_pixel = np.round(old_pixel)
                img_float[y, x] = new_pixel
                quant_error = old_pixel - new_pixel
                for dy in range(2):
                    for dx in range(-1, 2):
                        if fs_filter[dy, dx + 1] is not None:
                            img_float[y + dy, x + dx] += quant_error * fs_filter[dy, dx + 1]
        return (np.clip(img_float, 0, 1) * 255).astype(np.uint8)

    dithered_image = floyd_steinberg_dithering(image)
    output_path = '/mnt/data/dithered_image.png'
    cv2.imwrite(output_path, dithered_image)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(dithered_image, cmap='gray')
    plt.title('Dithered Image')
    plt.axis('off')

    plt.show()

    output_path




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, required=True, default = 'data/Replica/room0/results', help='Directory containing the images')
    parser.add_argument('--outdir', type=str, default='../converted_depth', help='Directory to save converted depth images')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl', 'vitl'], help='Model encoder type')
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
    scaler = Scaling()
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
        disp1 = depth.cpu().numpy()
        range1 = np.minimum(disp1.max() / (disp1.min() + 0.001), 100.0)
        max1 = disp1.max()
        min1 = max1 / range1

        depth1 = 1 / np.maximum(disp1, min1)
        depth1 = (depth1 - depth1.min()) / (depth1.max() - depth1.min())
        gamma = 255.0
        depth1 = depth1 * gamma
        depth1 = np.repeat(depth1[..., np.newaxis], 3, axis=-1)
        
        new_filename = 'depth' + filename[5:-3] + 'png' 


        gt_file_path = os.path.join(args.img_path, new_filename)
        

        gt_image = cv2.imread(gt_file_path)
        gt_image = gt_image

        
        _, predicted_depth_values, actual_depth_values = point_sampling(depth1, gt_image, 1000)
        popt, pcov = scaler.get_optimal_conversion(predicted_depth_values, actual_depth_values)

        converted_image = scaler.convert(image)

        cv2.imwrite(os.path.join(args.outdir, new_filename), converted_image)




