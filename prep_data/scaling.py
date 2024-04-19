import numpy as np
import random
from scipy.interpolate import griddata
import scipy
import cv2
import matplotlib.pyplot as plt
import scipy.optimize
import os
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

def get_pointcloud(width, height, distances, indicies, w2c, transform_pts=True, 
                   compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    import numpy as np
    w2c = numpy.eye(4)

    FX = 300
    FY = 300
    CX = 299.75
    CY = 169.75

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)[indicies]
    yy = yy.reshape(-1)[indicies]
    depth_z = distances
    # depth_z = depth[0].reshape(-1)
    # print(xx.device)
    # print(yy.device)
    # print(depth_z.device)
    # print(width, height)
    # # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld

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

def adjust_and_save_depth_images(base_path, output_path, gt_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # files = sorted([f for f in os.listdir(base_path) if f.startswith('depth') and f.endswith('.png')])
    files = sorted([f for f in os.listdir(base_path) if f.endswith('.png')])
    # print(type(files))
    scaler = Scaling()
    for filename in tqdm(files):
        file_path = os.path.join(base_path, filename)
        gt_file_path = os.path.join(gt_path, filename)
        # print(file_path)
        # print(gt_file_path)
        image = cv2.imread(file_path)
        gt_image = cv2.imread(gt_file_path)
        gt_image = gt_image
        # print(image.shape)
        # print(gt_image.shape)
        
        _, predicted_depth_values, actual_depth_values = point_sampling(image, gt_image, 1000)
        popt, pcov = scaler.get_optimal_conversion(predicted_depth_values, actual_depth_values)

        converted_image = scaler.convert(image)

        # blurred_image = cv2.GaussianBlur(converted_image, (5, 5), 10)

        output_file_path = os.path.join(output_path, filename)
        print(output_file_path)
        cv2.imwrite(output_file_path, converted_image)

        
        

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

    # Provide the path to the saved dithered image
    output_path




if __name__ == "__main__":
    base_path = ".\\notebooks\\rel"
    output_path = "."
    gt_path = ".\\notebooks\\gt"
    # adjust_and_save_depth_images(base_path, output_path, gt_path)
    # # smoothening()
    # exit()
    scaler = Scaling()
    files  = ["depth000000.png"] 
    for filename in files:
        file_path = os.path.join(base_path, filename)
        gt_file_path = os.path.join(gt_path, filename)
        print('estimated', file_path)
        print('gt', gt_file_path)
        image = cv2.imread(file_path)
        gt_image = cv2.imread(gt_file_path)
        # gt_image = gt_image//(2**7)

        # cv2.imwrite("1.png",image)
        # cv2.imwrite("2.png",gt_image)
        # cv2.waitkey(10)

        _, predicted_depth_values, actual_depth_values = point_sampling(image, gt_image, 300)
        popt, pcov = scaler.get_optimal_conversion(predicted_depth_values, actual_depth_values)
        # print(scaler.initialized)

        # plt.scatter(predicted_depth_values, actual_depth_values)
        # # plt.savefig("output.png")
        # plt.show()



        x = np.linspace(0, 255, 256)
        print(x)
        y = scaler.func(x, *popt)
        predicted_depth_image = image.reshape(-1,3)
        actual_depth_image = gt_image.reshape(-1,3)
        # # print()
        # plt.scatter(x, y, label = "conversion")
        # plt.scatter(predicted_depth_values, actual_depth_values, label = "conversion")
        # plt.xlabel = "AnyDepth disparity depth image values"
        # plt.ylabel = "corresponding lidar depth values"
        plt.xlabel("Depth Anything disparity depth image values")
        plt.ylabel("corresponding depth image values")
        plt.scatter(predicted_depth_image[:,0],actual_depth_image[:,0], 3, label = "data")
        plt.plot(x, y, c= "red", label = "transformation function")

        # plt.scatter(predicted_depth_values, func(predicted_depth_values, *popt))
        plt.legend()
        plt.show()

        converted_image = scaler.convert(image)
        cv2.imwrite("output_image.png", converted_image)
        cv2.imshow("output_image", converted_image)
        # while True:
        #     cv2.waitKey(1)
        






