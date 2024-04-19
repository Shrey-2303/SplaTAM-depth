import numpy as np
import cv2
import matplotlib.pyplot as plt

img_ref = cv2.imread('./frame000000.jpg')[:,:,::-1]
img_next = cv2.imread('./frame000020.jpg')[:,:,::-1]
depth_ref = cv2.imread('./reldepth000000.png')[:,:,::-1]
depth_next = cv2.imread('./reldepth000020.png')[:,:,::-1]

depth_abs_0 = cv2.imread("./absdepth000000.png")

import numpy as np

def get_pointcloud(color, depth, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    width, height = color.shape[1], color.shape[0]
    # CX = intrinsics[0][2]
    # CY = intrinsics[1][2]
    # FX = intrinsics[0][0]
    # FY = intrinsics[1][1]

    # FX = 300
    # FY = 300
    # CX = 299.75
    # CY = 169.75


    FX = 600.0
    FY = 600.0
    CX = 599.5
    CY = 339.5

    # Compute indices of pixels
    x_grid, y_grid = np.meshgrid(np.arange(width), 
                                    np.arange(height),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth.reshape(-1)

    print(width, height)
    # Initialize point cloud
    pts_cam = np.stack((xx * depth_z, yy * depth_z, depth_z))
    print(pts_cam.shape)
    if transform_pts:
        pix_ones = np.ones((height * width, 1))
        pts4 = np.concatenate((pts_cam, pix_ones), axis=1)
        c2w = np.linalg.inv(w2c)
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
    # cols = np.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    # point_cld = np.cat((pts, cols), -1)
    point_cld = pts

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld
    


w2c = np.eye(4)
depth = depth_abs_0[:,:,0]
# print(img_ref.shape)
color = img_ref
# color = color.to("cuda")
# print(color.shape)
# print(color.device)
# print(depth.shape)

ptcld = get_pointcloud(color, depth, w2c, transform_pts=False)


# print(ptcld)
# print(ptcld.shape)
# ptcld_np = np.array(ptcld.cpu())

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111, projection='3d')

import random
# print(ptcld[0:100,:])
length = ptcld.shape[1]
random_indices = random.sample(range(0, length), 400)
ptcld_filtered = np.array([[ptcld[0,i],ptcld[1,i],ptcld[2,i]] for i in random_indices])
print(ptcld_filtered)

ax.scatter(ptcld_filtered[:,0], ptcld_filtered[:,1], ptcld_filtered[:,2], s = 5, alpha = 0.6,c = ptcld_filtered[:,2] ,  cmap = "viridis")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()