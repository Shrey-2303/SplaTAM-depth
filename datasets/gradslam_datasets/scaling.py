import numpy as np
import random
from scipy.interpolate import griddata
import scipy
import cv2
import matplotlib.pyplot as plt
import scipy.optimize



def center_square(predicted_depth_image, actual_depth_image, length = 10):
    height, width, channel = predicted_depth_image.shape
    random_indices = random.sample(range(0, height*width), 2000)
    # reshaped_pred_depth = predicted_depth_image.reshape(-1,3)[:,0]
    # reshaped_actual_depth = actual_depth_image.reshape(-1,3)[:,0]
    # predicted_depth_values = [reshaped_pred_depth[i] for i in random_indices]
    # actual_depth_values = [reshaped_actual_depth[i] for i in random_indices]

    center_x = height//2 
    center_y = width//2

    box_len = length
    predicted_depth_values = (predicted_depth_image[center_x - box_len:center_x + box_len,center_y - box_len:center_y+ box_len]).reshape(-1)
    actual_depth_values = (actual_depth_image[center_x - box_len:center_x + box_len,center_y - box_len:center_y+ box_len]).reshape(-1)
    print(actual_depth_values)
    print(actual_depth_values)

    return random_indices, predicted_depth_values, actual_depth_values

def func(x,a,b,c):
    # values = []
    return a*np.log(b*x+c)
    # return np.exp(b,a*x) +c
    # return  x**2 * a - b * x + c

def interpolate_depth(predicted_depth_image, random_points, predicted_depth_values, actual_depth_values):
    random_x = [point for point in predicted_depth_values]
    random_x = np.vstack([random_x,np.ones(len(random_x))]).T
    random_y = [point for point in actual_depth_values]
    print(np.array(actual_depth_values).shape)
    popt, pcov = scipy.optimize.curve_fit(func, np.array(predicted_depth_values), np.array(actual_depth_values))

    return popt, pcov

# actual_depth_image = cv2.imread("notebooks/absdepth000000.png")
# predicted_depth_image = cv2.imread("notebooks/depth000000_an.png")
# random_points, predicted_depth_values, actual_depth_values = select_random_points(predicted_depth_image, actual_depth_image)
# plt.scatter(predicted_depth_values, actual_depth_values)
# plt.show()

# popt, pcov = interpolate_depth(predicted_depth_image, random_points, predicted_depth_values, actual_depth_values)
# # print(a)
# x = np.linspace(0, 254, 255)
# print(x)
# y = func(x, *popt)

# print(popt)


# print("predicted_depth_image")
# predicted_depth_image = predicted_depth_image.reshape(-1,3)
# actual_depth_image = actual_depth_image.reshape(-1,3)
# print(predicted_depth_image[0:100,0])
# print(actual_depth_image[0:100,0])
# # # print()
# plt.scatter(x, y, label = "conversion")
# plt.scatter(predicted_depth_values, actual_depth_values, label = "conversion")
# plt.scatter(predicted_depth_image[:,0],actual_depth_image[:,0], label = "data")
# # plt.scatter(predicted_depth_values, func(predicted_depth_values, *popt))
# plt.legend()
# plt.show()


# plt.hist(predicted_depth_image[:,0], bins = 100)
# # plt.show()
# plt.hist(actual_depth_image[:,0], bins = 100)
# plt.show()

# new_image = actual_depth_image*interpolated_depth[0] + interpolated_depth[1]
# cv2.imshow("image",new_image)
# cv2.imwrite("abs.jpg",new_image)
# print(interpolated_depth)
# normalized_predicted_depth = (predicted_depth_image / 255.0) * interpolated_depth.max()

