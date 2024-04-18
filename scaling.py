import numpy as np
import random
from scipy.interpolate import griddata
import scipy
import cv2
import matplotlib.pyplot as plt
import scipy.optimize

def select_random_points(predicted_depth_image, actual_depth_image):
    height, width, channel = predicted_depth_image.shape
    random_indices = random.sample(range(height, width), 40)
    reshaped_pred_depth = predicted_depth_image.reshape(-1,3)[:,0]
    reshaped_actual_depth = actual_depth_image.reshape(-1,3)[:,0]

    center_y = height//2
    center_x = width//2
    # roi 
    predicted_depth_values = [reshaped_pred_depth[i] for i in random_indices]
    actual_depth_values = [reshaped_actual_depth[i] for i in random_indices]

    return random_indices, predicted_depth_values, actual_depth_values

def func(x,a,b,c):
    # values = []
    return a*x +b
    # return np.exp(b,a*x) +c
    # return  x**2 * a - b * x + c
    # for point in x: values.append(a*(point**2) + b*(point) + c)
    # return values

def interpolate_depth(predicted_depth_image, random_points, predicted_depth_values, actual_depth_values):
    # Get the coordinates of the random points
    random_x = [point for point in predicted_depth_values]
    random_x = np.vstack([random_x,np.ones(len(random_x))]).T
    random_y = [point for point in actual_depth_values]
    print(np.array(actual_depth_values).shape)
    popt, pcov = scipy.optimize.curve_fit(func, np.array(predicted_depth_values), np.array(actual_depth_values))
    # a,c= np.linalg.lstsq(random_x, random_y, rcond=None)[0]

    # Interpolate predicted depth values based on both predicted and actual depth values at random points

    return popt, pcov




actual_depth_image = cv2.imread("notebooks/absdepth000000.png")
predicted_depth_image = cv2.imread("notebooks/reldepth000000.png")

# actual_depth_image = actual_depth_image.resize((720,720))
# predicted_depth_image = predicted_depth_image.resize((720,720))
random_points, predicted_depth_values, actual_depth_values = select_random_points(predicted_depth_image, actual_depth_image)
plt.scatter(predicted_depth_values, actual_depth_values)
plt.show()
print(random_points)
print(predicted_depth_values)
print(actual_depth_values)


popt, pcov = interpolate_depth(predicted_depth_image, random_points, predicted_depth_values, actual_depth_values)
# print(a)
x = np.linspace(0, 254, 255)
print(x)
y = func(x, *popt)

# print(popt)


# print("predicted_depth_image")
predicted_depth_image = predicted_depth_image.reshape(-1,3)
actual_depth_image = actual_depth_image.reshape(-1,3)
print(predicted_depth_image[0:100,0])
print(actual_depth_image[0:100,0])
# print()
plt.scatter(x, y, label = "conversion")
plt.scatter(predicted_depth_values, actual_depth_values, label = "conversion")
plt.scatter(predicted_depth_image[:,0],actual_depth_image[:,0], label = "data")
# plt.scatter(predicted_depth_values, func(predicted_depth_values, *popt))
plt.legend()
plt.show()
# new_image = actual_depth_image*interpolated_depth[0] + interpolated_depth[1]
# cv2.imshow("image",new_image)
# cv2.imwrite("abs.jpg",new_image)
# # print(interpolated_depth)
# normalized_predicted_depth = (predicted_depth_image / 255.0) * interpolated_depth.max()




# for i, (point, predicted_depth, actual_depth) in enumerate(zip(random_points, predicted_depth_values, actual_depth_values), 1):
#     print(f"Point {i}: Pixel coordinates: {point}, Predicted depth: {predicted_depth}, Actual depth: {actual_depth}")
