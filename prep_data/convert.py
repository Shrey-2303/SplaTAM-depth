import cv2
import numpy as np
import os
from tqdm import tqdm

def load_first_image(base_path):

    first_image_path = os.path.join(base_path, 'depth000000.png')
    first_image = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)
    if first_image is None:
        raise FileNotFoundError("The first depth image does not exist in the directory.")
    
    original_path = 'depth000000.png'
    original_image = cv2.imread(original_path)
    return first_image, original_image


def adjust_and_save_depth_images(base_path, output_path, first_image, original_image):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    i = 0
    files = sorted([f for f in os.listdir(base_path) if f.startswith('depth') and f.endswith('.png')])
    min_first = first_image.min()
    max_first = first_image.max()
    for filename in tqdm(files):
        file_path = os.path.join(base_path, filename)
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        # print(image.max())
        if image is None:
            continue

        # Normalize the current image based on the reference depth from the first image
        image = image.astype(np.float32)
        # image = 255 - image
        # image[image >= 1e-4] = 1/image[image >= 1e-4]

        # reference_depth = np.mean(first_image)
        # current_reference_depth = np.mean(image)
        # scale_factor = reference_depth / current_reference_depth
        # print(scale_factor)
        # print(image.max(), image.min())
        normalized_image = image #* scale_factor
        min_val = np.min(image)
        max_val = np.max(image)
        normalized_image =  (image - min_val) * (256 / (max_val - min_val))
        normalized_image = normalized_image.astype(np.uint8)
        # print(normalized_image.max())
        # print(normalized_image.min())
        # print(original_image.max())
        # print(original_image.min())

        # map_to_first = (normalized_image - min_first) / (max_first - min_first)
        # min_val = np.min(normalized_image)
        # max_val = np.max(normalized_image)
        # normalized_image =  (normalized_image - min_val) / (max_val - min_val) * 255
        # # print(normalized_image.max(), normalized_image.min())
        # # Save the normalized image


        output_file_path = os.path.join(output_path, filename)
        cv2.imwrite(output_file_path, normalized_image)  

        # i +=1 
        # if i ==2:break

def main():
    base_path = 'output'
    output_path = 'normalized'
    first_image, original_image = load_first_image(base_path)

    adjust_and_save_depth_images(base_path, output_path, first_image, original_image)

if __name__ == '__main__':
    main()