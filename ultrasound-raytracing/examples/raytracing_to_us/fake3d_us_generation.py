import argparse
import cv2
import numpy as np
import os
import nibabel as nib

def produce_image_and_mask(image_name, image_dir, segmentation_dir):
    # read image
    image_path = os.path.join(image_dir, image_name)
    mask_path = os.path.join(segmentation_dir, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path)
    # mask is a 3 channel image, (255, 255, 255) is the utline, (0, 255, 0) is the liver
    liver_color_bgr = np.array([0, 255, 0], dtype=np.uint8)
    # outline_color_bgr = np.array([255, 255, 255], dtype=np.uint8)

    # Create a binary mask for the liver
    mask_liver = np.all(mask == liver_color_bgr, axis=-1)
    mask_liver = mask_liver.astype(np.uint8) * 255

    non_zero_indices = np.argwhere(mask_liver > 0)
    min_coords = non_zero_indices.min(axis=0)
    max_coords = non_zero_indices.max(axis=0)

    # Store bounding box information
    bounding_box_info = {
        'min_x': int(min_coords[1]),
        'min_y': int(min_coords[0]),
        'max_x': int(max_coords[1]),
        'max_y': int(max_coords[0])
    }

    return image, mask_liver, bounding_box_info

def main(args):
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    image_list = os.listdir(image_dir)
    
    for image_name in image_list:
        image, mask, bounding_box_info = produce_image_and_mask(image_name, image_dir, mask_dir)
        
        # simply stack the mask 100 times
        mask_3d = np.stack([mask] * 100, axis=2)

        # define spacing
        x_len = bounding_box_info['max_x'] - bounding_box_info['min_x']
        y_len = bounding_box_info['max_y'] - bounding_box_info['min_y']
        
        affine = np.eye(4)
        if x_len < 400:
            affine[0, 0] = -1 * 0.5
            affine[1, 1] = 0.5
        else:
            affine[0, 0] = -1 * 0.25
            affine[1, 1] = 0.25
        
        affine[2, 2] = 1
        data = nib.Nifti1Image(mask_3d, affine)
        os.makedirs(args.output_dir, exist_ok=True)
        nib.save(data, os.path.join(args.output_dir, image_name.replace(".png", ".nii.gz")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fake 3D ultrasound images and masks.")
    parser.add_argument('--image_dir', required=True, help='Path to the image directory.')
    parser.add_argument('--mask_dir', required=True, help='Path to the mask directory.')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory.')

    args = parser.parse_args()
    main(args)
