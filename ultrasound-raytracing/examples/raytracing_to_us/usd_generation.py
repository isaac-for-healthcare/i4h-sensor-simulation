import os
import argparse
import tempfile
import numpy as np

import vtk
import vtkmodules

from utility import convert_to_mesh, convert_mesh_to_usd

from monai.config import print_config
from monai.bundle.scripts import create_workflow, download
from monai.transforms import LoadImaged, SaveImage, Compose, BorderPadd, SqueezeDimd


def nii_to_mesh(input_nii_path, output_nii_path, output_obj_path):
    """
    This function converts each organ into a separate OBJ file and generates a GLTF file
    containing all organs with hierarchical structure.
    It processes the input NIfTI file and groups 140 labels into 17 categories.

    Args:
        input_nii_path: path to the nii file
        output_nii_path: path to save the obj files
        output_obj_path: path to save the gltf file
    """
    if not os.path.exists(output_nii_path):
        os.makedirs(output_nii_path)

    labels = {
        "Liver": 255,
    }

    pre_trans = Compose(
        [
            LoadImaged(keys="label", ensure_channel_first=True),
            SqueezeDimd(keys="label", dim=0),
        ]
    )
    orig_seg = pre_trans({"label": input_nii_path})["label"]
    all_organ = np.zeros_like(orig_seg, dtype=np.uint8)
    all_label_values = {}

    save_trans = SaveImage(output_ext="nii.gz", output_dtype=np.uint8)
    for j, (organ_name, label_val) in enumerate(labels.items(), start=1):
        single_organ = np.zeros_like(orig_seg, dtype=np.uint8)
        print(f"Assigning index {j} to label {organ_name}")
        if isinstance(label_val, dict):
            for _, i in label_val.items():
                all_organ[orig_seg == i] = j
                single_organ[orig_seg == i] = j
        else:
            all_organ[orig_seg == label_val] = j
            single_organ[orig_seg == label_val] = j
        organ_filename = os.path.join(output_nii_path, organ_name)
        save_trans(single_organ[None], meta_data=orig_seg.meta, filename=organ_filename)

        convert_to_mesh(
            f"{organ_filename}.nii.gz",
            output_obj_path,
            f"{organ_name}.obj",
            label_value=j,
            smoothing_factor=0.5,
            reduction_ratio=0.0,
        )
        all_label_values[j] = organ_name

    all_organ_filename = os.path.join(output_nii_path, "all_organs")
    save_trans(all_organ[None], meta_data=orig_seg.meta, filename=all_organ_filename)
    convert_to_mesh(
        f"{all_organ_filename}.nii.gz",
        output_obj_path,
        "all_organs.gltf",
        label_value=all_label_values,
        smoothing_factor=0.6,
        reduction_ratio=0.0,
    )
    print(f"Saved whole segmentation {all_organ_filename}")


def generate_mesh(seg_dir):
    ct_list = os.listdir(os.path.join(seg_dir))
    ct_list = [ct for ct in ct_list if ct.endswith(".nii.gz")]
    for ct in ct_list:
        print(f"Processing {ct}")
        try:
            ct_name = ct.split(".nii.gz")[0]
            input_nii_path = os.path.join(seg_dir, ct)
            output_nii_path = os.path.join(seg_dir, ct_name, "nii")
            output_obj_path = os.path.join(seg_dir, ct_name, "obj")
            _ = nii_to_mesh(input_nii_path, output_nii_path, output_obj_path)

        except Exception as e:
            print(f"Error processing {ct}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mesh from CT data.")
    parser.add_argument('--seg_dir', type=str, required=True)
    args = parser.parse_args()

    generate_mesh(args.seg_dir)
