import os
import random
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import label as nd_label
from cmbnet.utils import utils_plotting, utils_general
from cmbnet.preprocessing.loading import get_metadata_from_cmb_format

# Data directories
DATADIRS = [
    "/home/cerebriu/data/datasets/processed_final/cmb_dou/Data",
    "/home/cerebriu/data/datasets/processed_final/cmb_valid/Data",
    "/home/cerebriu/data/datasets/processed_final/cmb_crb/Data",
]
SEED = 42


def process_images(args, datadirs, seed):
    for datadir in datadirs:
        for i in range(args.n_images):
            random.seed(seed + i)
            subject = random.choice(
                [
                    d
                    for d in os.listdir(datadir)
                    if "-H" not in d and os.path.isdir(os.path.join(datadir, d))
                ]
            )
            metadata = get_metadata_from_cmb_format(datadir, subject)
            analyze_image(args, metadata, subject)


def analyze_image(args, metadata, subject):
    im = nib.load(metadata["anno_path"])
    im_data = im.get_fdata()

    cmb_id = random.choice(list(metadata["CMBs_new"].keys()))
    com = metadata["CMBs_new"][cmb_id]["CM"]
    rad = metadata["CMBs_new"][cmb_id]["radius"]

    print("....")
    print(subject, cmb_id, com, rad)
    print("....")

    analyze_and_crop_im_data(args, im_data, com, rad, subject)


def analyze_and_crop_im_data(args, im_data, com, rad, subject):
    labeled_im_data, num_labels = nd_label(im_data)
    connected_component = labeled_im_data == labeled_im_data[tuple(com)]
    masked_im_data = im_data * connected_component

    n_vox = np.sum(masked_im_data > 0)
    rad2 = np.power((n_vox / (4 / 3 * np.pi)), 1 / 3)
    print(f"Radius: {rad}")
    print(f"Radius2: {rad2}")

    crop_size = args.crop_size
    cropped_im_data = crop_data_around_center(masked_im_data, com, crop_size)

    print(cropped_im_data.shape)
    filepath = f"{args.savedir}/{subject}_{com}_{rad}.png"
    if args.engine == "mayavi":
        utils_plotting.plot_microbleed_mayavi(
            cropped_im_data, filepath=filepath, im_size=(500, 500)
        )
    elif args.engine == "vtk":
        utils_plotting.plot_microbleed_vtk(
            cropped_im_data, filepath=filepath, im_size=(500, 500)
        )
    else:
        raise ValueError("Invalid engine")


def crop_data_around_center(data, center, size):
    crop_start = [int(c - size / 2) for c in center]
    crop_end = [int(c + size / 2) for c in center]
    return data[
        crop_start[0] : crop_end[0],
        crop_start[1] : crop_end[1],
        crop_start[2] : crop_end[2],
    ]


def parse_args():
    """
    Parses all script arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--savedir", type=str, default=None, help="Path to the input directory"
    )
    parser.add_argument(
        "--n_images", type=int, default=1, help="Number of images per datadir"
    )
    parser.add_argument(
        "--engine", type=str, default="mayavi", help="Plotting engine (mayavi or vtk)"
    )
    parser.add_argument("--crop_size", type=int, default=50, help="Crop size")  
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_images(args, DATADIRS, SEED)
