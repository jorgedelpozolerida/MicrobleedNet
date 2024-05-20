import os
import random
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import label as nd_label
from cmbnet.utils import utils_plotting, utils_general
from cmbnet.utils.utils_general import get_metadata_from_cmb_format
import glob


SEED = 1234


def process_images(args, seed):
    datadir = args.datadir
    analyzed_subjects = []

    all_subs = [
        d
        for d in os.listdir(datadir)
        if "-H" not in d
        and os.path.isdir(os.path.join(datadir, d))
        and "_pred" not in d
    ]
    all_subs_dat = {}
    for sub in all_subs:
        dataset_name = sub.split("-")[0]
        if dataset_name not in all_subs_dat:
            all_subs_dat[dataset_name] = []
        all_subs_dat[dataset_name].append(sub)

    for dataset_name, subs in all_subs_dat.items():
        # for i in range(args.n_images):
        #     random.seed(seed + i)
        #     subject = random.choice(subs)
        for subject in subs:
            metadata = get_metadata_from_cmb_format(datadir, subject)
            analyze_image(args, metadata, subject)
            analyzed_subjects.append(subject)

    if args.prediction_dir:
        subjects = analyzed_subjects
        metadata_pred = []
        for sub in subjects:
            pred_files = glob.glob(
                os.path.join(args.prediction_dir, sub, f"**/{sub}_PRED.nii.gz"),
                recursive=True,
            )
            if len(pred_files) == 0:
                print(f"WARNING: No prediction files found for {sub}")
            elif len(pred_files) > 1:
                print(f"WARNING: Multiple prediction files found for {sub}")
            assert os.path.exists(pred_files[0])
            sub_meta = {"id": sub, "pred_path": pred_files[0]}
            metadata_pred.append(sub_meta)

            gt_meta = get_metadata_from_cmb_format(datadir, sub)

            analyze_image(args, gt_meta, sub, pred_meta=sub_meta)


def analyze_image(args, metadata, subject, pred_meta=None):
    im = nib.load(metadata["anno_path"])
    im_data = im.get_fdata()

    # cmb_id = random.choice(list(metadata["CMBs_new"].keys()))

    for cmb_id in metadata["CMBs_new"]:
        com = metadata["CMBs_new"][cmb_id]["CM"]
        rad = metadata["CMBs_new"][cmb_id]["radius"]

        if pred_meta:
            pred_im = nib.load(pred_meta["pred_path"])
            pred_im_data = np.squeeze(pred_im.get_fdata())
            n_vox = np.sum(pred_im_data > 0)
            rad = (n_vox * (3 / (4 * np.pi))) ** (1 / 3)
            rad = round(rad, 2)
            print(pred_im_data.shape)
            pred_im_data[pred_im_data > 0] = 1
            analyze_and_crop_im_data(args, pred_im_data, com, rad, subject, pred=True)
        else:
            analyze_and_crop_im_data(args, im_data, com, rad, subject)


def analyze_and_crop_im_data(args, im_data, com, rad, subject, pred=False):

    # Label all connected components and extract the component including the center of mass
    labeled_im_data, num_labels = nd_label(im_data)
    connected_component = (
        labeled_im_data == labeled_im_data[tuple(map(int, com))]
    )  # Ensure com is used as integers

    # Mask the image data to retain only the connected component containing the center of mass
    masked_im_data = im_data * connected_component
    masked_im_data[masked_im_data > 0] = 1
    masked_im_data[masked_im_data < 1] = 0
    masked_im_data = masked_im_data.astype(np.uint8)
    # Calculate the number of voxels in the connected component and derive an equivalent spherical radius
    n_vox = np.sum(masked_im_data > 0)
    rad = round(
        ((n_vox * (0.5**3)) * (3 / (4 * np.pi))) ** (1 / 3), 2
    )  # correct radius
    if n_vox == 0:
        print(f"No voxels for subject {subject} in the connected component {com}")
        return

    extra = "_pred" if pred else ""
    filepath = f"{args.savedir}/{subject}_{com}_{rad}{extra}.png"

    if os.path.exists(filepath) and not args.overwrite:
        print(
            f"WARNING: File already exists: {filepath}. Skipping, or add --overwrite to overwrite"
        )
        return
    # Crop the image data around the center of mass based on the specified crop size
    cropped_im_data = crop_data_around_center(masked_im_data, com, args.crop_size)

    try:
        # Plot using the specified engine
        if args.engine == "mayavi":
            utils_plotting.plot_microbleed_mayavi(
                cropped_im_data,
                filepath=filepath,
            )
        elif args.engine == "vtk":
            utils_plotting.plot_microbleed_vtk(
                cropped_im_data,
                filepath=filepath,
            )
        else:
            raise ValueError("Invalid engine: options are 'mayavi' or 'vtk'")
    except Exception as e:
        print(f"Error plotting image for subject {subject}: {e}")
        return


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
        "--datadir", type=str, default=None, help="Path to the input directory"
    )
    parser.add_argument(
        "--savedir", type=str, default=None, help="Path to the output directory"
    )
    parser.add_argument(
        "--engine", type=str, default="vtk", help="Plotting engine (mayavi or vtk)"
    )
    parser.add_argument("--crop_size", type=int, default=40, help="Crop size")
    parser.add_argument(
        "--prediction_dir",
        type=str,
        default=None,
        help="Path to the predictions directory. If provided these are also plotted if found",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_images(args, SEED)
