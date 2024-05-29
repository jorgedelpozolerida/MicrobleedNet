import os
import subprocess
from multiprocessing import Pool
import cmbnet.utils.utils_general as utils_gen

# Define the base directories for the three levels
# level1_dirs = ["Scratch-Pretrained-FineTuned", "TL-Pretrained-FineTuned"]
level1_dirs = ["Scratch-Pretrained-FineTuned", "TL-Pretrained-FineTuned"]
level2_dirs = ["predict_cmb_crb", "predict_cmb_valid", "predict_cmb_dou"]
# level3_dirs = ["PPV", "F1macro", "valloss"]
# level3_dirs = ["valloss", "F1macro"]
level3_dirs = ["valloss"]


# Dictionary specifying the DataFrame and the column to split the analysis by
# splits = {
#     # "all_studies_df": "res_level",
#     "all_studies_df": "seq_type",
# }
splits = {
}


# Base directory paths
savedir_base = "/storage/evo1/jorge/MicrobleedNet/data-misc/evaluations"
cmb_pred_metadata_dir_base = "/storage/evo1/jorge/datasets/cmb/evaluations_cmb"

# CSV paths
metadata_csv_path = "/storage/evo1/jorge/MicrobleedNet/data-misc/csv/CMB_metadata_all.csv"
gt_radiomics_metadata_csv = "/storage/evo1/jorge/MicrobleedNet/data-misc/csv/CMB_radiomics_metadata.csv"
gt_cmb_metadata_csv = "/storage/evo1/jorge/MicrobleedNet/data-misc/csv/CMB_metadata_all.csv"
all_studies_csv = "/storage/evo1/jorge/MicrobleedNet/data-misc/csv/ALL_studies.csv"

def run_evaluation(command):
    """
    Function to run a subprocess with the given command.
    """
    print("Running command:", " ".join(command))
    subprocess.run(command)
    print(f"Completed command")

def run_evals():
    for l1 in level1_dirs:
        for l2 in level2_dirs:
            for l3 in level3_dirs:
                print(l1, l2, l3)
                predictions_dir = os.path.join(cmb_pred_metadata_dir_base, l1, l2, l3)
                savedir = os.path.join(savedir_base, l1, l2, l3)
                if l2 == "predict_cmb_valid":
                    dataset = "cmb_valid"
                elif l2 == "predict_cmb_dou":
                    dataset = "DOU"
                elif l2 == "predict_cmb_crb":
                    dataset = "CRB"
                # Ensure the savedir exists
                # os.makedirs(savedir, exist_ok=True)

                # Construct the command to run
                print(".......................................................")
                print("Running command with the following arguments:")
                print("\t",l1,"-", l2,"-", l3)
                print(".......................................................")

                # utils_gen.confirm_action()

                cmd = [
                    "python", "evaluate.py",
                    "--output_dir", savedir,
                    "--cmb_pred_metadata_dir", predictions_dir,
                    "--gt_radiomics_metadata_csv", gt_radiomics_metadata_csv,
                    "--gt_cmb_metadata_csv", gt_cmb_metadata_csv,
                    "--all_studies_csv", all_studies_csv,
                    "--dataset", dataset,
                ]
                run_evaluation(cmd)
             
if __name__ == "__main__":

    run_evals() 

    print("All evaluations completed.")
