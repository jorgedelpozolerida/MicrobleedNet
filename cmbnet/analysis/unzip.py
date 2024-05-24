import os
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def unzip_and_remove_zip(directory):
    """
    Walks through a directory, unzips any found zip files, and removes the zip files.

    Args:
        directory (str): The directory path to search for zip files.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Extract all the contents of zip file in the directory
                    zip_ref.extractall(root)
                # Remove the zip file after extracting
                os.remove(zip_path)
                logger.info(f"Extracted and removed zip file: {zip_path}")

# Example usage
directory_to_search = "/storage/evo1/jorge/datasets/cmb/predictions_last"  # Replace with your directory path
unzip_and_remove_zip(directory_to_search)
