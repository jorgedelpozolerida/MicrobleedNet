import os
import sys
import argparse
import traceback


import logging
import numpy as np
import pandas as pd
from clearml import Task


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

import os
import sys
import argparse
import traceback


import logging
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


csv_basedir = "../../data-misc/predict"

csvpaths = [d for d in os.listdir(csv_basedir) if d.endswith(".csv")]
savedir = "/home/cerebriu/data/datasets/predictions_last/"

if not os.path.exists(savedir):
    os.makedirs(savedir)
    print(f"Created {savedir}")

def download_task_artifacts(task, save_dir):
    """
    Download specified artifacts from a ClearML task to a given directory.

    Args:
        task (Task): The ClearML Task object from which to download artifacts.
        save_dir (str): The directory path where artifacts should be saved.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    for artifact_name, artifact in task.artifacts.items():
        if artifact_name.startswith("prediction-"):
            artifact_path = artifact.get_local_copy()  # Specify target folder
            _logger.info(f"Downloaded {artifact_name}")

def download_task_predictions(row):
    task_name = row['NAME']
    # NAME	TAGS: tags	TAGS: system_tags	STATUS	USER	STARTED	UPDATED	ITERATION	PARENT TASK: parent.name	PARENT TASK: parent.project.id	PARENT TASK: parent.project.name
    tags = row['TAGS: tags']
    tags = tags.split(",")
    folder_clearml = [t.split("_")[0] for t in tags if "TL-" in t or "Scratch-" in t][0]
    project_name = f"CMB/predictions/{folder_clearml}"

    metric = [t.split("_")[1] for t in tags if "PPV" in t or "valloss" in t or "F1macro" in t][0]
    subdir = os.path.join(savedir_task, task_name, metric)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
        print(f"Created {subdir}")

    # Query the task by name and tags
    tasks = Task.get_tasks(task_name=task_name, tags=tags, allow_archived=False, project_name=project_name)
    # for task in tasks:
    #     # print(f"\tTask: {task.name} ID: {task.id}, type: {task.task_type}, status: {task.status}, parent: {task.parent}")
    #     ...
    if not tasks:
        _logger.info(f"No completed tasks found for {task_name} with tags {tags}")
        return
    else:
        _logger.info(f"Found {len(tasks)} tasks for {task_name} with tags {tags}")
    # elif len(tasks) > 1:
    #     _logger.info(f"More than one task found for {task_name} with tags {tags}")

    #     return
    
    for task in tasks:
        # Retrieve and log detailed task information
        task_info = {
            'Task Name': task.name,
            'Task ID': task.id,
            'Project': task.project,
            'Task Type': task.task_type,
            'Status': task.status,
        }
        # Log detailed info
        # _logger.info(f"Task Details: {task_info}")
        download_task_artifacts(task, subdir)

for csvpath in  csvpaths:
    _logger.info(f"Processing {csvpath}")
    savedir_task = os.path.join(savedir, csvpath.split("_")[1])
    if not os.path.exists(savedir_task):
        os.makedirs(savedir_task)
        print(f"Created {savedir_task}")
    df = pd.read_csv(os.path.join(csv_basedir, csvpath))
    for i, row in df.iterrows():
        try:
            download_task_predictions(row)
        except Exception as e:
            _logger.error(traceback.format_exc())
            continue