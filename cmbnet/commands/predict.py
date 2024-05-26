#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script for running all needed predictions


NOTE: sent to clearml


@author: jorgedelpozolerida
@date: 18/05/2024
"""
import os
import argparse


import logging

import os
import subprocess
import yaml

from cmbnet.utils.utils_general import confirm_action

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main(args):
    config = load_config(args.config_yaml_file)

    datasets = config['datasets']
    models = config['models']
    cmd_params = config['command_parameters']

    base_dir_configs = cmd_params['configs_dir']

    for dataset in datasets:
        configs_dir_dataset = os.path.join(base_dir_configs, dataset['name'])
        if not os.path.exists(configs_dir_dataset):
            print(f"Configs directory for dataset {dataset['name']} does not exist")
            continue
        os.chdir(configs_dir_dataset)  # Change to the base directory

        for model in models:
            tags = f"{dataset['name']} {model['name']}"
            split_str = f"--splits '{dataset['split']}'" if dataset['split'] is not None else ''
        
            command = (
                f"crbr --tags {tags} --task-name predict_{dataset['name']} "
                f"--project-name {cmd_params['clearml_basedir']}/{model['subfolder']} -q {cmd_params['worker']} "
                f"--clearml remote predict --model-id '{model['id']}' --dataset-id '{dataset['id']}' {split_str}"
            )

            _logger.info(f"Running command for dataset {dataset['name']} and model {model['name']}")
            print("Dir ------------------------")
            print(os.getcwd())
            print("Command ------------------------")
            print(command)
            # confirm_action()
            subprocess.run(command, shell=True)
        os.chdir(base_dir_configs)


def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()


    parser.add_argument('--config_yaml_file', type=str, default=None, required=True,
                        help='Path to config file with all needed parameters')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)