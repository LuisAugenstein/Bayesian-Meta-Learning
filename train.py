import torch
import numpy as np
import os
import argparse
import pathlib
import uuid

from bayesian_meta_learning.runner.MainRunner import MainRunner
from bayesian_meta_learning.runner.CLVRunner import CLVRunner
from bayesian_meta_learning.parameter_description import parameter_description
from few_shot_meta_learning._utils import train_val_split_regression

from learner import Learner

# --------------------------------------------------
# SETUP INPUT PARSER
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Setup variables')

    for param in parameter_description:
        parser.add_argument(f"--{param['name']}", default=param['default'], type=param['type'], help=param['help'])
    args = parser.parse_args()

    # define config object from parser args
    config = {}
    for key in args.__dict__:
        config[key] = args.__dict__[key]

    # start the learner with the given config
    Learner.run(config)

if __name__ == "__main__":
    main()
