import argparse
from bayesian_meta_learning.parameter_description import parameter_description
from bayesian_meta_learning.learner import Learner

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
