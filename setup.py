from setuptools import setup
setup(
    name='bayesian_meta_learning',
    version='0.1.0',    
    description='A package to quickly run the different Meta-Learning algorithms',
    url='https://github.com/LuisAugenstein/Bayesian-Meta-Learning',
    author='Leon Jungemeyer and Luis Augenstein',
    author_email='leon.jungemeyer@student.kit.edu, luis.augenstein@web.de',
    license='MIT',
    packages=['bayesian_meta_learning'],
    install_requires=['torch',
                      'numpy',   
                      'metalearning_models',
                      'metalearning_benchmarks'                  
                      ],
)