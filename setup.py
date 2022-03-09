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
    install_requires=['cycler==0.11.0',
                      'fonttools==4.28.3',
                      'higher==0.2.1',
                      'kiwisolver==1.3.2',
                      'matplotlib==3.5.0',
                      'plotly>=5.4.0',
                      'metalearning-benchmarks @ git+https://github.com/michaelvolpp/metalearning_benchmarks.git@3005e0125ad8c57d112f693241cb2b052c3b249c'
                      'mtbnn @ git+https://github.com/michaelvolpp/mtbnn.git@00cabf80bb0173233774501c083ec08ea30b95f6',
                      'metalearning_models @ git+https://github.com/michaelvolpp/metalearning_models.git@clv_adapt_api_change'
                      'numpy==1.21.4',
                      'packaging==21.3',
                      'Pillow==8.4.0',
                      'pyparsing==3.0.6',
                      'python-dateutil==2.8.2',
                      'setuptools-scm==6.3.2',
                      'six==1.16.0',
                      'tomli==1.2.2',
                      'torch==1.10.0',
                      'torchvision==0.11.1',
                      'typing_extensions==4.0.1',
                      'kaleido>=0.2.1'
                      ],
)
