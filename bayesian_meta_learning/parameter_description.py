parameter_description = [{
        'name':  'load_dir_bmaml_chaser',
        'default': 'bmaml_chaser_models/model_100.pickle',
        'type': str,
        'help': 'path to a .pickle file with parameters from bmaml with chaser loss'
    }, {
        'name':  'noise_stddev',
        'default': 0.1,
        'type': float,
        'help': 'standard deviation of the white gaussian noise added to the data targets y'
    }, {
        'name':  'seed',
        'default': 123 ,
        'type': int,
        'help': 'general seed for everything but data generation'
    },  {
        'name':  'seed_offset',
        'default': 1234,
        'type': int,
        'help': 'data generation seed for the meta training tasks'
    },  {
        'name':  'seed_offset_test',
        'default': 12345,
        'type': int,
        'help': 'data generation seed for the meta testing task'
    },  {
        'name':  'num_points_per_train_task',
        'default': 50,
        'type': int,
        'help': 'number of datapoints in each meta training task'
    }, {
        'name':  'num_test_tasks',
        'default': 100,
        'type': int,
        'help': 'number of meta testing tasks'
    }, {
        'name':  'num_points_per_test_task',
        'default': 512,
        'type': int,
        'help': 'number of datapoints in each meta testing task'
    }, {
        'name':  'num_hidden',
        'default': 2,
        'type': int,
        'help': 'number of hidden layers if using a fully connceted network'
    }, {
        'name':  'hidden_size',
        'default': 40,
        'type': int,
        'help': 'hidden layer size if using a fully connected network'
    }, {
        'name':  'advance_leader',
        'default': 10,
        'type': int,
        'help': 'number of training steps the leader is ahead of the chaser(s). Only relevant for bmaml-chaser'
    }, {
        'name':  'num_inner_updates_testing',
        'default': 5,
        'type': int,
        'help': 'number of inner update steps used during testing'
    }, {
        'name':  'nlml_testing_enabled',
        'default': True,
        'type': bool,
        'help': 'whether to calculate neg log marginal likelihood or not.'
    }, {
        'name':  'reuse_models',
        'default': False,
        'type': bool,
        'help': 'Specifies if a saved state should be used if found or if the model should be trained from start.'
    }, {
        'name':  'normalize_benchmark',
        'default': False,
        'type': bool,
        'help': 'whether the task should be normalized to zero mean and unit variance'
    }, {
        'name':  'wandb',
        'default': False,
        'type': bool,
        'help': 'Specifies if logs should be written to WandB'
    }, {
        'name':  'num_visualization_tasks',
        'default': 4,
        'type': int,
        'help': 'number of randomly chosen meta testing tasks that are used for visualization'
    }, {
        'name':  'y_plotting_resolution',
        'default': 512,
        'type': int,
        'help': 'number of discrete y-axis points to evaluate for visualization'
    }, {
        'name':  'epochs_to_save',
        'default': 1000,
        'type': int,
        'help': 'number of epochs between saving the model'
    }, {
        'name':  'benchmark',
        'default': 'Sinusoid1D',
        'type': str,
        'help': 'possible values are Sinusoid1D, Affine1D, Quadratic1D, SinusoidAffine1D'
    }, {
        'name':  'num_ways',
        'default': 1,
        'type': int,
        'help': 'd_y dimension of targets'
    }, {
        'name':  'k_shot',
        'default': 5,
        'type': int,
        'help': 'number of datapoints in the context set (needs to be less than points_per_train_task)'
    }, {
        'name':  'algorithm',
        'default': 'maml',
        'type': str,
        'help': 'possible values are maml, platipus, bmaml, baseline and bmaml_chaser and clv'
    }, {
        'name':  'network_architecture',
        'default': 'FcNet',
        'type': str,
        'help': 'architecture of the neural network'
    }, {
        'name':  'num_epochs',
        'default': 10,
        'type': int,
        'help': 'number of training epochs. one epoch means working through all trainig datapoints one time'
    }, {
        'name':  'num_episodes_per_epoch',
        'default': 100,
        'type': int,
        'help': 'Number of meta train tasks. should be a multiple of minibatch'
    }, {
        'name':  'num_models',
        'default': 10,
        'type': int,
        'help': 'number of models (phi) we sample from the posterior in the end for evaluation. irrelevant for maml'
    }, {
        'name':  'minibatch',
        'default': 25,
        'type': int,
        'help': 'Minibatch of episodes (tasks) to update meta-parameters'
    }, {
        'name':  'minibatch_print',
        'default': 1,
        'type': int,
        'help': 'number of minibatches between each validation plotting to wandb'
    }, {
        'name':  'num_inner_updates',
        'default': 1,
        'type': int,
        'help': 'number of SGD steps during adaptation'
    }, {
        'name':  'inner_lr',
        'default': 0.001,
        'type': float,
        'help': 'learning rate during adaptation '
    }, {
        'name':  'meta_lr',
        'default': 0.001,
        'type': float,
        'help': 'learning rate during meta-update'
    }, {
        'name':  'KL_weight',
        'default': 0.01,
        'type': float,
        'help': 'weighting of the KL term in platipus'
    }, {
        'name':  'num_episodes',
        'default': 4,
        'type': int,
        'help': 'Number of validation tasks used for the MLBaseClass.evaluate() method'
    }, {
        'name':  'resume_epoch',
        'default': 0,
        'type': int,
        'help': '0 means fresh training. >0 means training continues from a corresponding stored model.'
    }, {
        'name':  'logdir_base',
        'default': '.',
        'type': str,
        'help': 'Folder to store models and logs'
    }, {
        'name':  'first_order',
        'default': True,
        'type': bool,
        'help': 'Should always be true for MAML basd algos'
    },  {
        'name':  'train_flag',
        'default': True,
        'type': bool,
        'help': 'whether to train or just test the algorithms'
    }
    ]