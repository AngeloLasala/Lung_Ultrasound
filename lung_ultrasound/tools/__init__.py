"""
Collection of configuration fro train test and val
"""
import lung_ultrasound.models.vital_models as classifiers
import os
import json
import torch


class cfg_train:
    """
    Configuration for training the LUS clip classification model
    """
    seed = 42

    ## folder configuration
    main_path = "/media/angelo/PortableSSD/Assistant_Researcher/Predict/OpenPOCUS"
    dataset = "DATA_Lung_Database"
    results= "results"

    ## dataset configuration
    size = (64,64)
    im_channels = 1
    fold_cv = 'fold_1'        # cross-validation fold 
    splitting = 'splitting.json'
    lenght = 1               # lenght of the segments of frames to return (in seconds)
    overlap = 0.2            # overlap between segments (percentage between 0 and 1)
    sampling_f = 30          # sampling frequency of the frames (default: 30)
    fps = 30

    ## model configuration
    model_path = '/home/angelo/Documenti/Lung_Ultrasound/lung_ultrasound/cfg' ## path to the Pisani models' weights
    model_name = "PisaniModel"
    pretrained_weights = False
    num_classes = 5
    mode = 'train'

    ## augumentation configuration
    h_flip_p = 0.5
    v_flip_p = 0.0
    rotation_deg = 23.0
    crop_scale = (1.0, 1.0)
    crop_ratio = (1.0, 1.0)
    brightness_p = 0.5
    contrast_p = 0.5
    brightness = 0.10
    contrast = 0.10

    ## train configuration
    device = 'cuda'
    epochs = 100
    learning_rate = 0.001
    batch_size = 8
    cosine_annealing = False       # use cosine annealing learning rate scheduler
    eval_freq = 1                  # evaluate every n epochs
    verbose = True
    ema_decay = 0.999              # value for ema decay for model saving
    early_stopping_patience = 60   # epoch for patience
    min_delta = 1e-4               # for stability in early stopping
    monitor_metric = 'weighted_f1'    



def load_model(config):
    """
    Load the model and initialize the network
    model_path: path where the config file and the model weights are.
    """

    # Convert dict config to object-like access if needed
    if isinstance(config, dict):
        class ConfigWrapper:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)
        config = ConfigWrapper(config)

        
    global device
    global net
    global verbose
    global params

    device = config.device
    verbose = config.verbose
    params = []
    net = None

    config_filename = '{}/VITAL-lusclassification_training_config.json'.format(config.model_path)
    if os.path.exists(config_filename):
        with open(config_filename, "r") as lf:
            params = json.load(lf)
    else:
        print('Cannot find config file {}'.format(config_filename), flush=True)
        exit(-1)
    
    # load model for video classification
    print('[LUSclassificationp_worker.py: initialize] load model {}...'.format(config.model_path))

    n_output_classes = 5   ## IMPORTANT this is the class of Pisani model not for the classification task, may not be the same!!
    input_size = config.size

    if params['model'] == 'tmpAttLSTM':
        net = classifiers.TempAttVideoClassifierLSTM(input_size=input_size, n_output_classes=n_output_classes, cnn_channels=params['cnn_channels'], dropout_p=params['dropout'], n_c_layers=params['n_classification_layers'])
    elif params['model'] == 'stAttLSTM':
        net = classifiers.SpatioTempAttVideoClassifierLSTM(input_size=input_size, n_output_classes=n_output_classes, cnn_channels=params['cnn_channels'], spatial_attention_layers_pos=params['spatial_att_layers'], dropout_p=params['dropout'], n_c_layers=params['n_classification_layers'])
    elif params['model'] == 'LSTM':
        net = classifiers.VideoClassifierLSTM(input_size=input_size, n_output_classes=n_output_classes, cnn_channels=params['cnn_channels'], dropout_p=params['dropout'], n_c_layers=params['n_classification_layers'])
    elif params['model'] == 'tConv':
        net = classifiers.VideoClassifierConv(input_size=input_size, n_output_classes=n_output_classes, cnn_channels=params['cnn_channels'], dropout_p=params['dropout'], n_c_layers=params['n_classification_layers'])

    net.to(device)
    if config.mode == 'train':
        net.train()
    else:
        net.eval()

    checkpoint_f = '{}/best_validation_acc_model.pth'.format(config.model_path)
    
    if config.pretrained_weights:
        if verbose: print('Loading model weights from {}'.format(checkpoint_f)) 
        state = torch.load(checkpoint_f, weights_only = False) 
        net.load_state_dict(state['model_state_dict'])

    else:
        if verbose: print('Load model with random initialization')

    if verbose:
        num_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f'Model parameters: {num_params/1e6:.2f} M - Trainable parameters: {trainable_params/1e6:.2f} M')
    return net
