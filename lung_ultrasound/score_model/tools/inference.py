"""
inference of vital lung ultrasound classification model. 
The model is loaded in the initialize function and the inference is done in the inference function. 
The model is loaded only once and the inference function can be called multiple times with different input data.
The input data is a list of frames (numpy arrays) and the output is a list of probabilities for each class. 
The classes are: A-lines, B-lines, Confluent B-line, Consolidation, Pleural Effussion.
"""
import json
import os
import torch
from torch.serialization import add_safe_globals
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import lung_ultrasound.models.vital_models as classifiers
from lung_ultrasound.dataset.dataset_vital import DatasetVital
from torch.utils.data import DataLoader
import lung_ultrasound.utils as utils
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

classes = ["A-lines", "B-lines", "Confluent B-line", "Consolidation", "Pleural Effussion"]

verbose = False
net = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = []


def get_classes():
    return classes


def initialize(input_size, model_path, verb: bool = False):
    """
    Load the model and initialize the network
    model_path: path where the config file and the model weights are.

    """
    global device
    global net
    global verbose
    global params

    config_filename = '{}/VITAL-lusclassification_training_config.json'.format(model_path)
    if os.path.exists(config_filename):
        with open(config_filename, "r") as lf:
            params = json.load(lf)
    else:
        print('Cannot find config file {}'.format(config_filename), flush=True)
        exit(-1)

    verbose = verb
    
    # load model for video classification
    print('[LUSclassificationp_worker.py: initialize] load model {}...'.format(model_path))

    n_output_classes = 5

    if params['model'] == 'tmpAttLSTM':
        net = classifiers.TempAttVideoClassifierLSTM(input_size=input_size, n_output_classes=n_output_classes, cnn_channels=params['cnn_channels'], dropout_p=params['dropout'], n_c_layers=params['n_classification_layers'])
    elif params['model'] == 'stAttLSTM':
        net = classifiers.SpatioTempAttVideoClassifierLSTM(input_size=input_size, n_output_classes=n_output_classes, cnn_channels=params['cnn_channels'], spatial_attention_layers_pos=params['spatial_att_layers'], dropout_p=params['dropout'], n_c_layers=params['n_classification_layers'])
    elif params['model'] == 'LSTM':
        net = classifiers.VideoClassifierLSTM(input_size=input_size, n_output_classes=n_output_classes, cnn_channels=params['cnn_channels'], dropout_p=params['dropout'], n_c_layers=params['n_classification_layers'])
    elif params['model'] == 'tConv':
        net = classifiers.VideoClassifierConv(input_size=input_size, n_output_classes=n_output_classes, cnn_channels=params['cnn_channels'], dropout_p=params['dropout'], n_c_layers=params['n_classification_layers'])

    net.to(device)
    net.eval()

    print(params['model'])
    

    checkpoint_f = '{}/best_validation_acc_model.pth'.format(model_path)

    if verbose:
        print('[LUS_classificationp_worker.py::initialize() - Load model {}'.format(checkpoint_f))
    state = torch.load(checkpoint_f, weights_only = False) 
    net.load_state_dict(state['model_state_dict'])

    if verbose:
        print(net)
    return net


def dowork(frames: np.array, verbose=0):
    with torch.no_grad():
        # frames = torch.from_numpy(frames).type(torch.float).to(device).unsqueeze(0).unsqueeze(0)/255.0
        
        try:
            out = net(frames)
            if 'stAtt' in params['model']:
                sAtt = out[2]
                sAtt = [((sAtt_i - torch.min(sAtt_i)) / (torch.max(sAtt_i) - torch.min(sAtt_i)) * 255.0).type(torch.uint8).cpu().numpy() for sAtt_i in sAtt]
            if 'Att' in params['model']:
                att = out[1]
                att = (att - torch.min(att)) / (torch.max(att) - torch.min(att))
            out = out[0]
            out_index = torch.argmax(out, dim=1)

        except Exception as ex:
            print('[Python exception caught] LUSp::process() - {}{}'.format(ex, ex.__traceback__.tb_lineno))
            exit(-1)

    out = torch.nn.functional.softmax(out, dim=1).cpu().numpy()
    # print('results')
    # print(out)
    # print(out_index)

    # average of all attentions
    attention = np.mean(np.stack(sAtt, axis=-1), axis=-1)
    # at at level att_idx
    #att_idx = 0
    #attention = sAtt[att_idx]
    #attention = frames[0,...].cpu().numpy()*255.0
    #print(attention.shape)
    #print(sAtt[att_idx].shape)

    #attention = attention[0, -1, ...] # take just the last frame
    # instead of taking the last frame, take a weighted average where the last takes the most weight
    var = 5
    weights = np.exp(np.expand_dims(np.array(np.arange(0, attention.shape[1]))/var,(0,2,3)))
    weights /= np.sum(weights)
    attention_weighted = np.sum(attention * weights, axis=1).squeeze() # take just the last frame

    #attention = np.ascontiguousarray(attention.transpose()).astype(np.uint8)
    attention_weighted = np.ascontiguousarray(attention_weighted).astype(np.uint8)
    #im = Image.fromarray(attention)
    #im.save("/home/ag09/data/VITAL/np_image.png")
    #np.save('/home/ag09/data/attention.npy', attention)
    #exit(-1)
    #print(attention.shape)
    return (out, attention_weighted.astype(np.uint8), attention)

if __name__ == '__main__':
    dataset_path = '/media/angelo/PortableSSD/Assistant_Researcher/Predict/LUS_data_covid19/DATA_covid_compvital'
    model_path = '/home/angelo/Documenti/Lung_Ultrasound/lung_ultrasound/cfg'
    splitting_json = 'splitting_clean.json'
    size = (64,64)      # size of each frame
    im_channels = 1     # number of channels of the image (1 for grayscale, 3 for RGB)
    fold_cv = 'fold_1'  # numer of the 5-fold cross validation fold to use (from fold_1 to fold_5)   
    split = 'train'
    lenght = 1         # lenght of the segments of frames to return (in seconds)
    overlap = 0.2      # overlap between segments (percentage between 0 and 1)
    sampling_f = 30      # sampling frequency of the frames (default: 30)
    fps = 30

    split_list = ['train', 'val', 'test']
    dataset_list = []
    for ii in split_list:
        dataset_i = DatasetVital(dataset_path = dataset_path,
                            data_augmentation = False,
                            size = size,
                            im_channels = im_channels,
                            lenght = lenght,
                            overlap = overlap,
                            fps = fps,
                            sampling_f = sampling_f,
                            splitting_json = splitting_json,
                            fold_cv = fold_cv,
                            split = ii, 
                            trasformations = True)
        dataset_list.append(dataset_i)

    ## concatenate the datasets
    dataset = torch.utils.data.ConcatDataset(dataset_list)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    initialize(size, model_path, verb=False)

    all_preds = []
    all_labels = []
    for frames, label, subject in dataloader:
        frames = frames.to(device)
        out, weighte_att, att = dowork(frames = frames)

        ##
        prediction_class = classes[np.argmax(out)]
        label_class = classes[np.argmax(label)]

        pred = np.argmax(out, axis=1)
        gt = np.argmax(label.cpu().numpy(), axis=1) 
        all_preds.append(pred[0])
        all_labels.append(gt[0])
        print(f"Subject: {subject[0]}, label: {label_class} - pred: {prediction_class}")


    utils.confusion_matrix(all_labels, all_preds, classes)
    




    