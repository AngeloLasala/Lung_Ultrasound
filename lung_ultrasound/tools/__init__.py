"""
Collection of configuration fro train test and val
"""

class cfg_train:
    seed = 42

    ## folder configuration
    main_path = "/media/angelo/PortableSSD/Assistant_Researcher/Predict/LUS_data_covid19"
    dataset = "DATA_covid"
    results= "results"

    ## model configuration
    model_name = "ResNetLUSCAM"
    backbone = "resnet50"
    pretrained = True
    freeze_backbone = False
    pooling = "avg"
    num_classes = 6

    ## train configuration
    device = 'cuda'
    epochs = 100
    batch_size = 16
    learning_rate = 1e-4
    size = (256,256)
    im_channels = 3
    fold_cv = 'fold_1'        # cross-validation fold 
    cosine_annealing = True   # use cosine annealing learning rate scheduler
    eval_freq = 1             # evaluate every n epochs
