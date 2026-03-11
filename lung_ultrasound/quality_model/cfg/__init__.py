class cfg:
    """
    Configuration for training the LUS clip classification model
    """
    seed = 42

    ## folder configuration
    main_path = "D:\Assistant_Researcher\Predict\OpenPOCUS"
    dataset = "Extrapolates_frames"
    results = "results"
    save_folder = "checkpoints"
    tensorboard_folder = "tensorboard"

    ## dataset configuratios
    size = 256
    fold_cv = 'fold_1'        # cross-validation fold 
    splitting = 'splitting_ext_plax.json'

    ## augumentation configuration (see dataset JointTrasformation for default value)
    img_size = size
    low_img_size = size
    ori_size = size
    crop = (32, 32)
    p_flip = 0.5
    p_rota = 0.5
    p_scale = 0.5
    p_gaussn = 0.0     # it doesn't work
    p_contr = 0.5
    p_gama = 0.5
    p_distor = 0.0
    color_jitter_params = None
    p_random_affine = 0
    long_mask = True

    ## Training parameters    ##########################################################
    model_name = 'UNet'
    workers = 1                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 8                      # batch size (default: 4)
    learning_rate = 5e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momentum
    num_classes = 3                     # then umber of classes (background + foreground)
    img_size = 256                      # the input size of model
    im_channels = 1
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    eval_mode = "mask_slice"            # the mode when evaluate the model, slice level or patient level
    class_weights = [1., 1., 1.]        # class weight for unbalances segmentation mask - [1.]*num_classes for deactivating class weights
    w_ce = 0.6                          # weight of CE loss function
    w_dice = 0.4                        # weight of Dice loss function (Note: w_ce+w_dice suppose to be 1)
    pre_trained = False
    mode = "train"
