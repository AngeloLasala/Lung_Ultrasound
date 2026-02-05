# Lung_Ultrasound
Design DL model for analyzing  image and video lung US data


## Install
This repository relys on the [PyTorch](https://pytorch.org/get-started/locally/) . Before install the requirements please read the official documentation and install torch. here a simple example:

create a virtual env with conda
```bash
conda create --name lus python==3.10
conda deactivate
conda activate lus
```

install torch
```bash
pip install torch torchvision torchaudio
```

downlosd the repository using ssh connection
```bash
git clone git@github.com:AngeloLasala/Lung_Ultrasound.git
```

install the requirements
```bash
pip install -e .
```

## Training

for the trianing, use `train.py` in folder `tools`  

```bash
python train.py --keep_log
```

the flag `keep_log` enable to save checkpoint anf tenorboard loggings. For visulizing tensorbors use the following command line

```bash
tensorboard --logdir name_of_log_dir
```

See class `cfg_train` for mode details about the training configuration setting
