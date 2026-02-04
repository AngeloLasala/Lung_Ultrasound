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