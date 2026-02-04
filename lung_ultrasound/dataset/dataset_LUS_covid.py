"""
Create the DatasetLUSCovid compatible with the Pythorch framework
and the main visualiztion dataset statistics
"""
import os
import json
import torchvision
from PIL import Image
from tqdm import tqdm
import json
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import math
import matplotlib.pyplot as plt


if __name__ == "__main__":
    a=0