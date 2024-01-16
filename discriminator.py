from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from os.path import isfile
from tqdm import tqdm
import json
import argparse

class EfficientNetB0(nn.Module):
    def __init__(self):
        super(EfficientNetB0, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.head = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024 , 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256 , 40)
            )

    def forward(self,x):
        x = self.model.features(x)
        #-----#
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.head(x)
        return x

_initialized = False

def initialize():
    global _initialized, _device, cpu_net, gpu_net
    _device = None
    if torch.cuda.is_available():
        _device = torch.device("cuda")
        gpu_net = EfficientNetB0()
        gpu_net.load_state_dict(torch.hub.load_state_dict_from_url("https://github.com/GurvanR/GANSpace-Reimplementation/raw/main/atclas2.pt", file_name="atclas2_gwilherm_lesne.pt"))
        gpu_net = gpu_net.to(_device)
        gpu_net.eval()
    cpu_net = EfficientNetB0()
    cpu_net.load_state_dict(torch.hub.load_state_dict_from_url("https://github.com/GurvanR/GANSpace-Reimplementation/raw/main/atclas2.pt", file_name="atclas2_gwilherm_lesne.pt", map_location=torch.device('cpu')))
    cpu_net.eval()
    _initialized = True

def classify(batch, device="cuda"):
    if not _initialized:
        initialize()
    with torch.inference_mode():
        if (device == "cuda" or device == _device) and torch.cuda.is_available():
            return gpu_net(batch.to(_device))
        else:
            return cpu_net(batch.to("cpu"))

def run_classification(dataset_path, output_path, proportion, batch_size=10):
    global _initialized
    if not _initialized:
        initialize()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.isdir(output_path):
        raise ValueError("Output path must be a directory, not overwriting existing file")
    filenames = [f for f in os.listdir(dataset_path) if isfile(os.path.join(dataset_path, f))]
    filenames_and_scores = [[] for _ in range(40)]
    totensor = transforms.ToTensor()

    for i in tqdm(range(0, len(filenames), batch_size)):
        # Load the batch
        batch = torch.zeros((batch_size, 3, 1024, 1024))
        batch_filenames = filenames[i:min(i+batch_size, len(filenames))]
        for j, f in enumerate(batch_filenames):
            path = os.path.join(dataset_path, f)
            if not isfile(path):
                continue
            img = Image.open(path)
            batch[j] = 255*totensor(img)
            img.close()
        batch = batch.to(_device)

        # Feed the batch to the network and store the results
        with torch.inference_mode():
            out = net(batch).cpu()
        for j, f in enumerate(batch_filenames):
            for att in range(40):
                filenames_and_scores[att].append((f, out[j, att].item()))

    # Store the results in JSON files
    for att in range(40):
        with open(os.path.join(output_path, f"att{att}_scores.json"), "w") as outfile:
            json.dump(dict(filenames_and_scores[att]), outfile)

        # Create two classes with the samples classified with the highest confidence
        filenames_and_scores[att].sort(key=lambda p: p[1])
        num_top = int(proportion*len(filenames)/100)
        filenames_minus1 = [(p[0], -1) for p in filenames_and_scores[att][:num_top]]
        filenames_plus1 = [(p[0], 1) for p in filenames_and_scores[att][-num_top:]]
        with open(os.path.join(output_path, f"att{att}_labels.json"), "w") as outfile:
            json.dump(dict(filenames_minus1 + filenames_plus1), outfile)