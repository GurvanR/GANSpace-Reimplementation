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
    
def main():
    parser = argparse.ArgumentParser(
        prog="classify_images",
        description="Classify images in a dataset and create two sets of images per attribute in order to apply InterFaceGAN on them.")
    parser.add_argument("dataset_path")
    parser.add_argument("output_path")
    parser.add_argument("-p", "--proportion", help="Proportion of images to put to each group of each attribute", default=0.02)
    args = parser.parse_args()
    """if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} <path to folder containing images> <path to output folder>")
        return 1"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = EfficientNetB0()
    net.load_state_dict(torch.load('atclas2.pt'))
    net = net.to(device)
    net.eval()
    
    filenames = [f for f in os.listdir(args.dataset_path) if isfile(os.path.join(args.dataset_path, f))]
    batch_size = 1
    filenames_and_scores = [[] for _ in range(40)]
    totensor = transforms.ToTensor()
    for i in tqdm(range(0, len(filenames), batch_size)):
        batch = torch.zeros((batch_size, 3, 1024, 1024))
        batch_filenames = filenames[i:min(i+batch_size, len(filenames))]
        for j, f in enumerate(batch_filenames):
            path = os.path.join(args.dataset_path, f)
            if not isfile(path):
                continue
            img = Image.open(path)
            batch[j] = 255*totensor(img)
            img.close()
        batch = batch.to(device)
        with torch.inference_mode():
            out = net(batch).cpu()
        for j, f in enumerate(batch_filenames):
            for att in range(40):
                filenames_and_scores[att].append((f, out[j, att].item()))
    for att in range(40):
        with open(os.path.join(args.output_path, f"att{att}_scores.json"), "w") as outfile:
            json.dump(dict(filenames_and_scores[att]), outfile)
        filenames_and_scores[att].sort(key=lambda p: p[1])
        num_top = int(args.proportion*len(filenames))
        filenames_minus1 = [(p[0], -1) for p in filenames_and_scores[att][:num_top]]
        filenames_plus1 = [(p[0], 1) for p in filenames_and_scores[att][-num_top:]]
        with open(os.path.join(args.output_path, f"att{att}_labels.json"), "w") as outfile:
            json.dump(dict(filenames_minus1 + filenames_plus1), outfile)

if __name__ == '__main__':
    main()