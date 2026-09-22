import sys
from pathlib import Path
# sys.path.append(str(Path.cwd().parent))
sys.path.insert(0, str(Path.cwd()))

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
import argparse

import models

parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int)
parser.add_argument("--thr1", type=float)
parser.add_argument("--thr2", type=float)
parser.add_argument("--model1", type=str)
parser.add_argument("--model2", type=str)
parser.add_argument("--num_points", type=int, default=25)

args = parser.parse_args()

# model1_path = "/home/todd30ap/online-batch-selection/experiments/cifar10_interpolation/9-16-26/DivBS-RhoLoss_log-schedule/20260915_213949_CIFAR10_DivBS_RhoLoss_progThreshold-1_Seed-1/snapshots/epochs/epoch064.pth.tar"
# model2_path = "/home/todd30ap/online-batch-selection/experiments/cifar10_interpolation/9-16-26/DivBS-RhoLoss_log-schedule/20260915_213949_CIFAR10_DivBS_RhoLoss_progThreshold-1_Seed-1/snapshots/epochs/epoch075.pth.tar"

# epoch = 64
# thr1 = 0.0
# thr2 = 0.0
# num_points = 25
# chart_title = f"DivBS-RhoLoss Interpolation between Epoch: 64 and Epoch: 75 with Thr: 1.0"
# data_name = f"DivBS-RhoLos_thr-1_epoch-64-75"

model1_path = args.model1
model2_path = args.model2

epoch = args.epoch
thr1 = args.thr1
thr2 = args.thr2
num_points = args.num_points

chart_title = f"DivBS-RhoLoss Interpolation between Thr: {thr1} and Thr: {thr2} at Epoch: {epoch}"
data_name = f"Train_DivBS-RhoLos_thr-{thr1}-{thr2}_epoch-{epoch}"


def get_model(model_path):
    snapshot = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = snapshot['state_dict']

    model = models.ResNet(m_type='resnet18', num_classes=10)
    model.load_state_dict(state_dict)
    return model

def interpolate_models(model1, model2):
    model = copy.deepcopy(model1)
    model.eval()

    params = list(model.parameters())
    params1 = list(model1.parameters())
    params2 = list(model2.parameters())

    # buffers = list(model.buffers())
    # buffers1 = list(model1.buffers())
    # buffers2 = list(model2.buffers())

    def model_at_t(t):
        with torch.inference_mode():
            for p, p1, p2 in zip(params, params1, params2):
                p.copy_(t * p1 + (1 - t) * p2)

            # for b, b1, b2 in zip(buffers, buffers1, buffers2):
            #     if b.dtype.is_floating_point:
            #         b.copy_(t * b1 + (1 - t) * b2)
            #     else:
            #         b.copy_(b1)
        return model
    return model_at_t

def plot_data(model1, model2, data_loader, loss_func, device, chart_title, data_name, num_points):
    print("Starting Interpolation", flush=True)

    with torch.inference_mode():
        model_func = interpolate_models(model1, model2)

        print("Interpolating Models Function Created", flush=True)
        inputs = np.linspace(0, 1, num_points)
        losses = []
        for t in inputs:
            print(f"Interpolating Model {t}", flush=True)
            model = model_func(t)
            model.train()
            loss = 0
            for images, targets in data_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss += loss_func(outputs, targets).item()
            losses.append(loss / len(data_loader))

        print("Saving Data", flush=True)
        np.save(f"./interpolation/data/{data_name}.npy", np.array(losses))

        print("Generating Plot", flush=True)
        plt.plot(inputs, losses)
        plt.xlabel('t')
        plt.ylabel('Loss')
        plt.title(chart_title)
        plt.savefig(f"./interpolation/plots/{data_name}.png")
        print("Plot Saved", flush=True)

print("Starting", flush=True)

device = torch.device("cuda")

print("Loading Model 1", flush=True)
model1 = get_model(model1_path).to(device)
print("Loading Model 2", flush=True)
model2 = get_model(model2_path).to(device)
print("Finished Loading Models", flush=True)

print("setting vars", flush=True)
im_size = (32, 32)
num_classes = 10
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

print("setting transforms", flush=True)
transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 32 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

print("Loading Dataset", flush=True)
dataset = datasets.CIFAR10('./_CIFAR', train=True, download=False, transform=transform)
print("Creating DataLoader", flush=True)
data_loader = DataLoader(dataset, batch_size=2048, shuffle=False)
print("DataLoader Created", flush=True)

plot_data(model1, model2, data_loader, nn.CrossEntropyLoss(), device, chart_title, data_name, num_points)