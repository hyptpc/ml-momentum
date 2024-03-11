#!/bin/sh

conda create -n ml-mom python=3.10 -y
conda init bash
conda activate ml-mom
conda install -y numpy psutil pyyaml rich matplotlib tqdm statistics
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
conda install -y pyg -c pyg