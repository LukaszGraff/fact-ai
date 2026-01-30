#!/bin/bash

conda env create -f environment_peda.yml

source activate peda_env

pip install -r requirements_peda.txt
pip install "d4rl==1.1" --no-deps
