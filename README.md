# Reproducing FairDICE: Fairness-Driven Offline Multi-Objective Reinforcement Learning

This repository contains the codebase of Group 2 for the course 5204FACT6Y - Fairness, Accountability, Confidentiality and Transparency in AI. The directory has the following structure:

- First, we describe the setup of the environment with the necessary libraries for running our scripts.
- Second, we provide the links for downloading the datasets used for training and evaluating the models both on the continuous and discrete domains.
- Third, we provide the source-code oif each experiment verifying the specific claims addressed in our paper in a specific folder (e.g. Claim 3). 


## Setup
  Use the offered Dockerfile for the setup and create conda environment using yml file.
  ```
  cd FairDICE
  conda env create -f environment.yml
  conda activate fairdice
  ```

## Data Download
For the continuous case, the D4MORL dataset, a benchmark suite designed for offline multi-objective reinforcement learning (MORL) was used. The dataset was introduced in the following paper:

To download the data, run:
```
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1wfd6BwAu-hNLC9uvsI1WPEOmPpLQVT9k?usp=sharing --output data
```
For the discrete case, only the Random-MOMDP dataset needs to be downloaded. We provide a dataset for the case in which episodes terminate upon reaching a goal state (in the "data_terminate" folder), and separately the dataset in which episodes go in for a fixed set horizon limit (in the "data" folder). The main plot presented in the for Claim 3 uses the data available in the "data" folder. The datasets can be accessed and downloaded from the following OneDrive directory (permission is granted for people within the UvA network).

- https://amsuni-my.sharepoint.com/:f:/r/personal/karoly_bodgal_student_uva_nl/Documents/5204FACT6Y%20-%20Fairness,%20Accountability,%20Confidentiality%20and%20Transparency%20in%20AI?csf=1&web=1&e=wWSxLH 

## Claim 1

## Claim 2

## Claim 3

## Claim 4



## License
This project is licensed under the MIT License.
