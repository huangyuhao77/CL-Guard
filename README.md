# [CL-Guard: Defending DNNs Against Backdoors via Fine-Grained Neuron Analysis and Collaborative Dual-Network Learning]

This repository contains the official implementation of the paper **"CL-Guard: Defending DNNs Against Backdoors via Fine-Grained Neuron Analysis and Collaborative Dual-Network Learning"**, submitted to AAAI 2026.

---

## 1. Overview
This project implements the proposed method **[CL-Guard]** described in our paper. It is designed to eliminate abnormal backdoor behaviors in DNN models.

## 2. Installation

You can run the following script to configurate necessary environment

```bash
  git clone git@gitee.com:dispy54692/clguard.git
  cd ClGuard
  conda create -n ClGuard python=3.8
  conda activate ClGuard
  sh ./sh/install.sh
```

## 3. Project Structure
```
├── README.md                # Project description
├── requirements.txt         # Dependency list
├── config/                  # Experiment configuration
├── data/                    # Dataset
├── utils/                   # Helper functions
├── clguard/                 # Main method
│   └── main.py       
├── sh/                      # Installation scripts                      
└── results/                 # Experiment results and model checkpoints
```
## 4. Usage
### Step 1: Prepare the dataset
Download or place datasets (e.g., CIFAR-10, GTSRB) under the data/ directory.

### Step 2: Train a backdoored model
The implementation can be referred to the attack modules in BackdoorBench.

### Step 3: Apply the defense method (CLGuard)
```bash
  python main.py --result_file "badnet_gtsrb_vgg16/badnet_m2n_1_pn500/"  --ratio 0.1 --runtype 'train'
 ```

### Step 4: Evaluate the defended model
```bash
  python main.py --result_file "badnet_gtsrb_vgg16/badnet_m2n_1_pn500/"  --runtype 'evaluate'
```
