# Lewiston - Wildfire

A comprehensive system that predicts a wildfire behavior using Convolutional LSTM, Generative AI, and data assimilation techniques. This system will use historical wildfire data, and satellite imagery feeds, to enhance prediction accuracy and provide actionable insights.

## Background

Predicting wildfires is essential for protecting human life, property, and the environment. It plays a critical role in disaster preparedness and response, enabling proactive measures to mitigate the devastating impacts of wildfires. As the number of wildfires increases with climate change, the development and implementation of advanced predictive models become even more crucial for ensuring safety, sustainability, and resilience in vulnerable regions.

This project aims to:

- Build a SURROGATE MODEL using RNN - Use a recurrent neural netword to train a suurogate model of a wildfires predictive model
- Build a SURROGATE MODEL using GENERATIVE AI - Use a generative AI method to train a wildfires generative model
- CORRECTION using Data Assimilation - Perform DA with the results of previous two models in a reduced space

## Getting Started

To run the code in this repo, please follow the steps below. You can skip the step of creating a new virtual environment if necessary.

1. Create a clean conda environment with Python >= 3.8 and activate it

```shell
conda create -n lewiston
conda activate lewiston
```

2. Clone this repository

```bash
git clone https://github.com/ese-msc-2023/acds3-wildfire-lewiston.git
```

3. Run commands below to install necessary libraries:

```bash
cd the-directory-name-where-the-repo-is
pip install -r requirements.txt
```

### Objective 1 - Convolutional LSTM Package

This package provides tools for loading wildfire `.npy` data and creating dataset, creating and training convolutional LSTM (ConvLSTM) models, and predicting background and observation data. 

#### Installation

```shell
cd convlstm
pip install .  # or pip install -e .
```

Then you can use this package to do testing(just run `tox`) etc.

#### Contents

- `model.py`: Defines the convolutional LSTM model.
- `train.py`: Contains training, validating, evaluating and predicting scripts and functions.
- `dataset.py`: Provides dataset utilities for loading and preprocessing data.
- `earlystop.py`: Defines early stop for training.
- `utils.py`: Various utility functions used throughout the package.

#### Usage

##### Model

To create and use the convLSTM model, you can do the following:

```python
from convlstm import ConvLSTMModel
conv_lstm_model = ConvLSTMModel(input_channels, hidden_channels, kernel_size, padding, activation, frame_size, num_layers)
```

##### Training

To train the model, use the `train.py` script:

```python
from convlstm import train_model
model = train_model(model, optimizer, criterion, train_dataset, val_dataset, batch_size, num_epochs, patience, delta, path)
```

##### Dataset

To load and preprocess your dataset, use the functions provided in `dataset.py`:

```python
from convlstm import WildfireDatasetImage
dataset = WildfireDatasetImage(data_path, seq_len, time_step, window_size, img_size, transform)
```

##### Utilities

For various utility functions, see `utils.py`:

```python
from convlstm import set_device
device = set_device()
```

### Objective 2 - Variational AutoEncoder package

This package provides tools for loading wildfire `.npy` data and creating dataset, creating and training VAE model, and predicting background and observation data. 

#### Installation

```shell
cd vae
pip install .  # or pip install -e .
```

Then you can use this package to do testing(just run `tox`) etc.

#### Contents

- `model.py`: Defines the VAE model.
- `train.py`: Contains training, validating, evaluating and predicting scripts and functions.
- `dataset.py`: Provides dataset utilities for loading and preprocessing data.
- `utils.py`: Various utility functions used throughout the package.

#### Usage

##### Model

To create and use the convLSTM model, you can do the following:

```python
from vae import VAE
vae_model = VAEVAE('cuda', input_dim, hidden_dim, latent_dim, activation="GELU")
```

##### Training

To train the model, use the `train.py` script:

```python
from vae import train_vae
model = train_vae(vae, train_loader, optimizer, epochs, checkpoint_interval, checkpoint_path)
```

##### Dataset

To load and preprocess your dataset, use the functions provided in `dataset.py`:

```python
from vae import WildfireDataset
dataset = WildfireDataset(datapath, n_previous, skipped_frame, transform, flatten=Falsee)
```

##### Utilities

For various utility functions, see `utils.py`:

### Objective 3 - CAE Package

This package provides tools for image processing, including a Convolutional Autoencoder (CAE) for image reconstruction, data assimilation methods, and image processing.
Link to download model: https://drive.google.com/file/d/1z3yZnS0_hGeUdt7j_ATs_8zZF_aTHIL_/view?usp=sharing

#### Installation

```shell
cd cae
pip install .  # or pip install -e .
```

Then you can use this package to do testing(just run `tox`) etc.

#### **Features**

1. Convolutional Autoencoder (CAE):

- Encoder: Compresses input images into lower-dimensional representations.
- Decoder: Reconstructs images from the encoded representations.

2. Data Assimilation:

- Implements data assimilation using a Kalman Filter.

3. Image Processing:

- Converts grayscale images to binary (black and white) images based on a specified threshold.

#### Utilities

Importing Packages

```python
from cae import *
```

Data Assimilation

```python
updated_states = assimilate_data(encoded_background, encoded_observation, B, R)
```

Image Processing

```python
binary_image = binarize_image(image, threshold=0.5)
```

## Repository Structure

```

acds3-wildfire-lewiston
├── gifs
│   └── wildfire_100.gif
├── convlstm
│   └── model.py
│   └── dataset.py
│   └── __init__.py
│   └── README.md
│   └── train.py
│   └── earlystop.py
│   └── utils.py
│   └── pyproject.toml
│   └── tox.ini
│   └── tests/
├── notebooks
│   └── Objective_2_Clean.ipynb
│   └── Objective_2_Exploratory.ipynb
│   └── Objective_1_Clean.ipynb
│   └── Objective_1_Exploratory.ipynb
│   └── data_exploration.ipynb
│   └── Objective_3_Clean.ipynb
│   └── Objective_3_Exploratory.ipynb
├── vae
│   └── __init__.py
│   └── dataset.py
│   └── model.py
│   └── train.py
│   └── utils.py
│   └── pyproject.toml
│   └── tox.ini
│   └── tests/
├── cae
│   └── __init__.py
│   └── model.py
│   └── pyproject.toml
│   └── tox.ini
│   └── tests/
├── other
│   └── vit.py
│   └── conv_vae.py
│   └── vq_vae.py
│   └── utils_objective0.py
│   └── objective0.py
├── reference.md
├── README.md
├── requirement.txt
├── LICENSE

```

## Results & Architectures

| Objective | Model                                        | Architectures                                            | Parameters                                                   | MSE                       |
| --------- | -------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------ | ------------------------- |
| 1         | ConvLSTM                                     | 2 ConvLSTM layers<br>2 BatchNorm3d layers<br>1 CNN layer | Image Size: (128, 128)<br>Hidden Channels: 16<br>Sequence Length: 2<br>Loss Function: BCE Loss<br>Learning Rate: 1e-5 | 0.08110                   |
| 2         | VAE                                          | Hidden layers - 256, 128; latent space - 32;                                                          | Batch size - 16; Every 10 images; Loss function BCE Loss; Learning Rate: 0.001                                                              | BCE Loss: 50.1; MSE (with obs): 0.1                         |
| 3         | Final Verision:CAE Compression+ Assimilation | 3 Con2D AutoEncoder Model + 5 Images Assimilation        | Matrix B weight: 1; Matrix R weight: 10                      | Before: 0,1; After: 0.004 |



## Data

We train and test models on the dataset which contains information below:

- [Ferguson_fire_train](https://imperiallondon-my.sharepoint.com/:u:/g/personal/rarcucci_ic_ac_uk/EXWfaCKGdupKjmjM-OT7_Q8B1ImRnWCp6UT-YCW0oOHcTA?e=L2wKYC): training data obtained from wildfires simulation, with shape `(12500, 256, 256)`.
- [Ferguson_fire_test](https://imperiallondon-my.sharepoint.com/:u:/g/personal/rarcucci_ic_ac_uk/EVrPGPqX7-dEmu4DPWadwngBZN7MLau1XKL5hJnwDTf4Ag?e=QywXIq): similar to Ferguson_file_train but obtained from different simulations, with shape `(5000, 256, 256)`.
- [Ferguson_fire_background](https://imperiallondon-my.sharepoint.com/:u:/g/personal/rarcucci_ic_ac_uk/EdKSgIONELRFtLdBEuoyX-4BoLSo0DgWWi3EbXETIYFHGw?e=Eks1v6): model data to be used for the data assimilation, with shape `(5, 256, 256)`.
- [Ferguson_fire_obs](https://imperiallondon-my.sharepoint.com/:u:/g/personal/rarcucci_ic_ac_uk/ETERVdjpQ1BPrVLgu27TPawB5QgkpTGJ2_q3Z31qwBiYfg?e=CTvaSB): observation data at different days after ignition(only one trajector), with shape `(5, 256, 256)`.

All are already pre-processed. The time step between Ferguson_fire_train and Ferguson_fire_test is 1, while that between Ferguson_fire_background and Ferguson_fire_obs is 10.

You can access the dataset by clicking those links.

Note: in the background file you will find model data already selected in time steps corresponding to satellites observed data.

## Authors

- Atys Panier: atys.panier23@imperial.ac.uk
- Benjamin Duncan: benjamin.duncan23@imperial.ac.uk
- Jingyi Liao: jingyi.liao23@imperial.ac.uk
- Rui Li: rui.li23@imperial.ac.uk
- Ruihan He: ruihan.he23@imperial.ac.uk
- Vanya Arikutharam: vanya.arikutharam23@imperial.ac.uk
- Yunjie Li: yunjie.li23@imperial.ac.uk
- Zuzanna Sadkiewicz: zuzanna.sadkiewicz23@imperial.ac.uk

## Reference

See the reference.md file

## License

Please find the MIT License for this repository [here](https://github.com/ese-msc-2023/acds3-wildfire-lewiston/blob/main/LICENSE).
