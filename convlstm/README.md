# ConvLSTM

## Overview

This package provides tools for loading wildfire `.npy` data and creating dataset, creating and training convolutional LSTM (ConvLSTM) models, and predicting background and observation data.

## Getting started

### Prerequisites

* Anaconda

MacOS: https://docs.anaconda.com/free/anaconda/install/mac-os/

Windows: https://docs.anaconda.com/free/anaconda/install/windows/

Linux: https://docs.anaconda.com/free/anaconda/install/linux/

### Installation

1. Create a clean conda environment with Python >= 3.8 and activate it

```shell
conda create -n lewiston
conda activate lewiston
```

2. Clone this repository

```shell
git clone https://github.com/ese-msc-2023/acds3-wildfire-lewiston.git
```

3. Navigate into this repository and then pip-install it. Please assure run the pip-install in the directory where convlstm directory and pyproject.toml are.

```shell
cd the-directory-name-where-convlstm-and-pyproject.toml-are
pip install .  # or pip install -e .
```

Then you can use this package to do testing(just run `pytest`) etc.

## Contents

- `model.py`: Defines the convolutional LSTM model.
- `train.py`: Contains training, validating, evaluating and predicting scripts and functions.
- `dataset.py`: Provides dataset utilities for loading and preprocessing data.
- `earlystop.py`: Defines early stop for training.
- `utils.py`: Various utility functions used throughout the package.

## Usage

### Model

To create and use the convLSTM model, you can do the following:

```python
from convlstm import ConvLSTMModel
conv_lstm_model = ConvLSTMModel(input_channels, hidden_channels, kernel_size, padding, activation, frame_size, num_layers)
```

### Training

To train the model, use the `train.py` script:

```python
from convlstm import train_model
model = train_model(model, optimizer, criterion, train_dataset, val_dataset, batch_size, num_epochs, patience, delta, path)
```

### Dataset

To load and preprocess your dataset, use the functions provided in `dataset.py`:

```python
from convlstm import WildfireDatasetImage
dataset = WildfireDatasetImage(data_path, seq_len, time_step, window_size, img_size, transform)
```

### Utilities

For various utility functions, see `utils.py`:

```python
from convlstm import set_device
device = set_device()
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Authors and Contact

🙋‍♀️Yunjie Li, yunjie.li23@imperial.ac.uk

🙋Ruihan He, rh323@ic.ac.uk

🙋‍♂️Rui Li, rl323@ic.ac.uk

💡Project: https://github.com/ese-msc-2023/acds3-wildfire-lewiston

