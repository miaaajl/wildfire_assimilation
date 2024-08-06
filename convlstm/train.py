import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from PIL import ImageFilter
from livelossplot import PlotLosses
from .utils import set_seed, set_device
from .earlystop import EarlyStopping

device = set_device()

def train(model, optimizer, criterion, data_loader):
    """
    Train the model.
    
    Parameters
    ----------
    model: nn.Module
        Model to train.
    optimizer: torch.optim
        Optimizer to use.
    criterion: nn.Module
        Loss function.
    data_loader: DataLoader
        DataLoader.
    
    Returns
    -------
    train_loss: float
        Loss of the training.
    """
    model.train()
    train_loss = 0.
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs.flatten(), targets.flatten())

        optimizer.zero_grad()
        loss.backward()
        train_loss += loss
        optimizer.step()

    return train_loss/len(data_loader.dataset)

def validate(model, criterion, data_loader, early_stopping):
    """
    Validate the model.
    
    Parameters
    ----------
    model: nn.Module
        Model to validate.
    criterion: nn.Module
        Loss function.
    data_loader: DataLoader
        DataLoader.
    early_stopping: EarlyStopping
        Early stopping object.
    
    Returns
    -------
    validation_loss: float
        Loss of the validation.
    early_stop: bool
        Early stopping condition.
    """
    model.eval()
    validation_loss = 0.
    for inputs, targets in data_loader:
        with torch.no_grad():
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.flatten(), targets.flatten())
            validation_loss += loss

    early_stopping(validation_loss/len(data_loader.dataset), model)
    early_stop = early_stopping.early_stop

    return validation_loss/len(data_loader.dataset), early_stop

def evaluate(model, data_loader):
    """
    Evaluate the model.

    Parameters
    ----------
    model: nn.Module
        Model to evaluate.
    data_loader: DataLoader
        DataLoader.
    
    Returns
    -------
    trues: torch.Tensor
        True values.
    preds: torch.Tensor
        Predicted values.
    """
    model.eval()
    trues, preds = [], []
    for inputs, targets in data_loader:
        with torch.no_grad():
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            trues.append(targets.cpu())
            preds.append(outputs.cpu())

    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0)
    return trues, preds

def train_model(model,
                optimizer,
                criterion,
                train_dataset,
                val_dataset,
                batch_size,
                num_epochs,
                patience=10,
                delta=1e-4,
                path='./checkpoint.pth'):
    """
    Train the model and validate it, using early stopping. Plot the loss.

    Parameters
    ----------
    model: nn.Module
        Model to train.
    optimizer: torch.optim
        Optimizer to use.
    criterion: nn.Module
        Loss function.
    train_dataset: Dataset
        Training dataset.
    val_dataset: Dataset
        Validation dataset.
    batch_size: int
        Batch size.
    num_epochs: int
        Number of epochs.
    patience: int
        Patience for early stopping.
    delta: float
        Delta for early stopping.
    path: str
        Path to save the model.
    
    Returns
    -------
    model: nn.Module
        Trained model.
    
    Examples
    --------
    >>> import ...
    >>> model = ConvLSTMModel(...)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> criterion = nn.MSELoss()
    >>> train_dataset = WildfireDatasetImage(...)
    >>> val_dataset = WildfireDatasetImage(...)
    >>> batch_size = 32
    >>> num_epochs = 100
    >>> patience = 10
    >>> delta = 1e-4
    >>> path = './checkpoint.pth'
    >>> model = train_model(model, optimizer, criterion,
    train_dataset, val_dataset, batch_size, num_epochs, patience, delta, path)
    """
    set_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    liveloss = PlotLosses()
    early_stopping = EarlyStopping(patience=patience, delta=delta, path=path)
    for _ in range(num_epochs):
        logs = {}
        train_loss = train(model, optimizer, criterion, train_loader)

        logs['' + 'loss'] = train_loss.item()

        validation_loss, early_stop = validate(model, criterion, validation_loader, early_stopping)
        logs['val_' + 'loss'] = validation_loss.item()

        liveloss.update(logs)
        liveloss.draw()

        if early_stop:
            break

    return model

def predict_model(model, data, dataset, seq_len, mode='nearest'):
    """
    Predict the model using background data or observed data.

    Interpolate the predicted values to the original size, and sharpen the image.

    Parameters
    ----------
    model: nn.Module
        Model to predict.
    data: torch.Tensor or numpy.ndarray
        Data to predict.
    dataset: Dataset
        Dataset to use.
    seq_len: int
        Sequence length.
    mode: str
        Interpolation mode.
    
    Returns
    -------
    trues: torch.Tensor
        True values. Shape: (batch_size, seq_len, height, width).
    preds: torch.Tensor
        Predicted values. Shape: (batch_size, seq_len, height, width).

    Examples
    --------
    >>> import ...
    >>> model = ConvLSTMModel(...)
    >>> data = np.load('data.npy')
    >>> dataset = WildfireDatasetImage(...)
    >>> seq_len = 2
    >>> mode = 'nearest'
    >>> trues, preds = predict_model(model, data, dataset, seq_len, mode)
    """
    img_size = data.shape[-1]
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    _, preds = evaluate(model, dataloader)
    trues = data[seq_len:] # get the original true values, without transformed
    nan_mask = torch.isnan(preds)
    preds[nan_mask] = 0
    preds = F.interpolate(preds, size=(img_size, img_size), mode=mode)
    for i in preds:
        i = transforms.ToPILImage()(i)
        i = i.filter(ImageFilter.SHARPEN)
        i = transforms.ToTensor()(i)

    return trues, preds
