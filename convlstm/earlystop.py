import torch

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Attributes:
    -----------
    patience: int
        Number of epochs to wait after the last best validation loss.
    delta: float
        Minimum change in the monitored quantity to qualify as an improvement.
    path: str
        Path to save the model.
    counter: int
        Counter for the number of epochs where the validation loss didn't improve.
    best_score: float
        Best score.
    early_stop: bool
        If True, stop the training.

    Examples
    --------
    >>> patience = 5
    >>> delta = 0.
    >>> path = './checkpoint.pth'
    >>> early_stopping = EarlyStopping(patience, delta, path)
    """
    def __init__(self, patience=5, delta=0., path='./checkpoint.pth'):
        """
        Initialize the EarlyStopping.

        Parameters:
        -----------
        patience: int
            Number of epochs to wait after the last best validation loss. Default is 5.
        delta: float
            Minimum change in the monitored quantity to qualify as an improvement. Default is 0.
        path: str
            Path to save the model. Default is './checkpoint.pth'.
        """

        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        """
        Call the EarlyStopping.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """
        Save the model checkpoint.
        
        Parameters:
        -----------
        model: nn.Module
            Model to save.
        """
        torch.save(model.state_dict(), self.path)
