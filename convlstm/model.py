import torch
import torch.nn as nn
from .utils import set_device

device = set_device()

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell.

    Attributes
    ----------
    activation: nn.Module
        Activation function.
    conv: nn.Conv2d
        Convolutional layer.
    w_ci: nn.Parameter
        Weight for the input gate.
    w_co: nn.Parameter
        Weight for the output gate.
    w_cf: nn.Parameter
        Weight for the forget gate.
    
    Examples
    --------
    >>> input_channels = 3
    >>> output_channels = 64
    >>> kernel_size = 3
    >>> padding = 1
    >>> activation = 'tanh'
    >>> frame_size = (128, 128)
    >>> conv_lstm_cell = ConvLSTMCell(input_channels, output_channels, kernel_size,
    padding, activation, frame_size)
    >>> input_tensor = torch.randn(1, input_channels, *frame_size)
    >>> h_prev = torch.randn(1, output_channels, *frame_size)
    >>> c_prev = torch.randn(1, output_channels, *frame_size)
    >>> h_out, c_out = conv_lstm_cell(input_tensor, H_prev, C_prev)
    """
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 padding,
                 activation,
                 frame_size):
        """
        Initialize the ConvLSTM cell.
        
        Parameters
        ----------
        input_channels: int
            Number of channels of the input tensor.
        output_channels: int
            Number of channels of the output tensor.
        kernel_size: int
            Size of the convolutional kernel.
        padding: int
            Padding of the convolutional kernel.
        activation: str
            Activation function name.
        frame_size: tuple
            Size of the image (height, width).
        """
        super(ConvLSTMCell, self).__init__()

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        self.conv = nn.Conv2d(input_channels+output_channels,
                              4*output_channels,
                              kernel_size,
                              padding=padding)

        # initialize the weights
        self.w_ci = nn.Parameter(torch.Tensor(output_channels, *frame_size))
        self.w_co = nn.Parameter(torch.Tensor(output_channels, *frame_size))
        self.w_cf = nn.Parameter(torch.Tensor(output_channels, *frame_size))

    def forward(self, input_tensor, h_prev, c_prev):
        """
        Forward pass of the ConvLSTM cell.
        
        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor.
        H_prev: torch.Tensor
            Previous hidden state tensor.
        C_prev: torch.Tensor
            Previous cell state tensor.
        
        Returns
        -------
        H_out: torch.Tensor
            Hidden state tensor.
        C_out: torch.Tensor
            Current cell state tensor.
        """
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([input_tensor, h_prev], dim=1))
        i_conv, f_conv, c_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.w_ci * c_prev)
        forget_gate = torch.sigmoid(f_conv + self.w_cf * c_prev)

        c_out = forget_gate * c_prev + input_gate * self.activation(c_conv)
        output_gate = torch.sigmoid(o_conv + self.w_co * c_out)

        h_out = output_gate * self.activation(c_out)

        return h_out, c_out

class ConvLSTM(nn.Module):
    """
    Convolutional LSTM layer.

    Attributes
    ----------
    output_channels: int
        Number of channels of the output tensor.
    conv_lstm_cell: ConvLSTMCell
        ConvLSTM cell.
    
    Examples
    --------
    >>> input_channels = 3
    >>> output_channels = 64
    >>> kernel_size = 3
    >>> padding = 1
    >>> activation = 'tanh'
    >>> frame_size = (128, 128)
    >>> conv_lstm = ConvLSTM(input_channels, output_channels, kernel_size,
    padding, activation, frame_size)
    >>> input_tensor = torch.randn(1, input_channels, 10, *frame_size)
    >>> output = conv_lstm(input_tensor)
    """
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 padding,
                 activation,
                 frame_size):
        """
        Initialize the ConvLSTM layer.
        
        Parameters
        ----------
        input_channels: int
            Number of channels of the input tensor.
        output_channels: int
            Number of channels of the output tensor.
        kernel_size: int
            Size of the convolutional kernel.
        padding: int
            Padding of the convolutional kernel.
        activation: str
            Activation function name.
        frame_size: tuple
            Size of the image (height, width).
        """
        super(ConvLSTM, self).__init__()
        self.output_channels = output_channels
        self.conv_lstm_cell = ConvLSTMCell(input_channels, output_channels, kernel_size,
                                           padding, activation, frame_size)

    def forward(self, input_tensor):
        """
        Forward pass of the ConvLSTM layer.
        
        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor.
        
        Returns
        -------
        output: torch.Tensor
            Output tensor.
        """

        # input_tensor: (batch_size, input_channels, seq_len, height, width)
        batch_size, _, seq_len, height, width = input_tensor.size()

        output = torch.zeros(batch_size, self.output_channels, seq_len, height, width,
                             device=device)

        h = torch.zeros(batch_size, self.output_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.output_channels, height, width, device=device)

        for t in range(seq_len):
            h, c = self.conv_lstm_cell(input_tensor[:, :, t], h, c)
            output[:, :, t] = h

        return output

class ConvLSTMModel(nn.Module):
    """
    Convolutional LSTM model. It is a stack of ConvLSTM layers.

    Input shape: (batch_size, input_channels, seq_len, height, width).
    
    Attributes
    ----------
    sequential: nn.Sequential
        Sequential layer.
    conv: nn.Conv2d
        Convolutional layer.
    
    Examples
    --------
    >>> input_channels = 3
    >>> hidden_channels = 64
    >>> kernel_size = 3
    >>> padding = 1
    >>> activation = 'tanh'
    >>> frame_size = (128, 128)
    >>> num_layers = 2
    >>> conv_lstm_model = ConvLSTMModel(input_channels, hidden_channels, kernel_size,
    padding, activation, frame_size, num_layers)
    >>> input_tensor = torch.randn(1, input_channels, 10, *frame_size)
    >>> output = conv_lstm_model(input_tensor)
    """
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size,
                 padding,
                 activation,
                 frame_size,
                 num_layers):
        """
        Initialize the ConvLSTM model.

        Parameters
        ----------
        input_channels: int
            Number of channels of the input tensor.
        hidden_channels: int
            Number of channels of the hidden tensor.
        kernel_size: int
            Size of the convolutional kernel.
        padding: int
            Padding of the convolutional kernel.
        activation: str
            Activation function name.
        frame_size: tuple
            Size of the image (height, width).
        num_layers: int
            Number of ConvLSTM layers.
        """
        super(ConvLSTMModel, self).__init__()

        self.sequential = nn.Sequential()

        self.sequential.add_module('conv_lstm_0',
                                   ConvLSTM(input_channels, hidden_channels, kernel_size,
                                            padding, activation, frame_size))
        self.sequential.add_module('batch_norm_0',
                                   nn.BatchNorm3d(hidden_channels))

        for i in range(1, num_layers):
            self.sequential.add_module(f'conv_lstm_{i}',
                                       ConvLSTM(hidden_channels, hidden_channels, kernel_size,
                                                padding, activation, frame_size))
            self.sequential.add_module(f'batch_norm_{i}',
                                       nn.BatchNorm3d(hidden_channels))

        self.conv = nn.Conv2d(hidden_channels, input_channels, kernel_size, padding=padding)

    def forward(self, x):
        """
        Forward pass of the ConvLSTM model. Get the last time step and apply a convolutional layer.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        
        Returns
        -------
        x: torch.Tensor
            Output tensor.
        """
        x = self.sequential(x)
        x = self.conv(x[:, :, -1])
        return nn.Sigmoid()(x)
    