import torch.nn as nn


class BinaryClassification(nn.Module):
    """
    Pytorch model for simple binary classifier. Applies batch norm and dropout
    on all hidden layers.
    """

    def __init__(self, dims,
                 activation_function="relu", dropout=0.1):
        """
        Initialization

        Args:
            dims (list of ints): this list of dimensions is a way of dynamically
                defining the amount of size of the hidden layers. It has format:
                    [input_dim, hidden_layer1_dim, ..., hidden_layerN_dim, output_dim]
            activation_function (string): activation function. currently implemented: ["relu"]
            dropout (float): dropout rate (
        """
        super(BinaryClassification, self).__init__()  # Number of input features is 12.

        self.linear_layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

        if activation_function == "relu":
            self.activation_function = nn.ReLU()

        self.batchnorms = nn.ModuleList([nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 2)])

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward step:

        Args:
            x (torch.tensor): samples
        Returns:
            x (torch.tensor): predictions
        """
        for indx, layer in enumerate(self.linear_layers[:-1]):
            x = self.activation_function(layer(x))
            x = self.batchnorms[indx](x)
            x = self.dropout(x)

        x = self.linear_layers[-1](x)

        return x
