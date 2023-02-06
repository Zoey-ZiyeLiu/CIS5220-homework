import torch
from typing import Callable


class MLP(torch.nn.Module):
    """
    MLP multilayer perceptron
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        # exec("self.actv = torch.nn.%s()" % activation)
        self.actv = activation()
        self.layers = torch.nn.ModuleList()
        in_dim = input_size
        self.initializer = initializer
        for i in range(hidden_count - 1):
            out_dim = hidden_size // 2
            lay = torch.nn.Linear(in_dim, out_dim)
            self.initializer(lay.weight)
            self.layers += [lay]
            self.layers += [torch.nn.Dropout(0.1)]
            in_dim = out_dim
        self.out = torch.nn.Linear(in_dim, num_classes)

        # self.actv=activation
        # print(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        # print(x.shape)
        # x = x.view(x.shape[0], -1)
        # print(x.shape)
        for layer in self.layers:
            x = self.actv(layer(x))
        x = self.out(x)
        return x
