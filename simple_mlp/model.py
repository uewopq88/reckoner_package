import torch
import torch.nn as nn
import torch.nn.functional as F

class mlp(nn.Module):
    '''
    The mlp classifier
    '''

    def __init__(self, in_size, hidden_size, output_size):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            output_size: encoder output dimension
        Output:
            (return value in forward) a tensor of shape (batch_size, output_size)
        '''
        super(mlp, self).__init__()
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.linear_3 = nn.Linear(output_size, 1)
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        torch.nn.init.xavier_uniform_(self.linear_3.weight)

        self.linear_1.bias.data.fill_(0)
        self.linear_2.bias.data.fill_(0)
        self.linear_3.bias.data.fill_(0)

    def forward(self, input_f):
        '''
        Args:
            input_f: tensor of shape (batch_size, in_size)
        '''
        y_1 = F.relu(self.linear_1(input_f))
        y_2 = self.linear_2(y_1)
        y_3 = self.linear_3(y_2)
        output = self.sigmoid(y_3)

        return output

class NoiseGenerator(nn.Module):
    '''
    Learnable noise generator
    '''
    def __init__(self, in_size, hidden_size, output_size):
        '''
        Args:
            noise
        Output:
            (return value in forward) a tensor of shape (batch_size, output_size)
        '''
        super(NoiseGenerator, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, noise):
        x = self.fc1(noise)
        generated_noise = torch.tanh(self.fc2(x))
        return generated_noise
