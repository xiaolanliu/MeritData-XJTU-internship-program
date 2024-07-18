import torch
from torch import nn
from torchsummary import summary


def trans(data_input):
    weeks = torch.reshape(data_input, (10, 52, 168))
    transformation = torch.zeros(10, 676, 2184)
    for i in range(13):
        row = i * 52
        col = i * 168
        transformation[:, row:row + 52, col:col + 168] += weeks
    return transformation


class C3DNet(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.sig = nn.Sigmoid()
        self.Conv1 = nn.Conv3d(1, 1, (6, 2, 2), stride=(2, 1, 2), padding=0)
        self.Conv2 = nn.Conv3d(1, 1, (4, 1, 2), stride=(4, 1, 2), padding=0)
        self.Pool3 = nn.AvgPool3d(2, stride=2, padding=0)
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
        self.linear4 = nn.Linear(in_features=27, out_features=80)
        self.linear5 = nn.Linear(in_features=80, out_features=200)
        self.linear6 = nn.Linear(in_features=200, out_features=676)
        self.softmax = nn.Softmax()
        self.batch_size = batch_size

    def forward(self, x):
        x = x[0]  # summary矫正项  [1,1,52,7,24]
        transformation = trans(x)
        x = self.sig(self.Conv1(x))  # [1,1,52,7,24]
        x = self.sig(self.Conv2(x))  # [1,1,6,6,6]
        x = self.Pool3(x)  # [1,1,3,3,3]
        x = self.flatten(x)  # [27]
        x = self.sig(self.linear4(x))  # [80]
        x = self.sig(self.linear5(x))  # [200]
        x = self.sig(self.linear6(x))  # [676]
        x = torch.einsum('bj,bjk->bk', [x, transformation])  # [2184]
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = C3DNet(10).to(device)

    print(summary(model, (10, 1, 1, 52, 7, 24)))
    print('1')
