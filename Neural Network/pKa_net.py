import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class PkaNet(nn.Module):

    def __init__(self):
        super(PkaNet, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(20, 64, 5, padding=2)  # input:(n, 21, 21, 21, 20) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(F.relu(self.conv1(x)), kernel_size=2, padding=1)
        x = F.max_pool3d(F.relu(self.conv2(x)), kernel_size=2, padding=1)
        x = F.max_pool3d(F.relu(self.conv3(x)), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.drop3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN1(nn.Module):

    def __init__(self):
        super(PkaNetBN1, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(20, 64, 5, padding=2)  # input:(n, 21, 21, 21, 20) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm1d(1000)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(F.relu(self.conv1(x)), kernel_size=2, padding=1)
        x = self.bn1(x)
        x = F.max_pool3d(F.relu(self.conv2(x)), kernel_size=2, padding=1)
        x = F.max_pool3d(F.relu(self.conv3(x)), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn2(F.relu(self.fc1(x))))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.drop3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN2(nn.Module):

    def __init__(self):
        super(PkaNetBN2, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(20, 64, 5, padding=2)  # input:(n, 21, 21, 21, 20) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(20)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm1d(1000)
        self.bn6 = nn.BatchNorm1d(500)
        self.bn7 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = self.bn1(x)
        x = F.max_pool3d(F.relu(self.bn2(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(F.relu(self.bn3(self.conv2(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(F.relu(self.bn4(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(F.relu(self.bn5(self.fc1(x))))
        x = self.drop2(F.relu(self.bn6(self.fc2(x))))
        x = self.drop3(F.relu(self.bn7(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN2F18(nn.Module):

    def __init__(self):
        super(PkaNetBN2F18, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(18, 64, 5, padding=2)  # input:(n, 21, 21, 21, 18) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(18)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm1d(1000)
        self.bn6 = nn.BatchNorm1d(500)
        self.bn7 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = self.bn1(x)
        x = F.max_pool3d(F.relu(self.bn2(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(F.relu(self.bn3(self.conv2(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(F.relu(self.bn4(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(F.relu(self.bn5(self.fc1(x))))
        x = self.drop2(F.relu(self.bn6(self.fc2(x))))
        x = self.drop3(F.relu(self.bn7(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN2F18PRELU(nn.Module):

    def __init__(self):
        super(PkaNetBN2F18PRELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(18, 64, 5, padding=2)  # input:(n, 21, 21, 21, 18) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(18)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm1d(1000)
        self.bn6 = nn.BatchNorm1d(500)
        self.bn7 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = self.bn1(x)
        device = x.device
        x = F.max_pool3d(F.prelu(self.bn2(self.conv1(x)), weight=torch.tensor(0.25).to(device)), kernel_size=2,
                         padding=1)
        x = F.max_pool3d(F.prelu(self.bn3(self.conv2(x)), weight=torch.tensor(0.25).to(device)), kernel_size=2,
                         padding=1)
        x = F.max_pool3d(F.prelu(self.bn4(self.conv3(x)), weight=torch.tensor(0.25).to(device)), kernel_size=2,
                         padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(F.prelu(self.bn5(self.fc1(x)), weight=torch.tensor(0.25).to(device)))
        x = self.drop2(F.prelu(self.bn6(self.fc2(x)), weight=torch.tensor(0.25).to(device)))
        x = self.drop3(F.prelu(self.bn7(self.fc3(x)), weight=torch.tensor(0.25).to(device)))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN2F18ELU(nn.Module):

    def __init__(self):
        super(PkaNetBN2F18ELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(18, 64, 5, padding=2)  # input:(n, 21, 21, 21, 18) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(18)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm1d(1000)
        self.bn6 = nn.BatchNorm1d(500)
        self.bn7 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = self.bn1(x)
        x = F.max_pool3d(F.elu(self.bn2(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(F.elu(self.bn3(self.conv2(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(F.elu(self.bn4(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(F.elu(self.bn5(self.fc1(x))))
        x = self.drop2(F.elu(self.bn6(self.fc2(x))))
        x = self.drop3(F.elu(self.bn7(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN2F19ELU(nn.Module):

    def __init__(self):
        super(PkaNetBN2F19ELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(19, 64, 5, padding=2)  # input:(n, 21, 21, 21, 19) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(19)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm1d(1000)
        self.bn6 = nn.BatchNorm1d(500)
        self.bn7 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = self.bn1(x)
        x = F.max_pool3d(F.elu(self.bn2(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(F.elu(self.bn3(self.conv2(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(F.elu(self.bn4(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(F.elu(self.bn5(self.fc1(x))))
        x = self.drop2(F.elu(self.bn6(self.fc2(x))))
        x = self.drop3(F.elu(self.bn7(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN2F11ELU(nn.Module):

    def __init__(self):
        super(PkaNetBN2F11ELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(11, 64, 5, padding=2)  # input:(n, 21, 21, 21, 11) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(11)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm1d(1000)
        self.bn6 = nn.BatchNorm1d(500)
        self.bn7 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = self.bn1(x)
        x = F.max_pool3d(F.elu(self.bn2(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(F.elu(self.bn3(self.conv2(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(F.elu(self.bn4(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(F.elu(self.bn5(self.fc1(x))))
        x = self.drop2(F.elu(self.bn6(self.fc2(x))))
        x = self.drop3(F.elu(self.bn7(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN3F18ELU(nn.Module):

    def __init__(self):
        super(PkaNetBN3F18ELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(18, 64, 5, padding=2)  # input:(n, 21, 21, 21, 18) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm1d(1000)
        self.bn5 = nn.BatchNorm1d(500)
        self.bn6 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(F.elu(self.bn1(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(F.elu(self.bn2(self.conv2(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(F.elu(self.bn3(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(F.elu(self.bn4(self.fc1(x))))
        x = self.drop2(F.elu(self.bn5(self.fc2(x))))
        x = self.drop3(F.elu(self.bn6(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN3F19S21ELU(nn.Module):

    def __init__(self):
        super(PkaNetBN3F19S21ELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(19, 64, 5, padding=2)  # input:(n, 21, 21, 21, 19) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm1d(1000)
        self.bn5 = nn.BatchNorm1d(500)
        self.bn6 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(F.elu(self.bn1(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(F.elu(self.bn2(self.conv2(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(F.elu(self.bn3(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(F.elu(self.bn4(self.fc1(x))))
        x = self.drop2(F.elu(self.bn5(self.fc2(x))))
        x = self.drop3(F.elu(self.bn6(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN4F18ELU(nn.Module):

    def __init__(self):
        super(PkaNetBN4F18ELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(18, 64, 5, padding=2)  # input:(n, 21, 21, 21, 18) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm1d(1000)
        self.bn5 = nn.BatchNorm1d(500)
        self.bn6 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(self.bn1(F.elu(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn2(F.elu(self.conv2(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn3(F.elu(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn4(F.elu(self.fc1(x))))
        x = self.drop2(self.bn5(F.elu(self.fc2(x))))
        x = self.drop3(self.bn6(F.elu(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN4F19S21ELU(nn.Module):

    def __init__(self):
        super(PkaNetBN4F19S21ELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(19, 64, 5, padding=2)  # input:(n, 21, 21, 21, 19) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm1d(1000)
        self.bn5 = nn.BatchNorm1d(500)
        self.bn6 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(self.bn1(F.elu(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn2(F.elu(self.conv2(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn3(F.elu(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn4(F.elu(self.fc1(x))))
        x = self.drop2(self.bn5(F.elu(self.fc2(x))))
        x = self.drop3(self.bn6(F.elu(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN4F19S17ELU(nn.Module):

    def __init__(self):
        super(PkaNetBN4F19S17ELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(19, 64, 5, padding=2)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm1d(1000)
        self.bn5 = nn.BatchNorm1d(500)
        self.bn6 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(self.bn1(F.elu(self.conv1(x))), kernel_size=2, padding=1)  # input:(n, 17, 17, 17, 20) -> out:(n, 9, 9, 9, 64)
        x = F.max_pool3d(self.bn2(F.elu(self.conv2(x))), kernel_size=2, padding=1)  # input:(n, 9, 9, 9, 64) -> out:(n, 5, 5, 5, 128)
        x = F.max_pool3d(self.bn3(F.elu(self.conv3(x))), kernel_size=2, padding=1)  # input:(n, 5, 5, 5, 128) -> out:(n, 3, 3, 3, 256)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn4(F.elu(self.fc1(x))))
        x = self.drop2(self.bn5(F.elu(self.fc2(x))))
        x = self.drop3(self.bn6(F.elu(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN4F19S19ELU(nn.Module):

    def __init__(self):
        super(PkaNetBN4F19S19ELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(19, 64, 5, padding=2)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm1d(1000)
        self.bn5 = nn.BatchNorm1d(500)
        self.bn6 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(self.bn1(F.elu(self.conv1(x))), kernel_size=2, padding=1)  # input:(n, 19, 19, 19, 20) -> out:(n, 10, 10, 10, 64)
        x = F.max_pool3d(self.bn2(F.elu(self.conv2(x))), kernel_size=2, padding=0)  # input:(n, 10, 10, 10, 64) -> out:(n, 5, 5, 5, 128)
        x = F.max_pool3d(self.bn3(F.elu(self.conv3(x))), kernel_size=2, padding=1)  # input:(n, 5, 5, 5, 128) -> out:(n, 3, 3, 3, 256)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn4(F.elu(self.fc1(x))))
        x = self.drop2(self.bn5(F.elu(self.fc2(x))))
        x = self.drop3(self.bn6(F.elu(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




class PkaNetBN4F19K96ELU(nn.Module):

    def __init__(self):
        super(PkaNetBN4F19K96ELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(19, 96, 5, padding=2)  # input:(n, 21, 21, 21, 19) -> out:(n, 11, 11, 11, 96)
        self.conv2 = nn.Conv3d(96, 192, 5, padding=2)  # input:(n, 11, 11, 11, 96) -> out:(n, 6, 6, 6, 192)
        self.conv3 = nn.Conv3d(192, 384, 5, padding=2)  # input:(n, 6, 6, 6, 192) -> out:(n, 3, 3, 3, 384)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(96)
        self.bn2 = nn.BatchNorm3d(192)
        self.bn3 = nn.BatchNorm3d(384)
        self.bn4 = nn.BatchNorm1d(1500)
        self.bn5 = nn.BatchNorm1d(750)
        self.bn6 = nn.BatchNorm1d(300)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(10368, 1500)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1500, 750)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(750, 300)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(300, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(self.bn1(F.elu(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn2(F.elu(self.conv2(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn3(F.elu(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn4(F.elu(self.fc1(x))))
        x = self.drop2(self.bn5(F.elu(self.fc2(x))))
        x = self.drop3(self.bn6(F.elu(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN3(nn.Module):

    def __init__(self):
        super(PkaNetBN3, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(20, 64, 5, padding=2)  # input:(n, 21, 21, 21, 20) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(20)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm1d(1000)
        self.bn6 = nn.BatchNorm1d(500)
        self.bn7 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(F.relu(self.bn2(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(F.relu(self.bn3(self.conv2(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(F.relu(self.bn4(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(F.relu(self.bn5(self.fc1(x))))
        x = self.drop2(F.relu(self.bn6(self.fc2(x))))
        x = self.drop3(F.relu(self.bn7(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetF19S21ELU(nn.Module):

    def __init__(self):
        super(PkaNetF19S21ELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(19, 64, 5, padding=2)  # input:(n, 21, 21, 21, 19) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(F.elu(self.conv1(x)), kernel_size=2, padding=1)
        x = F.max_pool3d(F.elu(self.conv2(x)), kernel_size=2, padding=1)
        x = F.max_pool3d(F.elu(self.conv3(x)), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(F.elu(self.fc1(x)))
        x = self.drop2(F.elu(self.fc2(x)))
        x = self.drop3(F.elu(self.fc3(x)))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetF19S21RELU(nn.Module):

    def __init__(self):
        super(PkaNetF19S21RELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(19, 64, 5, padding=2)  # input:(n, 21, 21, 21, 19) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(F.relu(self.conv1(x)), kernel_size=2, padding=1)
        x = F.max_pool3d(F.relu(self.conv2(x)), kernel_size=2, padding=1)
        x = F.max_pool3d(F.relu(self.conv3(x)), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.drop3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class PkaNetF19S21PRELU(nn.Module):

    def __init__(self):
        super(PkaNetF19S21PRELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(19, 64, 5, padding=2)  # input:(n, 21, 21, 21, 19) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        device = x.device
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(F.prelu(self.conv1(x), weight=torch.tensor(0.25).to(device)), kernel_size=2, padding=1)
        x = F.max_pool3d(F.prelu(self.conv2(x), weight=torch.tensor(0.25).to(device)), kernel_size=2, padding=1)
        x = F.max_pool3d(F.prelu(self.conv3(x), weight=torch.tensor(0.25).to(device)), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(F.prelu(self.fc1(x), weight=torch.tensor(0.25).to(device)))
        x = self.drop2(F.prelu(self.fc2(x), weight=torch.tensor(0.25).to(device)))
        x = self.drop3(F.prelu(self.fc3(x), weight=torch.tensor(0.25).to(device)))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN4F19S21PRELU(nn.Module):

    def __init__(self):
        super(PkaNetBN4F19S21PRELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(19, 64, 5, padding=2)  # input:(n, 21, 21, 21, 19) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm1d(1000)
        self.bn5 = nn.BatchNorm1d(500)
        self.bn6 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        device = x.device
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(self.bn1(F.prelu(self.conv1(x), weight=torch.tensor(0.25).to(device))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn2(F.prelu(self.conv2(x), weight=torch.tensor(0.25).to(device))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn3(F.prelu(self.conv3(x), weight=torch.tensor(0.25).to(device))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn4(F.prelu(self.fc1(x), weight=torch.tensor(0.25).to(device))))
        x = self.drop2(self.bn5(F.prelu(self.fc2(x), weight=torch.tensor(0.25).to(device))))
        x = self.drop3(self.bn6(F.prelu(self.fc3(x), weight=torch.tensor(0.25).to(device))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN4F18S21ELU(nn.Module):

    def __init__(self):
        super(PkaNetBN4F18S21ELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(18, 64, 5, padding=2)  # input:(n, 21, 21, 21, 18) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm1d(1000)
        self.bn5 = nn.BatchNorm1d(500)
        self.bn6 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(self.bn1(F.elu(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn2(F.elu(self.conv2(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn3(F.elu(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn4(F.elu(self.fc1(x))))
        x = self.drop2(self.bn5(F.elu(self.fc2(x))))
        x = self.drop3(self.bn6(F.elu(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN4F20S21ELU(nn.Module):

    def __init__(self):
        super(PkaNetBN4F20S21ELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(20, 64, 5, padding=2)  # input:(n, 21, 21, 21, 20) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm1d(1000)
        self.bn5 = nn.BatchNorm1d(500)
        self.bn6 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(self.bn1(F.elu(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn2(F.elu(self.conv2(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn3(F.elu(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn4(F.elu(self.fc1(x))))
        x = self.drop2(self.bn5(F.elu(self.fc2(x))))
        x = self.drop3(self.bn6(F.elu(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN4F20S19ELU(nn.Module):

    def __init__(self):
        super(PkaNetBN4F20S19ELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(20, 64, 5, padding=2)  # input:(n, 19, 19, 19, 20) -> out:(n, 10, 10, 10, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 10, 10, 10, 64) -> out:(n, 5, 5, 5, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 5, 5, 5, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm1d(1000)
        self.bn5 = nn.BatchNorm1d(500)
        self.bn6 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(self.bn1(F.elu(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn2(F.elu(self.conv2(x))), kernel_size=2, padding=0)
        x = F.max_pool3d(self.bn3(F.elu(self.conv3(x))), kernel_size=2, padding=1)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn4(F.elu(self.fc1(x))))
        x = self.drop2(self.bn5(F.elu(self.fc2(x))))
        x = self.drop3(self.bn6(F.elu(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN4F20S23ELU(nn.Module):

    def __init__(self):
        super(PkaNetBN4F20S23ELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(20, 64, 5, padding=2)  # input:(n, 23, 23, 23, 20) -> out:(n, 12, 12, 12, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 12, 12, 12, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm1d(1000)
        self.bn5 = nn.BatchNorm1d(500)
        self.bn6 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(self.bn1(F.elu(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn2(F.elu(self.conv2(x))), kernel_size=2, padding=0)
        x = F.max_pool3d(self.bn3(F.elu(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn4(F.elu(self.fc1(x))))
        x = self.drop2(self.bn5(F.elu(self.fc2(x))))
        x = self.drop3(self.bn6(F.elu(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PkaNetBN4F20S21RELU(nn.Module):

    def __init__(self):
        super(PkaNetBN4F20S21RELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(20, 64, 5, padding=2)  # input:(n, 21, 21, 21, 20) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm1d(1000)
        self.bn5 = nn.BatchNorm1d(500)
        self.bn6 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(500, 200)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(self.bn1(F.relu(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn2(F.relu(self.conv2(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn3(F.relu(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn4(F.relu(self.fc1(x))))
        x = self.drop2(self.bn5(F.relu(self.fc2(x))))
        x = self.drop3(self.bn6(F.relu(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class PkaNetRachelRELU(nn.Module):

    def __init__(self):
        super(PkaNetRachelELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(20, 64, 5, padding=2)  # input:(n, 23, 23, 23, 20) -> out:(n, 12, 12, 12, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 12, 12, 12, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.dl1 = 1000
        self.dl2 = 500
        self.dl3 = 200
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm1d(self.dl1)
        self.bn5 = nn.BatchNorm1d(self.dl2)
        self.bn6 = nn.BatchNorm1d(self.dl3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, self.dl1)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(self.dl1, self.dl2)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(self.dl2, self.dl3)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(self.dl3, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(self.bn1(F.elu(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn2(F.elu(self.conv2(x))), kernel_size=2, padding=0)
        x = F.max_pool3d(self.bn3(F.elu(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn4(F.elu(self.fc1(x))))
        x = self.drop2(self.bn5(F.elu(self.fc2(x))))
        x = self.drop3(self.bn6(F.elu(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
class PkaNetRachelRELU(nn.Module):

    def __init__(self):
        super(PkaNetRachelELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(20, 64, 5, padding=2)  # input:(n, 23, 23, 23, 20) -> out:(n, 12, 12, 12, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 12, 12, 12, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.dl1 = 1000
        self.dl2 = 500
        self.dl3 = 200
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm1d(self.dl1)
        self.bn5 = nn.BatchNorm1d(self.dl2)
        self.bn6 = nn.BatchNorm1d(self.dl3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, self.dl1)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(self.dl1, self.dl2)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(self.dl2, self.dl3)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(self.dl3, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(self.bn1(F.elu(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn2(F.elu(self.conv2(x))), kernel_size=2, padding=0)
        x = F.max_pool3d(self.bn3(F.elu(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn4(F.elu(self.fc1(x))))
        x = self.drop2(self.bn5(F.elu(self.fc2(x))))
        x = self.drop3(self.bn6(F.elu(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class PkaNetRachelELU(nn.Module):

    def __init__(self):
        super(PkaNetRachelELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.cv1 = 64
        self.cv2 = 128
        self.cv3 = 256
        self.conv1 = nn.Conv3d(20, self.cv1, 5, padding=2)  # input:(n, 21, 21, 21, 20) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(self.cv1, self.cv2, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(self.cv2, self.cv3, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(self.cv1)
        self.bn2 = nn.BatchNorm3d(self.cv2)
        self.bn3 = nn.BatchNorm3d(self.cv3)
        self.dl1 = 1000
        self.dl2 = 500
        self.dl3 = 200

        self.bn4 = nn.BatchNorm1d(self.dl1)
        self.bn5 = nn.BatchNorm1d(self.dl2)
        self.bn6 = nn.BatchNorm1d(self.dl3)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, self.dl1)
        # Could change this dropout to lower
        # Need to mention dropout layers
        # Also compare RMSE
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(self.dl1, self.dl2)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(self.dl2, self.dl3)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(self.dl3, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(self.bn1(F.elu(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn2(F.elu(self.conv2(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn3(F.elu(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn4(F.elu(self.fc1(x))))
        x = self.drop2(self.bn5(F.elu(self.fc2(x))))
        x = self.drop3(self.bn6(F.elu(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class PkaNetRachelActualRELU(nn.Module):

    def __init__(self):
        super(PkaNetRachelActualRELU, self).__init__()
        # 20 input data channel,  5 x 5 x 5 square convolution
        # kernel
        self.conv1 = nn.Conv3d(20, 64, 5, padding=2)  # input:(n, 21, 21, 21, 20) -> out:(n, 11, 11, 11, 64)
        self.conv2 = nn.Conv3d(64, 128, 5, padding=2)  # input:(n, 11, 11, 11, 64) -> out:(n, 6, 6, 6, 128)
        self.conv3 = nn.Conv3d(128, 256, 5, padding=2)  # input:(n, 6, 6, 6, 128) -> out:(n, 3, 3, 3, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm1d(1000)
        self.bn5 = nn.BatchNorm1d(500)
        self.bn6 = nn.BatchNorm1d(200)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6912, 1000)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(self.bn1(F.relu(self.conv1(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn2(F.relu(self.conv2(x))), kernel_size=2, padding=1)
        x = F.max_pool3d(self.bn3(F.relu(self.conv3(x))), kernel_size=2, padding=0)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn4(F.relu(self.fc1(x))))
        x = self.drop2(self.bn5(F.relu(self.fc2(x))))
        x = self.drop3(self.bn6(F.relu(self.fc3(x))))
        x = self.fc4(x)
        # print(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class DeepKaNet(nn.Module):

    def __init__(self, radii:int, channels:int, first_kernels:int):
        super(DeepKaNet, self).__init__()
        self.input_size = radii * 2 + 1
        self.conv_kernel_size = 5
        self.pool_kernel_size = 2
        self.conv_padding = self.conv_kernel_size // 2
        self.pool1_padding = (self.input_size - self.conv_kernel_size + 1
                              + self.conv_padding * 2) % self.pool_kernel_size
        pool1_outsize = (self.input_size - self.conv_kernel_size + 1 + 2 * self.conv_padding
                         + 2 * self.pool1_padding) // self.pool_kernel_size
        self.pool2_padding = (pool1_outsize - self.conv_kernel_size + 1
                              + 2 * self.conv_padding) % self.pool_kernel_size
        pool2_outsize = (pool1_outsize - self.conv_kernel_size + 1 + 2 * self.conv_padding
                        + 2 * self.pool2_padding) // self.pool_kernel_size
        self.pool3_padding = (pool2_outsize - self.conv_kernel_size + 1
                              + 2 * self.conv_padding) % self.pool_kernel_size
        pool3_outsize = (pool2_outsize - self.conv_kernel_size + 1 + 2 * self.conv_padding
                        + 2 * self.pool3_padding) // self.pool_kernel_size
        conv2_kernels = first_kernels * 2
        conv3_kernels = conv2_kernels * 2
        fc1_in_features = pool3_outsize ** 3 * conv3_kernels
        print(fc1_in_features)
        fc2_in_features = 1000
        fc3_in_featrues = 500
        fc4_in_features = 200
        drop_rate = 0.5
        final_out_features = 1
        # conv kernel
        self.conv1 = nn.Conv3d(channels, first_kernels, self.conv_kernel_size, padding=self.conv_padding)
        self.conv2 = nn.Conv3d(first_kernels, conv2_kernels, self.conv_kernel_size, padding=self.conv_padding)
        self.conv3 = nn.Conv3d(conv2_kernels, conv3_kernels, self.conv_kernel_size, padding=self.conv_padding)
        # batch normalize layer
        self.bn1 = nn.BatchNorm3d(first_kernels)
        self.bn2 = nn.BatchNorm3d(conv2_kernels)
        self.bn3 = nn.BatchNorm3d(conv3_kernels)
        self.bn4 = nn.BatchNorm1d(fc2_in_features)
        self.bn5 = nn.BatchNorm1d(fc3_in_featrues)
        self.bn6 = nn.BatchNorm1d(fc4_in_features)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(fc1_in_features, fc2_in_features)
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(fc2_in_features, fc3_in_featrues)
        self.drop2 = nn.Dropout(drop_rate)
        self.fc3 = nn.Linear(fc3_in_featrues, fc4_in_features)
        self.drop3 = nn.Dropout(drop_rate)
        self.fc4 = nn.Linear(fc4_in_features, final_out_features)


    def forward(self, x):
        """
        Forward propagation
        :param x: tensor, a mini batch of input data
        :return:
        """
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(self.bn1(F.elu(self.conv1(x))), kernel_size=self.pool_kernel_size, padding=self.pool1_padding)
        x = F.max_pool3d(self.bn2(F.elu(self.conv2(x))), kernel_size=self.pool_kernel_size, padding=self.pool2_padding)
        x = F.max_pool3d(self.bn3(F.elu(self.conv3(x))), kernel_size=self.pool_kernel_size, padding=self.pool3_padding)

        # Flatten
        x = x.view(-1, self.num_flat_features(x))

        # drop out set 0.5
        x = self.drop1(self.bn4(F.elu(self.fc1(x))))
        x = self.drop2(self.bn5(F.elu(self.fc2(x))))
        x = self.drop3(self.bn6(F.elu(self.fc3(x))))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        """
        caculate num of data features
        :param x: ternsor, input data
        :return:
        """
        size = x.size()[1:]  # all demension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



def get_model(model_name):
    """
    This funtion will return module class according model_name.
    :param model_name: String, choosed model name.
    :return model: nn.Module.
    """
    model_dict = {
        'PkaNet': PkaNet(),
        'PkaNetBN1': PkaNetBN1(),
        'PkaNetBN2': PkaNetBN2(),
        'PkaNetBN3': PkaNetBN3(),
        'PkaNetBN2F18': PkaNetBN2F18(),
        'PkaNetBN2F18PRELU': PkaNetBN2F18PRELU(),
        'PkaNetBN2F18ELU': PkaNetBN2F18ELU(),
        'PkaNetBN2F19ELU': PkaNetBN2F19ELU(),
        'PkaNetBN2F11ELU': PkaNetBN2F11ELU(),
        'PkaNetBN3F18ELU': PkaNetBN3F18ELU(),
        'PkaNetBN3F19S21ELU': PkaNetBN3F19S21ELU(),
        'PkaNetBN4F18ELU': PkaNetBN4F18ELU(),
        'PkaNetBN4F19S21ELU': PkaNetBN4F19S21ELU(),
        'PkaNetBN4F19S17ELU': PkaNetBN4F19S17ELU(),
        'PkaNetBN4F19S19ELU': PkaNetBN4F19S19ELU(),
        'PkaNetBN4F19K96ELU': PkaNetBN4F19K96ELU(),

        'PkaNetF19S21RELU': PkaNetF19S21RELU(),
        'PkaNetF19S21PRELU': PkaNetF19S21PRELU(),
        'PkaNetF19S21ELU': PkaNetF19S21ELU(),
        'PkaNetBN4F19S21PRELU': PkaNetBN4F19S21PRELU(),
        'PkaNetBN4F18S21ELU': PkaNetBN4F18S21ELU(),
        'PkaNetBN4F20S21ELU': PkaNetBN4F20S21ELU(),         # we are using now!!! : DeepKa
        'PkaNetBN4F20S21RELU': PkaNetBN4F20S21RELU(),
        'PkaNetBN4F20S19ELU': PkaNetBN4F20S19ELU(),
        'PkaNetBN4F20S23ELU': PkaNetBN4F20S23ELU(),
        'PkaNetRachelELU': PkaNetRachelELU(),
        'PkaNetRachelActualRELU':PkaNetRachelActualRELU()
    }
    return model_dict[model_name]
