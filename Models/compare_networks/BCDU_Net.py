from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, step=3, effective_step=[2]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs[0]



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout = 0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout = 0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_my(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels):
        super().__init__()
        
        self.up_my = nn.Sequential(
            nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2), 
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x1, x2):
        x1 = self.up_my(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        
        return x    
        

    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels, 16, kernel_size=1),
                        nn.Conv2d(16, out_channels, kernel_size=1)
                    )
                    
    def forward(self, x):
        return self.conv(x)


class BCDU_net_D3(nn.Module):
    def __init__(self, classes, channels):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(BCDU_net_D3, self).__init__()
        self.out_size = (224, 320)
        
        self.inc = DoubleConv(channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256, dropout = 0.3)
        self.down3 = Down(256, 512, dropout = 0.3)
        self.flat1 = DoubleConv(512, 512, dropout = 0.3)
        self.flat2 = DoubleConv(512*2, 512, dropout = 0.3)

        self.up3 = Up_my(512)
        self.convlstm3 = ConvLSTM(input_channels=256*2, hidden_channels=[128, 128, 256]).cuda()
        self.dconv3 = DoubleConv(256, 256)
        
        self.up2 = Up_my(256)
        self.convlstm2 = ConvLSTM(input_channels=128*2, hidden_channels=[64, 64, 128]).cuda()
        self.dconv2 = DoubleConv(128, 128)
        
        self.up1 = Up_my(128)
        self.convlstm1 = ConvLSTM(input_channels=64*2, hidden_channels=[32, 32, 64]).cuda()
        self.dconv1 = DoubleConv(64, 64)
        
        self.outc = OutConv(64, classes)
        initialize_weights(self)
        

    def forward(self, x):
        x1 = self.inc(x)        # [12, 3, 224, 320]  --> [12, 64, 224, 320]
        x2 = self.down1(x1)     # [12, 128, 112, 160]
        x3 = self.down2(x2)     # [12, 256, 56, 80]
        x4 = self.down3(x3)     # [12, 512, 28, 40]
        
        x5 = self.flat1(x4)     # [12, 512, 28, 40]
        x5_dense = torch.cat([x4, x5], dim=1)   # [12, 1024, 28, 40]
        x6 = self.flat2(x5_dense)               # [12, 512, 28, 40]
        
        x7 = self.up3(x6, x3)                   # [12, 512, 56, 80]
        x8 = self.convlstm3(x7)                 # 
        x9 = self.dconv3(x8)
        
        x10 = self.up2(x9, x2)
        x11 = self.convlstm2(x10)
        x12 = self.dconv2(x11)

        x13 = self.up1(x12, x1)
        x14 = self.convlstm1(x13)
        x15 = self.dconv1(x14)

        out = self.outc(x15)
        final = F.sigmoid(out)
        
        return final        

