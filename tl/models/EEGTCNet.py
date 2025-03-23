import torch
import torch.nn as nn

class EEGNetFeature(nn.Module):
    def __init__(self, Chans, Samples, kernLength, F1, D, F2, dropoutRate):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((kernLength//2-1, kernLength - kernLength//2, 0, 0)),
            nn.Conv2d(1, F1, (1, kernLength), stride=1, bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1*D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        
        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F1*D, F1*D, (1, 16), stride=1, groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, (1, 1), stride=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x.squeeze(2)  # [batch, F2, time_steps]

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout, activation):
        super().__init__()
        pad = (kernel_size-1) * dilation
        self.activation = activation
        self.conv = nn.Sequential(
            nn.ConstantPad1d((pad, 0), 0),
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            self.activation(),
            nn.Dropout(dropout),

            nn.ConstantPad1d((pad, 0), 0),
            nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            self.activation(),
            nn.Dropout(dropout)
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.activation()(self.conv(x) + self.downsample(x))

class TCN(nn.Module):
    def __init__(self, input_dim, filters, kernel_size, layers, dropout, activation):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(layers):
            dilation = 2**i
            in_ch = input_dim if i == 0 else filters
            self.blocks.append(TCNBlock(in_ch, filters, kernel_size, dilation, dropout, activation))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class EEGTCNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, layers=3, kernel_s=10, filt=10, 
                 dropout=0.3, F1=4, D=2, kernLength=64, dropout_eeg=0.1, activation='elu'):
        super().__init__()
        self.F2 = F1 * D
        
        self.eegnet = EEGNetFeature(
            Chans=Chans,
            Samples=Samples,
            kernLength=kernLength,
            F1=F1,
            D=D,
            F2=self.F2,
            dropoutRate=dropout_eeg
        )
        
        self.tcn = TCN(
            input_dim=self.F2,
            filters=filt,
            kernel_size=kernel_s,
            layers=layers,
            dropout=dropout,
            activation=nn.ELU if activation=='elu' else nn.ReLU
        )
        
        self.fc = nn.Linear(filt, nb_classes)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)  # [batch, Samples, Chans, 1]
        x = self.eegnet(x)         # [batch, F2, time_steps]
        x = self.tcn(x)           # [batch, filt, time_steps]
        x = x[:, :, -1]           # [batch, filt]
        return self.fc(x)
    
class EEGTCNet_feature(nn.Module):
    def __init__(self, n_classes, Chans=64, Samples=128, kernLenght=64, F1=4, D=2, dropout_eeg=0.1, 
                 filt=10, kernel_s=10, layers=3, dropout=0.3, activation='elu'):
        super().__init__()
        self.F2 = F1 * D
        
        self.eegnet = EEGNetFeature(
            Chans=Chans,
            Samples=Samples,
            kernLength=kernLenght,
            F1=F1,
            D=D,
            F2=self.F2,
            dropoutRate=dropout_eeg
        )
        
        self.tcn = TCN(
            input_dim=self.F2,
            filters=filt,
            kernel_size=kernel_s,
            layers=layers,
            dropout=dropout,
            activation=nn.ELU if activation=='elu' else nn.ReLU
        )
        
    def forward(self, x):
        x = self.eegnet(x)         # [batch, F2, time_steps]
        x = self.tcn(x)           # [batch, filt, time_steps]
        output = x.reshape(x.size(0), -1)  # [batch, filt]
        return output