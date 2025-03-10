import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())


class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.leaky_relu(Y, 0.2)


class Encoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            
            Residual(64, 64),
            Residual(64, 64),
            
            Residual(64, 128, use_1x1conv=True, strides=2),
            Residual(128, 128),
            
            Residual(128, 256, use_1x1conv=True, strides=2),
            Residual(256, 256),
            
            Residual(256, 512, use_1x1conv=True, strides=2),
            Residual(512, 512)
        )
    
    def forward(self, x):
        return self.net(x)

class Encoder2(nn.Module):
    def __init__(self, use_padding=False):
        super().__init__()
        self.use_padding = use_padding
        padding = int(use_padding)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=padding), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=padding), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=padding),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=padding),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=padding),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=padding),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=padding),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=padding),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=padding),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            )
    
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(F.max_pool2d(x1, 2))
        x3 = self.layer3(F.max_pool2d(x2, 2))
        x4 = self.layer4(F.max_pool2d(x3, 2))
        y =  F.max_pool2d(x4, 2)
        return (y, x1, x2, x3, x4)


class Encoder3(nn.Module):
    def __init__(self, use_padding=False):
        super().__init__()
        self.use_padding = use_padding
        padding = int(use_padding)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=padding), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=padding), nn.ReLU(),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=padding), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=padding), nn.ReLU(),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=padding), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=padding), nn.ReLU(),
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=padding), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=padding), nn.ReLU(),
            )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=padding), nn.ReLU(),
            )
    
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(F.max_pool2d(x1, 2))
        x3 = self.layer3(F.max_pool2d(x2, 2))
        x4 = self.layer4(F.max_pool2d(x3, 2))
        y =  F.max_pool2d(x4, 2)
        return (y, x1, x2, x3, x4)




class Decoder2(nn.Module):
    def __init__(self, use_padding=False):
        super().__init__()
        self.use_padding = use_padding
        padding = int(use_padding)
        self.layer5 = nn.Sequential(
            nn.Conv2d(512*3, 1024, 3, padding=padding),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, 3, padding=padding), 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            )
        self.up_conv4 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Sequential(
            nn.Conv2d(512*4, 512, 3, padding=padding), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=padding), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            )
        self.up_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Sequential(
            nn.Conv2d(256*4, 256, 3, padding=padding), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=padding), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            )
        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(128*4, 128, 3, padding=padding), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=padding), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            )
        self.up_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64*4, 64, 3, padding=padding), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=padding), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 1)
            )
        
    def forward(self, y1, y2, y3):
        E4, E3, E2, E1 = 4, 16, 41, 90
        if self.use_padding:
            x4 = self.up_conv4(self.layer5(torch.cat((y1[0], y2[0], y3[0]), dim=1)))
            x3 = self.up_conv3(self.layer4(torch.cat((y1[4], y2[4], y3[4], x4), dim=1)))
            x2 = self.up_conv2(self.layer3(torch.cat((y1[3], y2[3], y3[3], x3), dim=1)))
            x1 = self.up_conv1(self.layer2(torch.cat((y1[2], y2[2], y3[2], x2), dim=1)))
            y = self.layer1(torch.cat((y1[1], y2[1], y3[1], x1), dim=1))
        else:
            x4 = self.up_conv4(self.layer5(torch.cat((y1[0], y2[0], y3[0]), dim=1)))
            x3 = self.up_conv3(self.layer4(torch.cat((y1[4][:, :, E4:-E4, E4:-E4], y2[4][:, :, E4:-E4, E4:-E4], y3[4][:, :, E4:-E4, E4:-E4], x4), dim=1)))
            x2 = self.up_conv2(self.layer3(torch.cat((y1[3][:, :, E3:-E3-1, E3:-E3-1], y2[3][:, :, E3:-E3-1, E3:-E3-1], y3[3][:, :, E3:-E3-1, E3:-E3-1], x3), dim=1)))
            x1 = self.up_conv1(self.layer2(torch.cat((y1[2][:, :, E2:-E2, E2:-E2], y2[2][:, :, E2:-E2, E2:-E2], y3[2][:, :, E2:-E2, E2:-E2], x2), dim=1)))
            y = self.layer1(torch.cat((y1[1][:, :, E1:-E1, E1:-E1], y2[1][:, :, E1:-E1, E1:-E1], y3[1][:, :, E1:-E1, E1:-E1], x1), dim=1))
        return y


class DecoderCT(nn.Module):
    def __init__(self, use_padding=False):
        super().__init__()
        self.use_padding = use_padding
        padding = int(use_padding)
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=padding),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, 3, padding=padding), 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            )
        self.up_conv4 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Sequential(
            nn.Conv2d(512*2, 512, 3, padding=padding), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=padding), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            )
        self.up_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Sequential(
            nn.Conv2d(256*2, 256, 3, padding=padding), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=padding), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            )
        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(128*2, 128, 3, padding=padding), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=padding), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            )
        self.up_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64*2, 64, 3, padding=padding), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=padding), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 1)
            )
        
    def forward(self, y1):
        E4, E3, E2, E1 = 4, 16, 41, 90
        if self.use_padding:
            x4 = self.up_conv4(self.layer5(torch.cat((y1[0],), dim=1)))
            x3 = self.up_conv3(self.layer4(torch.cat((y1[4], x4), dim=1)))
            x2 = self.up_conv2(self.layer3(torch.cat((y1[3], x3), dim=1)))
            x1 = self.up_conv1(self.layer2(torch.cat((y1[2], x2), dim=1)))
            y = self.layer1(torch.cat((y1[1], x1), dim=1))
        else:
            x4 = self.up_conv4(self.layer5(torch.cat((y1[0],), dim=1)))
            x3 = self.up_conv3(self.layer4(torch.cat((y1[4][:, :, E4:-E4, E4:-E4], x4), dim=1)))
            x2 = self.up_conv2(self.layer3(torch.cat((y1[3][:, :, E3:-E3-1, E3:-E3-1], x3), dim=1)))
            x1 = self.up_conv1(self.layer2(torch.cat((y1[2][:, :, E2:-E2, E2:-E2], x2), dim=1)))
            y = self.layer1(torch.cat((y1[1][:, :, E1:-E1, E1:-E1], x1), dim=1))
        return y
        
        
class Decoder3(nn.Module):
    def __init__(self, use_padding=False):
        super().__init__()
        self.use_padding = use_padding
        padding = int(use_padding)
        self.layer5 = nn.Sequential(
            nn.Conv2d(512*3, 1024, 3, padding=padding), nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=padding), nn.ReLU(),
            )
        self.up_conv4 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Sequential(
            nn.Conv2d(512*4, 512, 3, padding=padding), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=padding), nn.ReLU(),
            )
        self.up_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Sequential(
            nn.Conv2d(256*4, 256, 3, padding=padding), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=padding), nn.ReLU(),
            )
        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(128*4, 128, 3, padding=padding), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=padding), nn.ReLU(),
            )
        self.up_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64*4, 64, 3, padding=padding), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=padding), nn.ReLU(),
            nn.Conv2d(64, 1, 1)
            )
        
    def forward(self, y1, y2, y3):
        E4, E3, E2, E1 = 4, 16, 41, 90
        if self.use_padding:
            x4 = self.up_conv4(self.layer5(torch.cat((y1[0], y2[0], y3[0]), dim=1)))
            x3 = self.up_conv3(self.layer4(torch.cat((y1[4], y2[4], y3[4], x4), dim=1)))
            x2 = self.up_conv2(self.layer3(torch.cat((y1[3], y2[3], y3[3], x3), dim=1)))
            x1 = self.up_conv1(self.layer2(torch.cat((y1[2], y2[2], y3[2], x2), dim=1)))
            y = self.layer1(torch.cat((y1[1], y2[1], y3[1], x1), dim=1))
        else:
            x4 = self.up_conv4(self.layer5(torch.cat((y1[0], y2[0], y3[0]), dim=1)))
            x3 = self.up_conv3(self.layer4(torch.cat((y1[4][:, :, E4:-E4, E4:-E4], y2[4][:, :, E4:-E4, E4:-E4], y3[4][:, :, E4:-E4, E4:-E4], x4), dim=1)))
            x2 = self.up_conv2(self.layer3(torch.cat((y1[3][:, :, E3:-E3-1, E3:-E3-1], y2[3][:, :, E3:-E3-1, E3:-E3-1], y3[3][:, :, E3:-E3-1, E3:-E3-1], x3), dim=1)))
            x1 = self.up_conv1(self.layer2(torch.cat((y1[2][:, :, E2:-E2, E2:-E2], y2[2][:, :, E2:-E2, E2:-E2], y3[2][:, :, E2:-E2, E2:-E2], x2), dim=1)))
            y = self.layer1(torch.cat((y1[1][:, :, E1:-E1, E1:-E1], y2[1][:, :, E1:-E1, E1:-E1], y3[1][:, :, E1:-E1, E1:-E1], x1), dim=1))
        return y
        
        

class UNet1(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = Encoder1
        self.encoder1 = encoder()
        self.encoder2 = encoder()
        self.encoder3 = encoder()
        self.decoder = nn.Sequential(
            nn.Conv2d(512*3, 512*3, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(512*3, 512*3, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512*3, 512*3, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512*3, 512*3, kernel_size=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512*3, 256*3, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(256*3, 128*3, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128*3, 64*3, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64*3, 3, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(3, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, ct, pet, mri):
        ct = self.encoder1(ct)
        pet = self.encoder2(pet)
        mri = self.encoder3(mri)
        x = torch.cat((ct, pet, mri), dim=1)
        x = self.decoder(x)
        return x


class UNet2(nn.Module):
    def __init__(self, use_padding=False):
        super().__init__()
        encoder = Encoder2
        decoder = Decoder2
        self.use_padding = use_padding
        self.encoder1 = encoder(use_padding=use_padding)
        self.encoder2 = encoder(use_padding=use_padding)
        self.encoder3 = encoder(use_padding=use_padding)
        self.decoder = decoder(use_padding=use_padding)
        
    def forward(self, ct, pet, mri, use_ct=True, use_pet=True, use_mri=True):
        ref_encoding = self.encoder1(ct)
        zero_encoding = tuple(torch.zeros_like(tensor) for tensor in ref_encoding)
        ct = self.encoder1(ct) if use_ct else zero_encoding
        pet = self.encoder2(pet) if use_pet else zero_encoding
        mri = self.encoder3(mri) if use_mri else zero_encoding
        x = self.decoder(ct, pet, mri)
        return x


class ResiPipe(nn.Module):
    def __init__(self):
        super().__init__()
        self.resi1 = nn.Sequential(*[Residual(64, 64) for _ in range(4)])
        self.resi2 = nn.Sequential(*[Residual(128, 128) for _ in range(3)])
        self.resi3 = nn.Sequential(*[Residual(256, 256) for _ in range(2)])
        self.resi4 = nn.Sequential(*[Residual(512, 512) for _ in range(1)])
    
    def forward(self, x, use_x=True):
        x4 = self.resi4(x[4])
        x3 = self.resi3(x[3])
        x2 = self.resi2(x[2])
        x1 = self.resi1(x[1])
        y = x[0]
        return (y, x1, x2, x3, x4)


class MultiResUNet(nn.Module):
    def __init__(self, use_padding=False):
        super().__init__()
        encoder = Encoder2
        decoder = Decoder2
        self.use_padding = use_padding
        self.encoder1 = encoder(use_padding=use_padding)
        self.encoder2 = encoder(use_padding=use_padding)
        self.encoder3 = encoder(use_padding=use_padding)
        self.resi1 = ResiPipe()
        self.resi2 = ResiPipe()
        self.resi3 = ResiPipe()
        self.decoder = decoder(use_padding=use_padding)
        
    def forward(self, ct, pet, mri, use_ct=True, use_pet=True, use_mri=True):
        ref_encoding = self.encoder1(ct)
        zero_encoding = tuple(torch.zeros_like(tensor) for tensor in ref_encoding)
        ct = self.encoder1(ct) if use_ct else zero_encoding
        pet = self.encoder2(pet) if use_pet else zero_encoding
        mri = self.encoder3(mri) if use_mri else zero_encoding
        ct = self.resi1(ct) if use_ct else zero_encoding
        pet = self.resi2(pet) if use_pet else zero_encoding
        mri = self.resi3(mri) if use_mri else zero_encoding
        x = self.decoder(ct, pet, mri)
        return x


class UNet3(nn.Module):
    def __init__(self, use_padding=False):
        super().__init__()
        encoder = Encoder3
        decoder = Decoder3
        self.use_padding = use_padding
        self.encoder1 = encoder(use_padding)
        self.encoder2 = encoder(use_padding)
        self.encoder3 = encoder(use_padding)
        self.decoder = decoder(use_padding)
        
    def forward(self, ct, pet, mri, use_ct=True, use_pet=True, use_mri=True):
        ref_encoding = self.encoder1(ct)
        zero_encoding = tuple(torch.zeros_like(tensor) for tensor in ref_encoding)
        ct = self.encoder1(ct) if use_ct else zero_encoding
        pet = self.encoder2(pet) if use_pet else zero_encoding
        mri = self.encoder3(mri) if use_mri else zero_encoding
        x = self.decoder(ct, pet, mri)
        return x
    

class UNetCT(nn.Module):
    def __init__(self, use_padding=False):
        super().__init__()
        encoder = Encoder2
        decoder = DecoderCT
        self.use_padding = use_padding
        self.encoder = encoder(use_padding)
        self.decoder = decoder(use_padding)
        
    def forward(self, ct, pet, mri, use_ct=True, use_pet=True, use_mri=True):
        ref_encoding = self.encoder1(ct)
        zero_encoding = tuple(torch.zeros_like(tensor) for tensor in ref_encoding)
        ct = self.encoder1(ct) if use_ct else zero_encoding
        x = self.decoder(ct)
        return x    


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)


    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi



class Type2Model(nn.Module):
    def initialize_weights(self):
        # He 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                
    def __init__(self, MRI='T1', unet=UNet2, use_padding=True):
        super().__init__()
        self.MRI = MRI
        self.use_padding = use_padding
        self.net = unet(use_padding=use_padding)
        self.initialize_weights()
        self.use_ct = True
        self.use_pet = True
        self.use_mri = True
    
    def forward(self, x):
        if self.MRI == 'T1':
            return self.net(x[0], x[1], x[2], use_ct=self.use_ct, use_pet=self.use_pet, use_mri=self.use_mri)
        elif self.MRI == 'T2':
            return self.net(x[0], x[1], x[3], use_ct=self.use_ct, use_pet=self.use_pet, use_mri=self.use_mri)
        else:
            raise ValueError(f"Unsupported MRI type: {self.MRI}")
        
        
        
    