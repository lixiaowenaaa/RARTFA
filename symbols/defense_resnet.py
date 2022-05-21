'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_dct as dct



# def dct(x, norm=None):
        
        
#     x_shape = x.shape
#     N = x_shape[-1]
#     x = x.contiguous().view(-1, N)

#     v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

#     Vc = torch.rfft(v, 1, onesided=False)

#     k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
#     W_r = torch.cos(k)
#     W_i = torch.sin(k)

#     V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

#     if norm == 'ortho':
#         V[:, 0] /= np.sqrt(N) * 2
#         V[:, 1:] /= np.sqrt(N / 2) * 2

#     V = 2 * V.view(*x_shape)


#     return V

class RandomTransferLayer(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(RandomTransferLayer, self).__init__()


    def dct(self,x,norm=None):
        
        
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
        
        Vc = torch.rfft(v, 1, onesided=False)
        print(Vc.shape)

        # k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        
        # W_r = torch.cos(k)
        # W_i = torch.sin(k)

        # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        # if norm == 'ortho':
        #     V[:, 0] /= np.sqrt(N) * 2
        #     V[:, 1:] /= np.sqrt(N / 2) * 2

        # V = 2 * V.view(*x_shape)

        return Vc

    def idct(self,X,norm=None):
        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        v = torch.irfft(V, 1, onesided=False)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape)

    def dct_2d(self,x):

        x1 = self.dct(x)
        # x2 = self.dct(x1.transpose(-1, -2))

        # return x2.transpose(-1, -2)
        return x1
    
    def idct_2d(self,X,norm=None):
    
        x1 = self.idct(X, norm=norm)
        x2 = self.idct(x1.transpose(-1, -2), norm=norm)

        return x2.transpose(-1, -2)

    def dct_3d(self, x, norm=None):
        X1 = self.dct(x, norm=norm)
        X2 = self.dct(X1.transpose(-1, -2), norm=norm)
        X3 = self.dct(X2.transpose(-1, -3), norm=norm)
        return X3.transpose(-1, -3).transpose(-1, -2)


    def idct_3d(self, X, norm=None):
        
        x1 = self.idct(X, norm=norm)
        x2 = self.idct(x1.transpose(-1, -2), norm=norm)
        x3 = self.idct(x2.transpose(-1, -3), norm=norm)
        return x3.transpose(-1, -3).transpose(-1, -2)

    def forward(self, x):

        x = self.dct_2d(x)
        
        # x[:,:,:,12:] = 0
        # x[:,:,12:,:] = 0
        
        # x = self.idct_2d(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.randTrans = RandomTransferLayer()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.convTrans1 = nn.Sequential(*self._transpose_layer(512, 256))
        self.convTrans2 = nn.Sequential(*self._transpose_layer(256, 128))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _transpose_layer(self, in_filter, out_filter):
        upsampling = []
        upsampling += [
            nn.Conv2d(in_filter, out_filter, 3, 1, 1),
            nn.BatchNorm2d(out_filter), 
            nn.ConvTranspose2d(out_filter,out_filter,4,2,1),
            nn.PReLU(),
        ]
        return upsampling
        

    def forward(self, x):
        
        conv1 = F.relu(self.bn1(self.conv1(x)))
        layer1 = self.layer1(conv1)   # size = 64*32*32
        

        for i in range(3):

            if i == 0:
                layer2 = self.layer2(layer1)   # size = 128*16*16
                layer3 = self.layer3(layer2)   # size = 256*8*8
                layer4 = self.layer4(layer3)   # size = 512*4*4

                up_1 = self.convTrans1(layer4)  # size = 256*8*8
                up_2 = self.convTrans2(up_1)    # size = 128*16*16
            
            else:
                layer_add = torch.add(layer2, up_2)
                layer3 = self.layer3(layer_add)   # size = 256*8*8
                layer4 = self.layer4(layer3)   # size = 512*4*4

                up_1 = self.convTrans1(layer4)  # size = 256*8*8
                up_2 = self.convTrans2(up_1)    # size = 128*16*16
        

        # conv1_ = F.relu(self.bn1(self.conv1(x)))
        # layer1_ = self.layer1(conv1_)   # size = 32*32
        # layer2_ = self.layer2(layer1_)   # size = 16*16
        # layer3_ = self.layer3(layer2_)   # size = 8*8
        # layer4_ = self.layer4(layer3_)   # size = 4*4

        # concat = torch.add(layer4, layer4_)

        
        pool1 = F.avg_pool2d(layer4, 4)   # size = 1*1
        out = pool1.view(pool1.size(0), -1)
        out = self.linear(out)
        
        return out, layer4


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])
