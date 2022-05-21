'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_dct as dct
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter

def deadzone_torch(block):
    q = torch.zeros([block.shape[2], block.shape[3]],dtype=torch.int64).type(torch.cuda.FloatTensor) 
    qt_block = torch.where((block<-.2)|(block>.2),block,q)
    return torch.round(qt_block)

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

class ProjectorBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    
    def forward(self, inputs):
        return self.op(inputs)

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


class LinearWithChannel(nn.Module):
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()
        
        #initialize weights
        self.w = torch.nn.Parameter(torch.zeros(channel_size, input_size, output_size))
        self.b = torch.nn.Parameter(torch.zeros(1, channel_size, output_size))
        
        #change weights to kaiming
        self.reset_parameters(self.w, self.b)
        
    def reset_parameters(self, weights, bias):
        
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
    
    def forward(self, x):
        
        output = torch.bmm(x.transpose(0,1), self.w).transpose(0,1) + self.b
        return output


class self_correlation(nn.Module):

    def __init__(self, percentage, input_channel):
        super(self_correlation, self).__init__()
        self.percentage = percentage

        self.fc1 = LinearWithChannel(self.percentage, self.percentage, input_channel)
        self.hard = nn.ReLU6()

    def self_selected(self, c, x, topk):
        
        B, C, H, W = x.shape
        one_dim_size = H * W
        
        embedding = torch.zeros((B,C,topk)).cuda()
        x = x.reshape(B, C, one_dim_size)
        c = c.reshape(B, one_dim_size)

        top = torch.topk(c, k=topk)

        selected = c - top[0][:,-1].unsqueeze(1)
        selected = (torch.floor(selected))
        selected = F.relu(selected+0.5)*2
        
        indexes = top[1]
        
        weights = batched_index_select(selected, 1, indexes)
        img = batched_index_select(x, 2, indexes)
        
        # Select topk points
        embedding[:,:,:topk] = weights.unsqueeze(1) * img
        
        # select K points
        # _,Cv,_,_ = v3.shape
        # v3 = F.softmax(v3,dim=1).view(B,Cv,one_dim_size)
        # v3t = v3.transpose(1,2)
        # out = torch.matmul(v3t, v3)
        # mask = torch.zeros((B, topk, topk)).cuda()
        # temp = batched_index_select(out, 1, indexes)
        # mask[:,:,:topk] = batched_index_select(temp, 2, indexes)

        # embedding = torch.bmm(embedding, mask)
        
        embedding = self.fc1(embedding)
        v3t = embedding.transpose(1,2)
        out = torch.matmul(v3t, embedding)
        
        # embedding = self.fc2(embedding)
        
        # embedding = torch.zeros((B,C,topk)).cuda()

        # out = torch.zeros((B,C,one_dim_size)).cuda()
        
        # for batch in range(B):
        #     out[batch,:,indexes[batch]] = embedding[batch]
        
        # out[:,:,:topk] = embedding
        
        return out

    def forward(self, im, weight):

        B, C, H, W = im.shape

        embedding = self.self_selected(weight, im, self.percentage)
        
        # embedding = embedding.reshape(B,C,H,W)
        
        return embedding

class LinearAttentionBlock(nn.Module):
    
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.attn1 = LinearAttentionBlock(512)
        self.attn2 = LinearAttentionBlock(512)
        self.attn3 = LinearAttentionBlock(512)

        self.proj1 = ProjectorBlock(64, 512)
        self.proj2 = ProjectorBlock(128, 512)
        self.proj3 = ProjectorBlock(256, 512)

        self.corr1 = self_correlation(50, 64)
        self.corr2 = self_correlation(50, 128)
        self.corr3 = self_correlation(50, 256)

        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding=0, bias=True)

        # self.A = nn.Linear(448*20, 512*3)

        self.classify = nn.Linear(7500, num_classes)#20-1200,10-300,30-2700,40-4800,50-7500

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for m in range(0, 32, 4):
            for j in range(0, 32, 4):
                block = out[:,:,m:m+4, j:j+4]
                dct_matrix = dct.dct_2d(block)
                dequant_matrix = deadzone_torch(dct_matrix)
                block = dct.idct_2d(dequant_matrix)
                out[:,:,m:m+4, j:j+4] = block  # 64*32*32

        l1 = self.layer1(out) # 64*32
        
        l2 = self.layer2(l1) # 128*16
        
        l3 = self.layer3(l2) # 256*8

        l4 = self.layer4(l3) # 512*4

        g = self.dense(l4)

        c1, g1 = self.attn1(self.proj1(l1), g)
        out1 = self.corr1(l1, c1)
        
        c2, g2 = self.attn2(self.proj2(l2), g)
        out2 = self.corr2(l2, c2)

        c3, g3 = self.attn3(self.proj3(l3), g)
        out3 = self.corr3(l3, c3)
        
        g = torch.cat((out1, out2, out3), dim=1)
        # g = out1 + out2 + out3
        
        # out = F.avg_pool2d(l4, 4)
        g = g.view(g.size(0), -1)
        
        out = self.classify(g)

        # writer = SummaryWriter("my_experiment")
        # atten_l1 = make_grid(l1.detach().cpu().unsqueeze(dim=1), nrow=6, padding=20, normalize=False, pad_value=1)
        # writer.add_image('test/l1', atten_l1)

        return out, c1, c2


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
