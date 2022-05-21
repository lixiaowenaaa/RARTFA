import torch
import torchvision
from torch import nn
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
import argparse
import shutil
#import torch_dct as dct
from sklearn.metrics import confusion_matrix
import itertools
from symbols import resnet_cifar10, resnet_svhn

from toolbox.advertorch.attacks import LinfPGDAttack
from tqdm import tqdm

from utils import Visualizer
import sys
from toolbox.advertorch.attacks import LinfPGDAttack,GradientSignAttack,LinfBasicIterativeAttack,CarliniWagnerL2Attack
from toolbox.advertorch.bpda import BPDAWrapper

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def norm(t, p=2):
    assert len(t.shape) == 4
    if p == 2:
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
    elif p == 1:
        norm_vec = t.abs().sum(dim=[1, 2, 3]).view(-1, 1, 1, 1)
    else:
        raise NotImplementedError('Unknown norm p={}'.format(p))
    norm_vec += (norm_vec == 0).float() * 1e-8
    return norm_vec

def momentum_prior_step(x, g, lr):
    # adapted from Boosting Adversarial Attacks with Momentum, CVPR 2018
    return x + lr * g / norm(g, p=1)

def linf_proj_step(image, epsilon, adv_image):
    return image + torch.clamp(adv_image - image, -epsilon, epsilon)

def linf_image_step(x, g, lr):
    return x + lr * torch.sign(g)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for k in range(tensor.shape[0]):
            for t, m, s in zip(tensor[k], self.mean, self.std):
                t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class Normalize_(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for k in range(tensor.shape[0]):
            for t, m, s in zip(tensor[k], self.mean, self.std):
                t.sub_(m).div_(s)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def save_checkpoint(state, is_best, save_root, filename='checkpoint.pth.tar'):

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    filename = os.path.join(save_root,filename)
    print (filename)
    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, os.path.join(save_root,'model_best.pth.tar'))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():

        maxk = max(topk)

        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))


        res = []
        for k in topk:

            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

            res.append(correct_k.mul_(100.0 / batch_size))

        return res, pred

def clamp(input, min=None, max=None):
    if min is not None and max is not None:
        return torch.clamp(input, min=min, max=max)
    elif min is None and max is None:
        return input
    elif min is None and max is not None:
        return torch.clamp(input, max=max)
    elif min is not None and max is None:
        return torch.clamp(input, min=min)
    else:
        raise ValueError("This is impossible")

def _get_norm_batch(x, p):
    batch_size = x.size(0)
    return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)

def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    # TODO: move this function to utils

    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    # loss is averaged over the batch so need to multiply the batch
    # size to find the actual gradient of each input sample

    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return batch_multiply(1. / norm, x)


def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor.data[ii] *= vector[ii]
    return batch_tensor
    """
    return (
        batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()

def batch_clamp(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_clamp_tensor_by_vector(float_or_vector, tensor)
        return tensor
    elif isinstance(float_or_vector, float):
        tensor = clamp(tensor, -float_or_vector, float_or_vector)
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor

def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def pgd_attack(img, target, model, device, eps=.031):

    delta = torch.empty(img.shape).fill_(0).to(device)
    delta = nn.Parameter(delta)
    delta.requires_grad_()

    # mean=(0.4914, 0.4822, 0.4465)
    # std=(0.2023,  0.1994,  0.2010)
    # mean=(0.5, 0.5, 0.5)
    # std=(0.5,  0.5,  0.5)
    mean=(0., 0., 0.)
    std=(1,  1,  1)

    unnormalized_img = UnNormalize(mean, std)
    normalized_img = Normalize_(mean, std)
    crossentropy = nn.CrossEntropyLoss().to(device)

    for ii in range(10):
        
        output, c1, c2 = model(img + delta)

        init_pred = output.max(1, keepdim=True)[1]
        loss = crossentropy(output, target)
    
        model.zero_grad()
        loss.backward()

        # Attack setting
        grad_sign = torch.sign(normalize_by_pnorm(delta.grad.data, p=1))

        delta.data = delta.data + batch_multiply(0.01, grad_sign)

        delta.data = batch_clamp(eps, delta.data)

        img.data = unnormalized_img(img.data)

        delta.data = clamp(img.data + delta.data, 0, 1
                            ) - img.data
        delta.grad.data.zero_()

        img.data = normalized_img(img.data)

    img.data = unnormalized_img(img.data)

    x_adv = img.data + delta.data

    x_adv = normalized_img(x_adv)
    
    return x_adv

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PGD Defense model')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--quanti', type=float, default=0.3, metavar='N',
                        help='threshold of output z')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--hidden-size', type=int, default=20, metavar='N',
                        help='how big is z')
    parser.add_argument('--intermediate-size', type=int, default=128, metavar='N',
                        help='how big is linear around z')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')

    args = parser.parse_args()
    img_shape = (args.channels, args.img_size, args.img_size)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Configure data loader
    img_root = './data/imagenet/'

    # Configure model loader
    modelC_path = './models/cifar10/modelC/model_best.pth.tar'

    # Configure save root
    img_saveroot = './models/pure/imagenet/dc_img/'
    model_saveroot = './models/pure/imagenet/'


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modelC = resnet_cifar10.ResNet18().to(device)

    # ------------
    # Load model C
    #-------------
    if torch.cuda.device_count() > 1:
        modelC = nn.DataParallel(modelC)
    checkpoint = torch.load(modelC_path)['state_dict']
    modelC.load_state_dict(checkpoint, strict = False)
    
    # Loss function
    crossentropy = nn.CrossEntropyLoss().to(device)
    mseloss = nn.MSELoss().to(device)
    adversarial_loss = nn.L1Loss().to(device)
    tv = TVLoss().to(device)
    
    # Normalize dataset and dataloader
    normalize = transforms.Normalize((0,), (1,))

    #TODO Add more aug
    img_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    img_transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=img_root, split='train',
                                        download=True, transform=img_transform_test)                       
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=0)
    tenval_dataset = torchvision.datasets.ImageFolder(root=img_root, split='test',
                                       download=True, transform=img_transform_test)
    ten_val_loader = DataLoader(tenval_dataset, batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=0)
      
    
    #--------------------
    # Training  GAN
    #--------------------
    best_prec1 = 0
    best_epoch = 0
    if os.path.exists(img_saveroot):
        shutil.rmtree(img_saveroot)
    batch_time = AverageMeter()

    Tensor = torch.cuda.FloatTensor
    TensorLong = torch.cuda.LongTensor
    
    optimizerC = torch.optim.SGD(modelC.parameters(), lr=args.lr)
    
    visualizer = Visualizer(image_size=32)
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(args.epochs):

        end = time.time()
        Acc_A = AverageMeter()
        modelC.train()
        tbar = tqdm(train_loader)

        for i, (data, target) in enumerate(tbar):

            img_clean = data.to(device)
            target = target.to(device)  
            #adv_untargeted = pgd_attack(img_clean, target, modelC, device)

            # U,S,V = torch.svd(img_clean)
            
            # U,S,V = torch.svd(adv_untargeted)
            
            # S[:,:,5:] = 0
            
            # optimizerG.zero_grad()
            optimizerC.zero_grad()

            output, c1, c2 = modelC(img_clean)

            loss = crossentropy(output, target)
            # loss2 = crossentropy(output2, target)
            # loss = loss1 + loss2
            
            loss.backward()
            optimizerC.step()

            visualizer.show(img_clean.cpu().data, \
                c1.cpu().data, c2.cpu().data)

            tbar.set_description('\r[%d/%d][%d/%d] [loss_cln: %.3f] ' % \
                (epoch, args.epochs, i, len(train_loader), loss.item()))
        
        #------------------------
        # Evaluate the Generator
        #------------------------
        top0 = AverageMeter()
        modelC.eval()
        correct = 0
        data_len = 0
        print('\n')
        # with torch.no_grad():

        tbar = tqdm(ten_val_loader)

        for i, (data, target) in enumerate(tbar):
            data_len += data.size(0)
            data, target= data.to(device), target.to(device)
            
            #defense_withbpda = BPDAWrapper(modelC, forwardsub=lambda x: x)
            #defended_model = nn.Sequential(defense_withbpda)
            # bpda_adversary = LinfPGDAttack(
            #     modelC, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.031,
            #     nb_iter=10, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
            #     targeted=False)
            # adv_untargeted = bpda_adversary.perturb(data, target)  
            #adv_untargeted = pgd_attack(data, target, modelC, device)
           
            with torch.no_grad():
                #output0, c1, c2 = modelC(adv_untargeted)
                output0, c1, c2 = modelC(data)


            prec0, pred = accuracy(output0, target, topk=(1,))
            top0.update(prec0[0], data.size(0))
            tbar.set_description('\r[%d/%d][%d/%d] [Acc: %.3f]' % \
                (epoch, args.epochs, i, len(ten_val_loader), top0.avg[0].item()))

            if i == 0:
                y_target = target
                y_pred = pred
            else:
                y_target = torch.cat((y_target, target), dim=0)
                y_pred = torch.cat((y_pred, pred), dim=1)

        y_target = y_target.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        y_pred = np.squeeze(y_pred)
        cnf_matrix = confusion_matrix(y_target, y_pred)
        np.set_printoptions(precision=2)
        print('\n')
        print(cnf_matrix)
        
        total = 0
        print(data_len)
        for k in range(10):
            total += cnf_matrix[k][k]
        total = float(total) / float(data_len) *100
        print('Test: [{0}/{1}]\t'
            'Accuracy: {acc:.2f}%\t'
            .format(i+1, len(ten_val_loader), acc=total))
        
        # remember best prec@1 and save checkpoint
        is_best = top0.avg >= best_prec1
        best_prec1 = max(top0.avg, best_prec1)
        if is_best == 1:
            best_epoch = epoch
    
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': resnet_cifar10.ResNet18(),
            'state_dict': modelC.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizerC.state_dict(),
        } ,is_best, save_root=os.path.join(model_saveroot,'modelC'))

        print('[Best score: %.3f] [Best epoch %d]' %(best_prec1, best_epoch))

        # # Save image grid
        # imgs_lr = nn.functional.interpolate(adv_untargeted[:9], scale_factor=4)
        # gen_hr = nn.functional.interpolate(data.detach()[:9], scale_factor=4)
        # #gt = nn.functional.interpolate(data[:9], scale_factor=4)

        # gen_hr = make_grid(gen_hr, nrow=9, normalize=True)
        # imgs_lr = make_grid(imgs_lr, nrow=9, normalize=True)
        # #gt = make_grid(gt, nrow=9, normalize=True)

        # img_grid = torch.cat((imgs_lr, gen_hr), dim=1)

        # if not os.path.exists(img_saveroot):
        #     os.makedirs(img_saveroot)
        # save_image(img_grid, "%s/%d.png"% (img_saveroot, epoch))
