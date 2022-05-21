import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import os
import numpy as np

import torch_dct as dct
from sklearn.metrics import confusion_matrix
from symbols import resnet_cifar10, Rt_GAN, custom_data, data_load, vgg, resnet
from tqdm import tqdm

from toolbox.advertorch.attacks import L1PGDAttack,L2PGDAttack,LinfPGDAttack,GradientSignAttack,LinfBasicIterativeAttack,CarliniWagnerL2Attack
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
        
        #output,c1,c2 = model(img + delta)
        output = model(img + delta)

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

def test(modelsource, modelC, device, test_loader,eps=0.031):

    # Accuracy counter
    correct = 0
    adv_examples = []
    data_len = 0
    
    Tensor = torch.cuda.FloatTensor

    # Loop over all examples in test set
    #top2 = AverageMeter()

    # BPDA ATTACK 
    #defense_withbpda = BPDAWrapper(modelC)
    #defended_model = nn.Sequential(defense_withbpda)
    bpda_adversary = LinfPGDAttack(
         modelC, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
         nb_iter=10, eps_iter=0.031, rand_init=True, clip_min=0.0, clip_max=1.0,
         targeted=False)
    
    # FGSM attacks
    # fgsm_adversary = GradientSignAttack(
    #         modelsource, eps=8/255, 
    #         clip_min=0.0, clip_max=1.0,
    #         targeted=False)

    # BIM Attacks
    # bim_adversary = LinfBasicIterativeAttack(modelsource, \
    #     loss_fn=None, eps=8/255, \
    #     nb_iter=10, eps_iter=0.01, clip_min=0.0, clip_max=1.0, targeted=False)
    #top2 = AverageMeter()

    # CW Attacks
    # cw_adversary = CarliniWagnerL2Attack(modelsource, 10, \
    #     confidence=0, targeted=False, learning_rate=0.031, binary_search_steps=9,\
    #          max_iterations=10, \
    #     abort_early=True, initial_const=0.001, clip_min=0.0, clip_max=1.0, loss_fn=None)
    
    # adversary = LinfPGDAttack(
    #     modelsource, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.031,
    #     nb_iter=10, eps_iter=3/255, rand_init=True, clip_min=0.0, clip_max=1.0,
    #     targeted=False)

    top2 = AverageMeter()
    modelsource.eval()
    modelC.eval()

    tbar = tqdm(test_loader)
    

    for i, (data, target) in enumerate(tbar):

        data_len += data.size(0)
        data, target = data.to(device), target.to(device)
        #print(target)
        #adv_untargeted = pgd_attack(data, target, modelsource, device)

        #adv_untargeted = fgsm_adversary.perturb(data, target)
        # adv_untargeted = bim_adversary.perturb(data, target)
        #vadv_untargeted = cw_adversary.perturb(data, target)
        adv_untargeted = bpda_adversary.perturb(data, target)

        # output, gen0, gen1, gen2, x, out_feat = modelDefense(adv_untargeted)

        #x = modelDefense(adv_untargeted)

        output, c1, c2 = modelC(adv_untargeted)
        #output, c1, c2 = modelC(data)
        final_pred = torch.max(output.data,1)[1]
        #print(final_pred.shape) 
        #final_pred = output.max(1, keepdim=True)[1]

        final_pred = final_pred.t()
        #print(final_pred.shape)
        #print(target.shape)

        prec2, pred = accuracy(output, target, topk=(1,))

        top2.update(prec2[0], data.size(0))

        if i == 0:
            y_target = target
            y_pred = final_pred
        else:
            y_target = torch.cat((y_target, target), dim=0)
            y_pred = torch.cat((y_pred, final_pred), dim=0)

        tbar.set_description('\r[{:d}/{:d}] [Acc: {:.3f}]' .format \
                ( i, len(test_loader), top2.avg[0]))

    y_target = y_target.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_pred = np.squeeze(y_pred)
    cnf_matrix = confusion_matrix(y_target, y_pred)
    np.set_printoptions(precision=2)
    print(cnf_matrix)

    total = 0
    print(data_len)
    for k in range(10):
        total += cnf_matrix[k][k]
    total = float(total) / float(data_len) *100
    print('Test: [{0}/{1}]\t'
        'Accuracy: {acc:.2f}%\t'
        .format(i+1, len(val_loader), acc=total))
    
    # Save image grid
    imgs_lr = nn.functional.interpolate(adv_untargeted[:9], scale_factor=4)
    gen_hr = nn.functional.interpolate(data.detach()[:9], scale_factor=4)
    gt = nn.functional.interpolate(data[:9], scale_factor=4)

    gen_hr = make_grid(gen_hr, nrow=9, normalize=True)
    imgs_lr = make_grid(imgs_lr, nrow=9, normalize=True)
    gt = make_grid(gt, nrow=9, normalize=True)

    img_grid = torch.cat((gt, imgs_lr, gen_hr), dim=1)
 
    save_image(img_grid, "./testl1.png")

if __name__=="__main__":
    # Configure data loader
    img_root = './data/cifar10/'
    valdir = os.path.join(img_root, 'test')

    # Configure model loader
    #modelC_path = './models/cifar10/modelC/model_best.pth.tar'
    #modelG_path = './models/cifar10/modelG/model_best.pth.tar'

    # ------------
    # Load model C
    #-------------
    modelC_path = './models/pgd/cifar10/modelC/model_best.pth.tar'
    #modelC_path = './models/cifar10/attention_pure/modelC/model_best.pth.tar'
    modelC = resnet_cifar10.ResNet18()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        modelC = nn.DataParallel(modelC)
    checkpoint = torch.load(modelC_path)['state_dict']
    modelC.load_state_dict(checkpoint,False)
    modelC.to(device)
    writer = SummaryWriter(args.outf)

    # Baseline
    source_model = './models/resnet18/cifar10/model_best.pth.tar'
    #source_model = './models/vgg13/model_best.pth.tar'
    #modelsource = vgg.VGG13()
    modelsource = resnet.ResNet18()
    if torch.cuda.device_count() > 1:
        modelsource = nn.DataParallel(modelsource)
    check = torch.load(source_model)['state_dict']
    modelsource.load_state_dict(check,False)
    modelsource.to(device)


    normalize = transforms.Normalize((0,), (1,))
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])


    val_dataset = torchvision.datasets.CIFAR10(root=img_root, train=False,
                                       download=True, transform=img_transform)
    val_loader = DataLoader(val_dataset, batch_size=32,
                                shuffle=True,
                                num_workers=0)
    
    # val_dataset = data_load.ImageFolder(valdir, transform=img_transform)
    # val_loader = DataLoader(val_dataset, batch_size=256,
    #                             shuffle=True,
    #                             num_workers=0)

    test(modelsource, modelC, device, val_loader, eps=0.031)