import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils

def visualize_attn_softmax(I, c, up_factor, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # print(img.size)
    # compute the heatmap
    N,C,W,H = c.size()
    #print(c.shape)
    a = F.softmax(c.view(N,C,-1), dim=2).view(N,C,W,H)
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    #print(a.shape)
    attn = utils.make_grid(a, nrow=nrow, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    # atten = attn.view(N,C,-1)
    # top_k_idx=atten.argsort()[::-1][0:20]
    # attn_value = min(atten[top_k_idx])
    # b = attn.copy()
    # b[b<attn_value] = 0
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255

    b = a.reshape(N, 32*32)  
    top = torch.topk(b, k=80)
    attn_value = top[0][:,-1].unsqueeze(1)
    b[b<attn_value] = 0
    attn2 = utils.make_grid(b.view(N,C,32,32), nrow=nrow, normalize=True, scale_each=True)
    attn2 = attn2.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn2  = cv2.applyColorMap(attn2 , cv2.COLORMAP_JET)
    attn2  = cv2.cvtColor(attn2 , cv2.COLOR_BGR2RGB)
    attn2  = np.float32(attn2 ) / 255
    # add the heatmap to the image
    vis1 = 0.6*img + 0.4*attn
    vis2 = 0.6*img + 0.4*attn2 

    return torch.from_numpy(vis1).permute(2,0,1), torch.from_numpy(vis2).permute(2,0,1)

def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))

def visualize_attn_sigmoid(I, c, up_factor, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    a = torch.sigmoid(c)
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=nrow, normalize=False)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return torch.from_numpy(vis).permute(2,0,1)
