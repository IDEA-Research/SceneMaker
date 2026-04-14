import os
import torch
import matplotlib.pyplot as plt

import pdb
    
def viz_weight(weight, idx=0, save_path="outputs/viz/weight"):
    B, H, T, S = weight.shape
    if S == 769:
        weight = weight[:, :, :1, :]
        # split the last dimension into 2 parts 257,512
        weight_img, weight_pcd = torch.split(weight, [257, 512], dim=-1)
        # average on H dimension
        weight_img = weight_img.mean(dim=1)
        weight_pcd = weight_pcd.mean(dim=1)

        # sum on the last dimension
        weight_img = weight_img.mean(dim=-1)
        weight_pcd = weight_pcd.mean(dim=-1)

        # concat the 3 parts
        weight = torch.cat([weight_img, weight_pcd], dim=-1).reshape(-1, 5, 2)
        # viz as heatmap
        for i in range(weight.shape[0]):
            if i >= weight.shape[0]//2:
                plt.imshow(weight[i].clone().detach().cpu().float().numpy(), cmap='coolwarm', interpolation='nearest')
                plt.title(f'Rotation Attn Weight')
                plt.xticks(ticks=[0, 1], labels=['image', 'pcd'])
                plt.xlabel('Condition')
                plt.ylabel('Objects')
                plt.colorbar()
                plt.savefig(os.path.join(save_path, f'rotation_weight_b_{i}_t_{idx}.png'))
                plt.clf()
    
    elif S == 1028:
        weight = weight[:, :, :2, :]
        # split the last dimension into 4 parts 1,1,514,512
        weight_trans, weight_size, weight_img, weight_mask, weight_pcd = torch.split(weight, [1, 1, 257, 257, 512], dim=-1)
        # average on H dimension
        weight_trans = weight_trans.mean(dim=1)
        weight_size = weight_size.mean(dim=1)
        weight_img = weight_img.mean(dim=1)
        weight_mask = weight_mask.mean(dim=1)
        weight_pcd = weight_pcd.mean(dim=1)

        # sum on the last dimension
        weight_trans = weight_trans.mean(dim=-1)
        weight_size = weight_size.mean(dim=-1)
        weight_img = weight_img.mean(dim=-1)
        weight_mask = weight_mask.mean(dim=-1)
        weight_pcd = weight_pcd.mean(dim=-1)
        
        # concat the 5 parts
        weight = torch.stack([weight_trans, weight_size, weight_img, weight_mask, weight_pcd], dim=-1).reshape(-1, 5, 2, 5)
        trans_weight, size_weight = weight[:, :, 0, :], weight[:, :, 1, :]
        # viz as heatmap
        for i in range(weight.shape[0]):
            if i >= weight.shape[0]//2:
                # translation weight
                plt.imshow(trans_weight[i].clone().detach().cpu().float().numpy(), cmap='coolwarm', interpolation='nearest')
                plt.title(f'Translation Attn Weight')
                plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['translation', 'size', 'image', 'mask', 'pcd'])
                plt.xlabel('Condition')
                plt.ylabel('Objects')
                plt.colorbar()
                plt.savefig(os.path.join(save_path, f'translation_weight_b_{i}_t_{idx}.png'))
                plt.clf()
                
                # size weight
                plt.imshow(size_weight[i].clone().detach().cpu().float().numpy(), cmap='coolwarm', interpolation='nearest')
                plt.title(f'Size Attn Weight')
                plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['translation', 'size', 'image', 'mask', 'pcd'])
                plt.xlabel('Condition')
                plt.ylabel('Objects')
                plt.colorbar()
                plt.savefig(os.path.join(save_path, f'size_weight_b_{i}_t_{idx}.png'))
                plt.clf()
                
                
def viz_weight_caption(weight, idx=0, save_path="outputs/viz/weight"):
    B, H, T, S = weight.shape
    if T == 513:
        weight = weight[:, :, :1, :]
        # split the last dimension into 3 parts 257,512,77
        weight_img, weight_pcd, weight_text = torch.split(weight, [257, 512, 77], dim=-1)
        # average on H dimension
        weight_img = weight_img.mean(dim=1)
        weight_pcd = weight_pcd.mean(dim=1)
        weight_text = weight_text.mean(dim=1)
        # sum on the last dimension
        weight_img = weight_img.mean(dim=-1)
        weight_pcd = weight_pcd.mean(dim=-1)
        weight_text = weight_text.mean(dim=-1)
        # concat the 3 parts
        weight = torch.cat([weight_img, weight_pcd, weight_text], dim=-1).reshape(-1, 5, 3)
        # viz as heatmap
        for i in range(weight.shape[0]):
            if i >= weight.shape[0]//2:
                plt.imshow(weight[i].clone().detach().cpu().float().numpy(), cmap='coolwarm', interpolation='nearest')
                plt.title(f'Rotation Attn Weight')
                plt.xticks(ticks=[0, 1, 2], labels=['image', 'pcd', 'text'])
                plt.xlabel('Condition')
                plt.ylabel('Objects')
                plt.colorbar()
                plt.savefig(os.path.join(save_path, f'rotation_weight_b_{i}_t_{idx}.png'))
                plt.clf()
    
    elif T == 514:
        weight = weight[:, :, :2, :]
        # split the last dimension into 5 parts 1,1,514,512,77
        weight_trans, weight_size, weight_img, weight_pcd, weight_text = torch.split(weight, [1, 1, 514, 512, 77], dim=-1)
        # average on H dimension
        weight_trans = weight_trans.mean(dim=1)
        weight_size = weight_size.mean(dim=1)
        weight_img = weight_img.mean(dim=1)
        weight_pcd = weight_pcd.mean(dim=1)
        weight_text = weight_text.mean(dim=1)
        # sum on the last dimension
        weight_trans = weight_trans.mean(dim=-1)
        weight_size = weight_size.mean(dim=-1)
        weight_img = weight_img.mean(dim=-1)
        weight_pcd = weight_pcd.mean(dim=-1)
        weight_text = weight_text.mean(dim=-1)
        
        # concat the 5 parts
        weight = torch.stack([weight_trans, weight_size, weight_img, weight_pcd, weight_text], dim=-1).reshape(-1, 5, 2, 5)
        trans_weight, size_weight = weight[:, :, 0, :], weight[:, :, 1, :]
        # viz as heatmap
        for i in range(weight.shape[0]):
            if i >= weight.shape[0]//2:
                # translation weight
                plt.imshow(trans_weight[i].clone().detach().cpu().float().numpy(), cmap='coolwarm', interpolation='nearest')
                plt.title(f'Translation Attn Weight')
                plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['translation', 'size', 'image', 'pcd', 'text'])
                plt.xlabel('Condition')
                plt.ylabel('Objects')
                plt.colorbar()
                plt.savefig(os.path.join(save_path, f'translation_weight_b_{i}_t_{idx}.png'))
                plt.clf()
                
                # size weight
                plt.imshow(size_weight[i].clone().detach().cpu().float().numpy(), cmap='coolwarm', interpolation='nearest')
                plt.title(f'Size Attn Weight')
                plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['translation', 'size', 'image', 'pcd', 'text'])
                plt.xlabel('Condition')
                plt.ylabel('Objects')
                plt.colorbar()
                plt.savefig(os.path.join(save_path, f'size_weight_b_{i}_t_{idx}.png'))
                plt.clf()