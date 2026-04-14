import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data
from torchvision import transforms
import pickle
from PIL import Image
import numpy as np
from configs.data_config import Relation_Config
import math
import collections
import glob
import json
import random
from net_utils.bins import *
from scipy import io
from tqdm import tqdm
import cv2


HEIGHT_PATCH = 256
WIDTH_PATCH = 256
rel_cfg = Relation_Config()
d_model = int(rel_cfg.d_g/4)

data_transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
pil2tensor = transforms.ToTensor()

class Front_Det_Dataset(Dataset):
    def __init__(self, config, mode):
        super(Front_Det_Dataset, self).__init__()
        self.mode=mode
        self.config=config
        if mode=="train":
            self.data_path=os.path.join(config['data']['data_path'],'train')
        elif mode=="test":
            self.data_path=os.path.join(config['data']['data_path'],'test')
        self.split=glob.glob(self.data_path+"/*.pkl")

    def __len__(self):
        return len(self.split)

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    def __getitem__(self, index):

        file_path = self.split[index]
        with open(file_path, 'rb') as f:
            sequence = pickle.load(f)
        image = Image.fromarray(sequence['rgb_img'])
        width,height=image.size
        depth = Image.fromarray(sequence['depth_map'])
        try:
            camera = sequence['camera']
            #print(camera["K"])
        except:
            print(sequence['sequence_id'])
        boxes = sequence['boxes']
        layout=sequence['layout']
        # build relational geometric features for each object
        n_objects = boxes['bdb2D_pos'].shape[0]
        # g_feature: n_objects x n_objects x 4
        # Note that g_feature is not symmetric,
        # g_feature[m, n] is the feature of object m contributes to object n.
        #print(boxes['bdb2D_pos'])
        g_feature = [[((loc2[0] + loc2[2]) / 2. - (loc1[0] + loc1[2]) / 2.) / (loc1[2] - loc1[0]),
                      ((loc2[1] + loc2[3]) / 2. - (loc1[1] + loc1[3]) / 2.) / (loc1[3] - loc1[1]),
                      math.log((loc2[2] - loc2[0]) / (loc1[2] - loc1[0])),
                      math.log((loc2[3] - loc2[1]) / (loc1[3] - loc1[1]))] \
                     for id1, loc1 in enumerate(boxes['bdb2D_pos'])
                     for id2, loc2 in enumerate(boxes['bdb2D_pos'])]

        locs = [num for loc in g_feature for num in loc]

        pe = torch.zeros(len(locs), d_model)
        position = torch.from_numpy(np.array(locs)).unsqueeze(1).float()
        #print(position.shape)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        boxes['g_feature'] = pe.view(n_objects * n_objects, rel_cfg.d_g)

        # encode class
        cls_codes = torch.zeros([len(boxes['size_cls']), 9])
        cls_codes[range(len(boxes['size_cls'])), boxes['size_cls']] = 1
        boxes['size_cls'] = cls_codes

        #layout = sequence['layout']
        patch = []
        box_feat=[]
        for bdb in boxes['bdb2D_pos']:
            img = image.crop((bdb[0]/2, bdb[1]/2, bdb[2]/2, bdb[3]/2))
            img = np.asarray(img)/255.0
            if self.mode=="train":
                img = self.augment_image(img)
            img = data_transforms(img).float()
            patch.append(img)

            box_feat.append(torch.tensor([(bdb[2]-bdb[0])/width,(bdb[3]-bdb[1])/height,(bdb[2]+bdb[0])/width,(bdb[3]+bdb[1])/height]))
        #print(box_feat)
        boxes['patch'] = torch.stack(patch)
        boxes['box_feat']=torch.stack(box_feat)
        image = data_transforms(image)
        if self.mode != 'test':
            for d, k in zip([camera, boxes], ['world_R_inv', 'bdb3d_inv']):
                if k in d.keys():
                    d.pop(k)
        return {'image':image, 'depth': pil2tensor(depth).squeeze(), 'layout':layout,'boxes_batch':boxes, 'camera':camera, 'sequence_id': sequence['sequence_id']}


default_collate = torch.utils.data.dataloader.default_collate
def recursive_convert_to_torch(elem):
    if torch.is_tensor(elem):
        return elem
    elif type(elem).__module__ == 'numpy':
        if elem.size == 0:
            return torch.zeros(elem.shape).type(torch.DoubleTensor)
        else:
            return torch.from_numpy(elem)
    elif isinstance(elem, int):
        return torch.LongTensor([elem])
    elif isinstance(elem, float):
        return torch.DoubleTensor([elem])
    elif isinstance(elem, collections.Mapping):
        return {key: recursive_convert_to_torch(elem[key]) for key in elem}
    elif isinstance(elem, collections.Sequence):
        return [recursive_convert_to_torch(samples) for samples in elem]
    else:
        return elem


def collate_fn(batch):
    """
    Data collater.

    Assumes each instance is a dict.
    Applies different collation rules for each field.
    Args:
        batches: List of loaded elements via Dataset.__getitem__
    """
    collated_batch = {}
    # iterate over keys
    for key in batch[0]:
        if key == 'boxes_batch':
            collated_batch[key] = dict()
            for subkey in batch[0][key]:
                if subkey == 'mask' or subkey == "jid":
                    tensor_batch = [elem[key][subkey] for elem in batch]
                else:
                    list_of_tensor = [recursive_convert_to_torch(elem[key][subkey]) for elem in batch]
                    tensor_batch = torch.cat(list_of_tensor)
                collated_batch[key][subkey] = tensor_batch
        elif key == 'depth':
            collated_batch[key] = [elem[key] for elem in batch]
        else:
            collated_batch[key] = default_collate([elem[key] for elem in batch])

    interval_list = [elem['boxes_batch']['patch'].shape[0] for elem in batch]
    collated_batch['obj_split'] = torch.tensor(
        [[sum(interval_list[:i]), sum(interval_list[:i + 1])] for i in range(len(interval_list))])

    return collated_batch

def worker_init_fn(worker_id):
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)

def Front_det_dataloader(config, mode='train'):
    dataloader = DataLoader(dataset=Front_Det_Dataset(config, mode),
                            num_workers=config['data']['num_workers'],
                            batch_size=config['data']['batch_size'],
                            shuffle=(mode == 'train'),
                            collate_fn=collate_fn,
                            worker_init_fn=worker_init_fn)
    return dataloader
