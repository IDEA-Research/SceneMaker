import numpy as np
import pickle as p
import torch
bin = {}
NUM_ORI_BIN=6
ORI_BIN_WIDTH = float(2 * np.pi / NUM_ORI_BIN)
ori_bin=[[(i - NUM_ORI_BIN / 2) * ORI_BIN_WIDTH, (i - NUM_ORI_BIN / 2 + 1) * ORI_BIN_WIDTH] for i
                          in range(NUM_ORI_BIN)]
bin["ori_bin"]=ori_bin
NUM_DEPTH_BIN = 10
DEPTH_WIDTH = 1.0
# centroid_bin = [0, 6]
bin['centroid_bin'] = [[i * DEPTH_WIDTH, (i + 1) * DEPTH_WIDTH] for i in
                       range(NUM_DEPTH_BIN)]

NUM_LAYOUT_ORI_BIN = 2
ORI_LAYOUT_BIN_WIDTH = np.pi / 4
bin['layout_ori_bin'] = [[np.pi / 4 + i * ORI_LAYOUT_BIN_WIDTH, np.pi / 4 + (i + 1) * ORI_LAYOUT_BIN_WIDTH] for i in range(NUM_LAYOUT_ORI_BIN)]

'''camera bin'''
PITCH_NUMBER_BINS = 2
PITCH_WIDTH = 40 * np.pi / 180
ROLL_NUMBER_BINS = 2
ROLL_WIDTH = 20 * np.pi / 180

# pitch_bin = [[-60 * np.pi/180, -20 * np.pi/180], [-20 * np.pi/180, 20 * np.pi/180]]
bin['pitch_bin'] = [[-60.0 * np.pi / 180 + i * PITCH_WIDTH, -60.0 * np.pi / 180 + (i + 1) * PITCH_WIDTH] for
                    i in range(PITCH_NUMBER_BINS)]
# roll_bin = [[-20 * np.pi/180, 0 * np.pi/180], [0 * np.pi/180, 20 * np.pi/180]]
bin['roll_bin'] = [[-20.0 * np.pi / 180 + i * ROLL_WIDTH, -20.0 * np.pi / 180 + (i + 1) * ROLL_WIDTH] for i in
                   range(ROLL_NUMBER_BINS)]

category_label_mapping = {"table":0,
                          "sofa": 1,
                          "cabinet": 2,
                          "night_stand":3,
                          "chair":4,
                          "bookshelf":5,
                          "bed":6,
                          "desk":7,
                          "dresser":8
                          }
mean_size_path='./data/3dfront/3dfuture-mean_size.pkl'
with open(mean_size_path,'rb') as f:
    content=p.load(f)
    avg_size={}
    #print(content)
    for key in content:
        label_id=category_label_mapping[key]
        avg_size[label_id]=content[key]
#print(avg_size)
bin['avg_size'] = np.vstack([avg_size[key] for key in range(len(avg_size))])

mean_layout_path="./data/3dfront/avg_layout.pkl"
f=open(mean_layout_path,'rb')
avg_layout=p.load(f)
f.close()

bins_tensor={}
for key in bin:
    bins_tensor[key]=torch.tensor(bin[key]).float().cuda()