import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import os.path as osp
import numpy as np
import matplotlib
import sys
import cv2
import matplotlib.pyplot as plt
from tools.options import Options

DISPLAY = 'DISPLAY' in os.environ
if not DISPLAY:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from network.atloc import AtLoc, AtLocPlus
from data.dataloaders import SevenScenes, RobotCar, MF, Topo, Topo2, Topo3
from tools.utils import load_state_dict
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, models

# config
opt = Options().parse()
cuda = torch.cuda.is_available()
device = "cuda:" + ",".join(str(i) for i in opt.gpus) if cuda else "cpu"

# Model
feature_extractor = models.resnet34(pretrained=True)
atloc = AtLoc(feature_extractor, droprate=opt.test_dropout, pretrained=False, lstm=opt.lstm)
if opt.model == 'AtLoc':
    model = atloc
elif opt.model == 'AtLocPlus':
    model = AtLocPlus(atlocplus=atloc)
else:
    raise NotImplementedError
model.eval()

stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')
stats = np.loadtxt(stats_file)
# transformer
data_transform = transforms.Compose([
    transforms.Resize(opt.cropsize),
    transforms.CenterCrop(opt.cropsize),
    transforms.ToTensor(),
    transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# Load the dataset
kwargs = dict(scene=opt.scene, data_path=opt.data_dir, train=True, transform=data_transform, target_transform=target_transform, seed=opt.seed)
kwargs2 = dict(scene=opt.scene, data_path=opt.data_dir, train=True, transform=None, target_transform=target_transform, seed=opt.seed)

if opt.model == 'AtLoc':
    if opt.dataset == '7Scenes':
        data_set = SevenScenes(**kwargs)
    elif opt.dataset == 'RobotCar':
        data_set = RobotCar(**kwargs)
    elif opt.dataset == 'Topo':
        data_set = Topo(**kwargs)
        data_set2 = Topo(**kwargs2)
    elif opt.dataset == 'comballaz':
        data_set = Topo2(**kwargs)
        data_set2 = Topo2(**kwargs2)
    elif opt.dataset == 'EPFL':
        dset = Topo3(**kwargs)
    else:
        raise NotImplementedError
elif opt.model == 'AtLocPlus':
    kwargs = dict(kwargs, dataset=opt.dataset, skip=opt.skip, steps=opt.steps, variable_skip=opt.variable_skip)
    data_set = MF(real=opt.real, **kwargs)
else:
    raise NotImplementedError
L = len(data_set)
kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
loader = DataLoader(data_set, batch_size=1, shuffle=False, **kwargs)

# load weights
model.to(device)
weights_filename = osp.expanduser(opt.weights)
if osp.isfile(weights_filename):
    checkpoint = torch.load(weights_filename, map_location=device)
    load_state_dict(model, checkpoint['model_state_dict'])
    print('Loaded weights from {:s}'.format(weights_filename))
else:
    print('Could not load weights from {:s}'.format(weights_filename))
    sys.exit(-1)

if opt.final_weights is not '':
    weights2_filename = osp.expanduser(opt.final_weights)
    if osp.isfile(weights2_filename):
        checkpoint2 = torch.load(weights2_filename, map_location=device)
        #load_state_dict(model, checkpoint2['model_state_dict'])
        print('Loaded second weights from {:s}'.format(weights2_filename))
    else:
        print('Could not load second weights from {:s}'.format(weights2_filename))
        sys.exit(-1)

# get frame size
img, _ = data_set[0]

print('Saving imgs to  to {:s} with frames size {:d} x {:d}'.format(opt.results_dir, img.size(2), img.size(1)))

# inference
cm_jet = plt.cm.get_cmap('jet')
for batch_idx, (data, target) in enumerate(loader):
    if batch_idx < 10 and True:
        continue

    data2 = data.clone()
    out_filename = osp.join(opt.results_dir, '{:s}_{:s}_attention_{:s}_{:s}.png'.format(opt.dataset, opt.scene, opt.model,str(batch_idx)))

    data = data.to(device)
    data_var = Variable(data, requires_grad=True)

    model.zero_grad()
    pose = model(data_var)
    pose.mean().backward()

    act = data_var.grad.data.cpu().numpy()
    act = act.squeeze().transpose((1, 2, 0))
    img = data[0].cpu().numpy()
    img = img.transpose((1, 2, 0))

    act *= img
    act = np.amax(np.abs(act), axis=2)
    act -= act.min()
    act /= act.max()
    act = cm_jet(act)[:, :, :3]
    act *= 255

    img *= stats[1]
    img += stats[0]
    img *= 255
    img = img[:, :, ::-1]

    img = 0.5 * img + 0.5 * act
    img = np.clip(img, 0, 255)
    if opt.final_weights is not '':
        load_state_dict(model, checkpoint2['model_state_dict'])
        data = data2.to(device)
        data_var = Variable(data, requires_grad=True)

        model.zero_grad()
        pose = model(data_var)
        pose.mean().backward()
        act1 = act.copy()
        act = data_var.grad.data.cpu().numpy()
        act = act.squeeze().transpose((1, 2, 0))
        img2 = data[0].cpu().numpy()
        img2 = img2.transpose((1, 2, 0))

        act *= img2
        act = np.amax(np.abs(act), axis=2)
        act -= act.min()
        act /= act.max()
        act = cm_jet(act)[:, :, :3]
        act *= 255

        img2 *= stats[1]
        img2 += stats[0]
        img2 *= 255
        img2 = img2[:, :, ::-1]

        img2 = 0.5 * img2 + 0.5 * act
        img2 = np.clip(img2, 0, 255)

        grayscale1 = cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2GRAY)
        grayscale2 = cv2.cvtColor(img2.astype('float32'), cv2.COLOR_RGB2GRAY)
        grayscale3 = cv2.subtract(grayscale1,grayscale2)
        img3 = cv2.cvtColor(grayscale3, cv2.COLOR_GRAY2RGB)
        img3 = (img3 - np.mean(img3, axis=(0, 1))) // np.std(img3, axis=(0, 1))
        #img3 = ((img3 - img3.min()) / (img3.max() - img3.min())) * 255
        original,_ = data_set2[batch_idx]

        original = np.array(original.convert('RGB'))
        original = cv2.resize(original, (256,256))
        plt.figure()
        plt.imshow(original)
        plt.savefig(opt.results_dir+'/Original_' + str(batch_idx) + '.png', bbox_inches='tight')
        plt.imshow(img3, cmap='gist_heat_r',alpha=0.5)
        plt.savefig(opt.results_dir+'/AttentionMap_'+str(batch_idx)+'.png',bbox_inches='tight')

        original = original[:,:,::-1]
        imgcp = img3.copy()
        img3[:,:,0] = 0
        img3[:,:,1] = 0
        originalcp = original.copy()
        originalcp[:,:,2] = 0
        img3 = np.where(img3>0, 0,img3) + np.where(imgcp >0,original,originalcp)
        img = np.concatenate((original,img,img2,img3),axis=1)
        load_state_dict(model, checkpoint['model_state_dict'])


    cv2.imwrite(out_filename,img.astype(np.uint8))

    if batch_idx % 200 == 0:
        print('{:d} / {:d}'.format(batch_idx, len(loader)))

    print('{:s} written'.format(out_filename))
    # TODO: edit options to add number of desired images
    if batch_idx >= 20 and True:
        print('Stored images in '+opt.results_dir)
        break



