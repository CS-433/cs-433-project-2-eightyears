from data.dataloaders import Topo, Topo2
from tools.options import Options
from torchvision import transforms, models
import numpy as np
import os.path as osp
import torch

opt = Options().parse()
#cuda = torch.cuda.is_available()
#device = "cuda:" + ",".join(str(i) for i in opt.gpus) if cuda else "cpu"

stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')
stats = np.loadtxt(stats_file)

data_transform = transforms.Compose([
    transforms.Resize(opt.cropsize),
    transforms.CenterCrop(opt.cropsize),
    transforms.ToTensor(),
    transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())



kwargs = dict(scene=opt.scene, data_path=opt.data_dir, train=True, transform=data_transform, target_transform=target_transform, seed=opt.seed)
data_set = Topo2(**kwargs)
