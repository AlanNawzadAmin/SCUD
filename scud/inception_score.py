import numpy as np
import torch
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import data

def imgtrans(x):
    x = np.transpose(x, (1,2,0))
    return x

def preproc(images):
    pils = [transforms.ToPILImage()(255 - imgtrans(img)) for img in images.astype(float)]
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    pils = torch.stack([(preprocess(img) * 255).to(torch.uint8) for img in pils])
    return pils

def load_real_data(N):
    cfg = OmegaConf.load('configs/basic.yaml')
    train_dataloader, test_dataloader = data.get_dataloaders(cfg)
    train_dataloader = iter(train_dataloader)
    real_ims = np.concatenate([next(train_dataloader)[0]
                               for i in tqdm(range(int(np.ceil(N/16))))])
    return real_ims[:N]

def inception_score(images, batch_size=32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inception_score_metric = InceptionScore().to(device)

    pils = preproc(images)
    for i in tqdm(range(0, len(images), batch_size)):
        batch = pils[i:i+batch_size].to(device)
        inception_score_metric.update(batch)

    return inception_score_metric.compute()

def fid_score(ims, real_ims, batch_size=32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fid = FrechetInceptionDistance(feature=2048).to(device)

    for i in tqdm(range(0, len(real_ims), batch_size)):
        fid.update(torch.tensor(real_ims[i:i+batch_size]).to(torch.uint8).to(device), real=True)  
    for i in tqdm(range(0, len(ims), batch_size)):
        fid.update(torch.tensor(ims[i:i+batch_size]).to(torch.uint8).to(device), real=False)
    return fid.compute()
