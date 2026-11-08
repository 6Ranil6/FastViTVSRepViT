from pathlib import Path
import hashlib
import json
import random
import numpy as np
import torch
import timm
from PIL import Image
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/flowers-102'
NUM_CLASSES = 102

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def device_for(name='auto'):
    if name != 'auto':
        return torch.device(name)
    return torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

def image_transform(cfg, train=False, pixels=False):
    size = cfg['image_size']
    spatial = [T.RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=InterpolationMode.BILINEAR), T.RandomHorizontalFlip()] if train else [T.Resize(round(size / 0.875), interpolation=InterpolationMode.BILINEAR), T.CenterCrop(size)]
    if pixels:
        return T.Compose(spatial)
    return T.Compose(spatial + [T.ToTensor(), T.Normalize(cfg['mean'], cfg['std'])])

class Flowers(Dataset):
    def __init__(self, split, cfg, train=False):
        labels = loadmat(DATA / 'imagelabels.mat')['labels'].reshape(-1).astype(int) - 1
        key = {'train': 'trnid', 'val': 'valid', 'test': 'tstid'}[split]
        self.ids = loadmat(DATA / 'setid.mat')[key].reshape(-1).astype(int).tolist()
        self.labels = labels
        self.transform = image_transform(cfg, train)

    def __len__(self):
        return len(self.ids)

    def path(self, image_id):
        return DATA / 'jpg' / f'image_{image_id:05d}.jpg'

    def __getitem__(self, i):
        image_id = self.ids[i]
        with Image.open(self.path(image_id)) as image:
            x = self.transform(image.convert('RGB'))
        return x, int(self.labels[image_id - 1]), image_id

def model_for(cfg, key, pretrained=False):
    return timm.create_model(cfg['models'][key], pretrained=pretrained, num_classes=NUM_CLASSES)

def load_model(checkpoint):
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = model_for(ckpt['config'], ckpt['model'])
    model.load_state_dict(ckpt['state_dict'])
    return model.eval(), ckpt

def metrics(y, pred):
    cm = confusion_matrix(y, pred, labels=list(range(NUM_CLASSES)))
    present = cm.sum(1) > 0
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=list(range(NUM_CLASSES)), average='macro', zero_division=0)),
            'mean_class_accuracy': float(np.mean(cm.diagonal()[present] / cm.sum(1)[present]))}, cm

@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    ys, preds, ids, logits = [], [], [], []
    loss_sum = 0.0
    for x, y, image_ids in loader:
        out = model(x.to(device)).cpu()
        loss_sum += torch.nn.functional.cross_entropy(out, y, reduction='sum').item()
        ys.extend(y.tolist()); preds.extend(out.argmax(1).tolist()); ids.extend(image_ids.tolist())
        logits.append(out.numpy())
    result, cm = metrics(ys, preds)
    result['loss'] = loss_sum / len(ys)
    result['n'] = len(ys)
    return result, cm, ids, ys, preds, np.concatenate(logits)
