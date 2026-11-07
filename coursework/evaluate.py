import argparse
import csv
import torch
from torch.utils.data import DataLoader
from .core import *

def main():
    p = argparse.ArgumentParser(); p.add_argument('checkpoint'); p.add_argument('--device', default='auto')
    args = p.parse_args(); torch.set_num_threads(4)
    model, ckpt = load_model(args.checkpoint)
    device = device_for(args.device); cfg = ckpt['config']
    data = Flowers('test', cfg)
    result, cm, ids, ys, preds, logits = predict(model.to(device), DataLoader(data, batch_size=cfg['batch_size']), device)
    result.update(model=ckpt['model'], variant=cfg['models'][ckpt['model']], checkpoint_sha256=sha256(args.checkpoint), seed=cfg['seed'], training_parameters=sum(p.numel() for p in model.parameters()))
    out = Path(args.checkpoint).parent
    write_json(out / 'test_metrics.json', result)
    np.save(out / 'confusion_matrix.npy', cm)
    np.save(out / 'test_logits.npy', logits)
    with open(out / 'predictions.csv', 'w') as f:
        w=csv.writer(f); w.writerow(['image_id','target','prediction']); w.writerows(zip(ids,ys,preds))
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
