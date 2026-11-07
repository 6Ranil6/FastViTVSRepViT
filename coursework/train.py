import argparse
import csv
import platform
import time
import importlib.metadata
import torch
from torch.utils.data import DataLoader
from .core import *

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['fastvit', 'repvit'], required=True)
    p.add_argument('--config', default=str(ROOT / 'configs/default.json'))
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int)
    p.add_argument('--run-root', type=Path, default=ROOT / 'runs',
                   help='Parent directory for an independent experiment')
    args = p.parse_args()
    cfg = read_json(args.config)
    cfg['models'] = {args.model: cfg['models'][args.model]}
    if args.seed is not None: cfg['seed'] = args.seed
    seed_everything(cfg['seed'])
    torch.set_num_threads(4)
    device = device_for(args.device)
    run = args.run_root / 'full' / f"{args.model}-seed{cfg['seed']}"
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'best.pt').exists():
        raise SystemExit(f'Refusing to overwrite {run}; choose another seed or move existing run.')
    write_json(run / 'config.json', cfg)
    write_json(run / 'environment.json', {'platform': platform.platform(), 'device': str(device), 'python': platform.python_version(), 'versions': {k: importlib.metadata.version(k) for k in ['torch', 'torchvision', 'timm', 'executorch']}, 'dataset_manifest': read_json(DATA / 'manifest.json')})
    train = Flowers('train', cfg, train=True)
    val = Flowers('val', cfg)
    gen = torch.Generator().manual_seed(cfg['seed'])
    train_dl = DataLoader(train, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['workers'], generator=gen)
    val_dl = DataLoader(val, batch_size=cfg['batch_size'], num_workers=cfg['workers'])
    model = model_for(cfg, args.model, pretrained=True).to(device)
    start = time.perf_counter(); best = -1.; epoch = 0
    with open(run / 'history.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch','stage','train_loss','val_loss','val_accuracy','val_macro_f1','seconds'])
        writer.writeheader()
        for stage, epochs, lr in [('head', cfg['head_epochs'], cfg['head_lr']), ('finetune', cfg['finetune_epochs'], cfg['finetune_lr'])]:
            for param in model.parameters(): param.requires_grad_(stage == 'finetune')
            classifier = model.get_classifier()
            for param in classifier.parameters(): param.requires_grad_(True)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=cfg['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
            for _ in range(epochs):
                epoch += 1; tick = time.perf_counter(); total = 0.; n = 0
                model.train()
                if stage == 'head':
                    # Фиксируем статистику BatchNorm и отключаем случайные преобразования в слоях основы.
                    model.eval(); classifier.train()
                for x, y, _ in train_dl:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(x), y)
                    loss.backward(); optimizer.step()
                    total += loss.item() * len(y); n += len(y)
                scheduler.step()
                result, *_ = predict(model, val_dl, device)
                row = dict(epoch=epoch, stage=stage, train_loss=total/n, val_loss=result['loss'], val_accuracy=result['accuracy'], val_macro_f1=result['macro_f1'], seconds=time.perf_counter()-tick)
                writer.writerow(row); f.flush(); print(args.model, row, flush=True)
                if result['accuracy'] > best:
                    best = result['accuracy']
                    checkpoint = {'state_dict': {k:v.detach().cpu() for k,v in model.state_dict().items()}, 'config': cfg, 'model': args.model, 'epoch': epoch, 'val_accuracy': best}
                    tmp = run / 'best.tmp'; torch.save(checkpoint, tmp); tmp.replace(run / 'best.pt')
    write_json(run / 'training.json', {'seconds': time.perf_counter()-start, 'best_val_accuracy': best, 'epochs': epoch})
    print('Saved', run / 'best.pt', flush=True)

if __name__ == '__main__': main()
