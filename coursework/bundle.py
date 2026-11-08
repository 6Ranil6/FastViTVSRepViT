import argparse
import shutil
import torch
from executorch.runtime import Runtime
from PIL import Image
from .core import *

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed', type=int, default=42); p.add_argument('--per-class', type=int, default=1)
    args=p.parse_args()
    if not 1 <= args.per_class <= 20: raise SystemExit('--per-class must be in 1..20')
    torch.set_num_threads(4)
    exports={k:ROOT/'exports/full'/f'{k}-seed{args.seed}' for k in ('fastvit','repvit')}
    metas={k:read_json(v/'export.json') for k,v in exports.items()}
    cfg=metas['fastvit']['config']
    expected=read_json(ROOT/'configs/default.json')
    for key, meta in metas.items():
        actual=meta['config']
        assert actual['models'][key] == expected['models'][key], f'Stale model export: {key}'
        assert {k:v for k,v in actual.items() if k != 'models'} == {k:v for k,v in cfg.items() if k != 'models'}, 'Models must use identical experiment settings'
        assert sha256(exports[key]/f'{key}.pte') == meta['pte_sha256'], 'Export checksum mismatch'
    data=Flowers('test',cfg)
    selected=[]; counts=[0]*102
    # Фиксированная подвыборка по классам. Выбор изображений не зависит от прогнозов моделей.
    for image_id in sorted(data.ids):
        target=int(data.labels[image_id-1])
        if counts[target]<args.per_class:
            selected.append((image_id,target)); counts[target]+=1
    assert min(counts)==args.per_class
    bundle=ROOT/'android/app/src/main/assets/bundle'
    bundle.mkdir(parents=True,exist_ok=True)
    images=[]
    programs={k:Runtime.get().load_program(str(v/f'{k}.pte')) for k,v in exports.items()}
    methods={k:v.load_method('forward') for k,v in programs.items()}
    pre=image_transform(cfg,pixels=True)
    for image_id,target in selected:
        with Image.open(data.path(image_id)) as im: pixels=np.asarray(pre(im.convert('RGB')),dtype=np.uint8).copy()
        name=f'image_{image_id:05d}.rgb'; (bundle/name).write_bytes(pixels.tobytes())
        x=torch.from_numpy(pixels).permute(2,0,1).float()/255.
        x=(x-torch.tensor(cfg['mean'])[:,None,None])/torch.tensor(cfg['std'])[:,None,None]
        # ExecuTorch ожидает непрерывный буфер NCHW с тем же порядком данных, что и в Android.
        x = x.unsqueeze(0).contiguous()
        outputs={k:methods[k].execute([x])[0].flatten().tolist() for k in methods}
        images.append({'file':name,'id':image_id,'target':target,'reference_logits':outputs})
    for k,v in exports.items(): shutil.copy2(v/f'{k}.pte',bundle/f'{k}.pte')
    write_json(bundle/'manifest.json',{'schema':1,'config':cfg,'subset_description':f'First {args.per_class} test image IDs per class; fixed before prediction','labels':read_json(ROOT/'data/labels.json'),'models':metas,'images':images})
    print('Android assets:',bundle,'images:',len(images))

if __name__=='__main__': main()
