import argparse
import copy
import time
import os
import sys
import torch
from timm.utils import reparameterize_model
from torchvision.ops.misc import FrozenBatchNorm2d
from .core import *

def freeze_remaining_batch_norm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            assert not child.training and child.track_running_stats
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            with torch.no_grad():
                frozen.running_mean.copy_(child.running_mean)
                frozen.running_var.copy_(child.running_var)
                if child.affine:
                    frozen.weight.copy_(child.weight)
                    frozen.bias.copy_(child.bias)
            setattr(module, name, frozen)
        else:
            freeze_remaining_batch_norm(child)

def deploy_model(model, key):
    model = copy.deepcopy(model).eval().cpu()
    if key == 'repvit':
        model.fuse()
    else:
        model = reparameterize_model(model)
    freeze_remaining_batch_norm(model)
    return model.eval()

def main():
    p = argparse.ArgumentParser(); p.add_argument('checkpoint')
    p.add_argument('--output-dir', type=Path, help='Separate export directory for an independent experiment')
    args = p.parse_args(); torch.set_num_threads(4)
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.runtime import Runtime
    original, ckpt = load_model(args.checkpoint); cfg = ckpt['config']; key = ckpt['model']
    model = deploy_model(original, key)
    data = Flowers('val', cfg)
    samples = [data[i][0].unsqueeze(0) for i in range(8)]
    with torch.no_grad():
        reference = [original(x) for x in samples]
        fused = [model(x) for x in samples]
    fuse_error = max((a-b).abs().max().item() for a,b in zip(reference, fused))
    for a,b in zip(reference, fused): torch.testing.assert_close(a,b,atol=1e-3,rtol=1e-3)
    out = args.output_dir or ROOT / 'exports' / 'full' / f"{key}-seed{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    ep = torch.export.export(model, (samples[0],))
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    program = edge.to_executorch()
    path = out / f'{key}.pte'; path.write_bytes(program.buffer)
    runtime = Runtime.get().load_program(str(path)); method = runtime.load_method('forward')
    actual = [method.execute([x])[0] for x in samples]
    max_error = max((a-b).abs().max().item() for a,b in zip(reference, actual))
    for a,b in zip(reference, actual): torch.testing.assert_close(a,b,atol=2e-3,rtol=2e-3)
    assert all(a.argmax().item() == b.argmax().item() for a,b in zip(reference,actual))
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(samples[0])
    flops = counter.get_total_flops()
    deploy_parameters = sum(p.numel() for p in model.parameters()) + sum(m.weight.numel() + m.bias.numel() for m in model.modules() if isinstance(m, FrozenBatchNorm2d))
    write_json(out / 'export.json', {'model': key, 'variant': cfg['models'][key], 'seed': cfg['seed'], 'config': cfg, 'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256': sha256(args.checkpoint), 'pte_sha256': sha256(path), 'pte_bytes': path.stat().st_size, 'backend': 'XNNPACK CPU', 'precision': 'FP32', 'executorch': '1.0.0', 'deploy_parameters': deploy_parameters, 'batch_norm_export': 'Remaining eval BatchNorm2d expressed as frozen affine mul/add; affine weights included in parameter count', 'counted_flops': flops, 'approx_macs': flops/2, 'flop_note': 'PyTorch FlopCounter coverage; 2 FLOPs per multiply-accumulate; excludes uncounted ops', 'fuse_max_abs_error': fuse_error, 'pte_max_abs_error': max_error, 'parity_samples': 8})
    print('Validated export:', path, 'max error:', max_error, flush=True)

if __name__ == '__main__': main()
