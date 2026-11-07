"""Собирает результаты Android из CSV отдельно для каждого числа потоков и хеша модели."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from .core import ROOT

def read_run(path):
    with open(path) as f: rows=list(csv.DictReader(f))
    meta={r['model']:r['value'] for r in rows if r['record']=='meta'}
    if meta.get('backend') != 'XNNPACK_CPU_FP32': raise ValueError(f'{path}: unexpected backend')
    results=[]
    for model in ('fastvit','repvit'):
        summary={r['index']:float(r['value']) for r in rows if r['record']=='summary' and r['model']==model}
        times=[float(r['value']) for r in rows if r['record']=='latency' and r['model']==model]
        if len(times)!=int(meta['measurements']) or 'reference_agreement' not in summary:
            raise ValueError(f'{path}: incomplete run for {model}')
        if summary['reference_agreement']!=1 or summary['max_abs_logit_error']>.02:
            raise ValueError(f'{path}: export parity failed for {model}')
        results.append((model,summary,meta))
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('csvs',nargs='+');args=p.parse_args()
    groups=defaultdict(list)
    for path in args.csvs:
        for model,result,meta in read_run(path):
            groups[(meta['device'],model,meta['threads'],meta[model])].append(result)
    out=ROOT/'runs/report';out.mkdir(parents=True,exist_ok=True)
    columns=['device','model','threads','sha256','series','p50_mean_ms','p50_std_ms','p95_mean_ms','subset_accuracy','max_thermal_status']
    with open(out/'mobile_comparison.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for (device,model,threads,digest),runs in sorted(groups.items()):
            p50=[r['p50_ms'] for r in runs]
            writer.writerow(dict(device=device,model=model,threads=threads,sha256=digest,series=len(runs),p50_mean_ms=np.mean(p50),p50_std_ms=np.std(p50,ddof=1) if len(runs)>1 else '',p95_mean_ms=np.mean([r['p95_ms'] for r in runs]),subset_accuracy=np.mean([r['subset_accuracy'] for r in runs]),max_thermal_status=max(r['thermal_after'] for r in runs)))
    print(out/'mobile_comparison.csv')

if __name__=='__main__':main()
