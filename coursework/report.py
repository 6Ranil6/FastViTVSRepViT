import csv
from statistics import mean, stdev
from .core import *

SEEDS = (42, 43, 44)
QUALITY_METRICS = ('accuracy', 'macro_f1', 'mean_class_accuracy')
MANUAL_RESULTS_MARKER = '<!-- Результаты экспорта и запуска -->'


def seed_summary(rows):
    by_variant = {}
    for row in rows:
        if row['seed'] in SEEDS:
            by_variant.setdefault(row['variant'], []).append(row)
    summary = []
    for variant, runs in sorted(by_variant.items()):
        if sorted(run['seed'] for run in runs) != list(SEEDS):
            continue
        item = {'variant': variant, 'seeds': ','.join(map(str, SEEDS))}
        for key in QUALITY_METRICS:
            values = [run[key] for run in runs]
            item[f'{key}_mean'] = mean(values)
            item[f'{key}_sd'] = stdev(values)
        summary.append(item)
    return summary

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    rows=[]
    histories = {}
    validation_fig, validation_ax = plt.subplots(figsize=(9, 5))
    out=ROOT/'runs/report'; out.mkdir(parents=True,exist_ok=True)
    for metrics_path in sorted((ROOT/'runs/full').glob('*/test_metrics.json')):
        result=read_json(metrics_path)
        if result['n'] != 6149: continue
        train=read_json(metrics_path.parent/'training.json')
        row={k:result[k] for k in ('model','seed','accuracy','macro_f1','mean_class_accuracy','training_parameters')}
        row['variant']=read_json(metrics_path.parent/'config.json')['models'][result['model']]
        row['training_seconds']=train['seconds']
        export=ROOT/'exports/full'/metrics_path.parent.name/'export.json'
        if export.exists():
            row.update({k:read_json(export)[k] for k in ('deploy_parameters','approx_macs','pte_bytes','pte_max_abs_error')})
        rows.append(row)
        cm=np.load(metrics_path.parent/'confusion_matrix.npy')
        supports=cm.sum(axis=1)
        recalls=np.divide(cm.diagonal(),supports,out=np.zeros(102,dtype=float),where=supports>0)
        worst=np.argsort(recalls)[:15]
        fig,ax=plt.subplots(figsize=(8,4))
        ax.bar([str(i+1) for i in worst],recalls[worst])
        ax.set(xlabel='Flower class ID (1-based)',ylabel='Test recall',ylim=(0,1),title=metrics_path.parent.name+' — lowest recall')
        fig.tight_layout();fig.savefig(out/f'{metrics_path.parent.name}-errors.png');plt.close(fig)
        with open(out/f'{metrics_path.parent.name}-per-class.csv','w') as f:
            w=csv.writer(f);w.writerow(['class_id_1based','test_images','correct','recall'])
            w.writerows((i+1,int(supports[i]),int(cm[i,i]),float(recalls[i])) for i in range(102))
        with open(metrics_path.parent/'history.csv') as f: hist=list(csv.DictReader(f))
        run_name = metrics_path.parent.name
        histories[run_name] = hist
        validation_ax.plot([int(r['epoch']) for r in hist],
                           [float(r['val_accuracy']) for r in hist],
                           marker='o', markersize=3, label=run_name)
    if not rows: raise SystemExit('No complete test evaluations; run evaluate first.')
    columns=sorted(set().union(*(r.keys() for r in rows)))
    with open(out/'comparison.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=columns); w.writeheader(); w.writerows(rows)
    first_finetune_epochs = {
        next((int(r['epoch']) for r in history if r['stage'] == 'finetune'), None)
        for history in histories.values()
    }
    transition = next(iter(first_finetune_epochs)) if len(first_finetune_epochs) == 1 else None
    if transition and transition > 1:
        validation_ax.axvline(transition - 0.5, color='0.4', linestyle='--',
                             label='Начало дообучения всей сети')
    validation_ax.set(xlabel='Эпоха', ylabel='Top-1 на валидации',
                      xticks=sorted({int(r['epoch']) for hist in histories.values() for r in hist}))
    validation_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    validation_ax.grid(axis='y', alpha=0.25)
    validation_ax.legend()
    validation_fig.tight_layout()
    validation_fig.savefig(ROOT/'validation.png', dpi=160)
    plt.close(validation_fig)
    lines=['# Реальные результаты экспериментов','', '| Модель | Seed | Test accuracy | Macro-F1 | Обучение, мин |','|---|---:|---:|---:|---:|']
    lines += [f"| {r['variant']} | {r['seed']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['training_seconds']/60:.1f} |" for r in rows]
    summary = seed_summary(rows)
    if summary:
        with (out/'seed_summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        lines += ['', '## Устойчивость по seed 42, 43 и 44', '',
                  'Среднее ± выборочное стандартное отклонение (n = 3). Метрики указаны в процентах, отклонения — в процентных пунктах.', '',
                  '| Модель | Top-1 accuracy | Macro-F1 | Средняя точность по классам |',
                  '|---|---:|---:|---:|']
        for item in summary:
            values = [f"{item[f'{key}_mean']*100:.2f} ± {item[f'{key}_sd']*100:.2f}" for key in QUALITY_METRICS]
            lines.append('| ' + ' | '.join([item['variant'], *values]) + ' |')
        if len(summary) == 2:
            seed_accuracy = {(r['variant'], r['seed']): r['accuracy'] for r in rows}
            first, second = (item['variant'] for item in summary)
            deltas = [(seed_accuracy[first, seed] - seed_accuracy[second, seed])*100 for seed in SEEDS]
            lines += ['', f'Парная разница Top-1 ({first} минус {second}) по seed 42, 43 и 44: '
                      + ', '.join(f'{delta:+.2f}' for delta in deltas)
                      + f' п. п. Среднее ± SD: {mean(deltas):+.2f} ± {stdev(deltas):.2f} п. п.']
    if len(summary) < 2:
        lines += ['', 'Сводка по модели появится после полной оценки всех трёх seed (42, 43, 44).']
    lines += ['', '## Как шло обучение', '',
              '![Точность на валидации по эпохам для каждой модели и seed](validation.png)', '',
              'По горизонтали отложены эпохи обучения, по вертикали — доля правильных предсказаний на валидационной выборке (Top-1). Каждая линия соответствует одной модели и одному seed.']
    if transition and transition > 1:
        changes = []
        for history in histories.values():
            accuracy = {int(r['epoch']): float(r['val_accuracy']) for r in history}
            changes.append(100 * (accuracy[transition] - accuracy[transition - 1]))
        lines += ['', f'До эпохи {transition} обучалась только классификационная голова. Пунктир отмечает переход к дообучению всей сети.']
        if all(change < 0 for change in changes):
            drops = [-change for change in changes]
            lines += ['', f'На первой эпохе дообучения всей сети точность снизилась во всех запусках на {min(drops):.2f}–{max(drops):.2f} п. п.']
        if all(max(float(r['val_accuracy']) for r in history if r['stage'] == 'finetune')
               > max(float(r['val_accuracy']) for r in history if r['stage'] == 'head')
               for history in histories.values()):
            lines += ['', 'Затем точность восстановилась. Лучший результат дообучения в каждом запуске превысил максимум этапа обучения головы. К концу кривые меняются уже не так сильно, и в некоторых запусках максимум достигнут до последней эпохи.']
    lines += ['', 'Для оценки на test использовали веса с максимальной Top-1 на валидации. Если максимум достигнут несколько раз, берётся первая из этих эпох.', '',
              '| Запуск | Лучшая эпоха | Top-1 на валидации |', '|---|---:|---:|']
    for run_name, history in histories.items():
        best = max(history, key=lambda r: float(r['val_accuracy']))
        lines.append(f"| {run_name} | {best['epoch']} | {float(best['val_accuracy']):.2%} |")
    report_path = ROOT / 'report.md'
    if report_path.exists():
        _, marker, manual_results = report_path.read_text().partition(MANUAL_RESULTS_MARKER)
        if marker:
            lines.extend(['', marker + manual_results.rstrip()])
    report_path.write_text('\n'.join(lines) + '\n')
    print(report_path)

if __name__=='__main__': main()
