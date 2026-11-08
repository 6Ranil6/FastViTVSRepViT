"""Обновляет таблицу и диаграммы мобильных замеров в отчёте по сводному CSV."""

import csv
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt

from .core import ROOT

START = '<!-- mobile-results:start -->'
END = '<!-- mobile-results:end -->'
MODEL_NAMES = {'fastvit': 'FastViT-SA12', 'repvit': 'RepViT-M0.9'}


def plot_mobile_results(rows, output_dir):
    matplotlib.use('Agg')

    series = defaultdict(list)
    for row in rows:
        series[row['device'], row['model'], row['sha256']].append(row)

    threads = sorted({int(row['threads']) for row in rows})
    positions = {count: index for index, count in enumerate(threads)}
    width = 0.75 / len(series)
    show_device = len({device for device, _, _ in series}) > 1
    latency, (median_axis, tail_axis) = plt.subplots(
        1, 2, sharey=True, figsize=(10, 4.5), layout='constrained'
    )
    accuracy, accuracy_axis = plt.subplots(figsize=(7, 4.2), layout='constrained')

    for index, ((device, model, digest), results) in enumerate(sorted(series.items())):
        label = MODEL_NAMES.get(model, model)
        if show_device:
            label = f'{label} · {device}'
        if sum(key[:2] == (device, model) for key in series) > 1:
            label = f'{label} · {digest[:8]}'
        offset = (index - (len(series) - 1) / 2) * width
        x = [positions[int(row['threads'])] + offset for row in results]
        spread = [float(row['p50_std_ms'] or 'nan') for row in results]
        median_axis.bar(
            x, [float(row['p50_mean_ms']) for row in results],
            width, yerr=spread, capsize=3, label=label,
        )
        tail_axis.bar(x, [float(row['p95_mean_ms']) for row in results], width)
        bars = accuracy_axis.bar(
            x, [100 * float(row['subset_accuracy']) for row in results],
            width, label=label,
        )
        accuracy_axis.bar_label(bars, fmt='%.2f%%', padding=3, fontsize=9)

    for axis in (median_axis, tail_axis, accuracy_axis):
        axis.set_xticks(range(len(threads)), threads)
        axis.set_xlabel('Потоки CPU')
        axis.set_axisbelow(True)
        axis.grid(axis='y', alpha=0.25)
        axis.set_ylim(bottom=0)

    median_axis.set(title='p50 · среднее ± SD', ylabel='Задержка, мс')
    tail_axis.set_title('p95 · среднее')
    median_axis.legend()
    accuracy_axis.set(title='Top-1 на контрольной подвыборке', ylabel='Точность, %')
    accuracy_axis.set_yticks(range(0, 101, 20))
    accuracy_axis.set_ylim(0, 108)
    accuracy.legend(loc='outside upper center', ncols=min(2, len(series)), frameon=False)

    latency.savefig(output_dir / 'mobile_latency.png', dpi=180)
    accuracy.savefig(output_dir / 'mobile_accuracy.png', dpi=180)
    plt.close(latency)
    plt.close(accuracy)


def update_report(csv_path, report_path):
    with csv_path.open(encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f'В {csv_path} нет результатов замеров.')

    lines = [
        START,
        '## Мобильные замеры',
        '',
        '| Устройство | Модель | Потоки | Испытания | p50, мс | SD p50, мс | p95, мс | Top-1 на подвыборке | Макс. тепловой статус |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in sorted(rows, key=lambda item: (item['device'], int(item['threads']), item['model'], item['sha256'])):
        model = MODEL_NAMES.get(row['model'], row['model'])
        spread = f"{float(row['p50_std_ms']):.2f}" if row['p50_std_ms'] else '—'
        lines.append(
            f"| {row['device']} | {model} | {row['threads']} | {row['series']} | "
            f"{float(row['p50_mean_ms']):.2f} | {spread} | {float(row['p95_mean_ms']):.2f} | "
            f"{float(row['subset_accuracy']):.2%} | {float(row['max_thermal_status']):g} |"
        )
    lines.extend([
        '',
        '### Что означают колонки',
        '',
        '- **Устройство** — модель телефона, на котором проводились замеры.',
        '- **Модель** — нейросеть, для которой рассчитаны показатели.',
        '- **Потоки** — число потоков CPU, заданное при запуске модели.',
        '- **Испытания** — число отдельных прогонов в этом режиме.',
        '- **p50, мс** — медианная задержка одного вызова модели, усреднённая по испытаниям.',
        '- **SD p50, мс** — выборочное стандартное отклонение медианной задержки между испытаниями.',
        '- **p95, мс** — порог задержки, в который уложились не менее 95% замеров в каждом испытании. '
        'В таблице приведено среднее этих порогов.',
        '- **Top-1 на подвыборке** — средняя по испытаниям доля фото, '
        'для которых класс с наибольшей оценкой совпал с правильным.',
        '- **Макс. тепловой статус** — наибольший статус Android, записанный после замеров '
        'среди испытаний для этой модели и числа потоков. '
        'Код 1 означает лёгкое ограничение производительности из-за нагрева, код 2 — умеренное. '
        '[Документация Android](https://developer.android.com/reference/android/os/PowerManager#THERMAL_STATUS_LIGHT)',
        '',
        'После каждого прогона журнал приложения показывает p50, p95 и число правильных предсказаний. '
        'В сохранённом CSV эти показатели находятся в строках `summary` '
        'под ключами `p50_ms`, `p95_ms` и `subset_accuracy`. '
        'Там же указаны тепловые статусы до и после замеров под ключами `thermal_before` и `thermal_after`. '
        '[CSV отдельных прогонов](report_phone/)',
        '',
        f'[Сводный CSV]({csv_path.name})',
        '',
        '### Диаграммы мобильных замеров',
        '',
        '![Задержка p50 и p95 в зависимости от числа потоков CPU](mobile_latency.png)',
        '',
        '![Точность Top-1 в зависимости от числа потоков CPU](mobile_accuracy.png)',
        END,
    ])
    block = '\n'.join(lines)

    report = report_path.read_text(encoding='utf-8')
    if START in report or END in report:
        before, start, rest = report.partition(START)
        _, end, after = rest.partition(END)
        if not start or not end or report.count(START) != 1 or report.count(END) != 1:
            raise ValueError('Проверьте границы раздела мобильных замеров в отчёте.')
        report = before + block + after
    else:
        report = report.rstrip() + '\n\n' + block + '\n'
    plot_mobile_results(rows, report_path.parent)
    report_path.write_text(report, encoding='utf-8')


def main():
    report_path = ROOT / 'report.md'
    update_report(ROOT / 'mobile_comparison.csv', report_path)
    print(report_path)


if __name__ == '__main__':
    main()
