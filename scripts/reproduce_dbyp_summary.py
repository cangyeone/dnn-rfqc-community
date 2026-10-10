"""Recompute the DB/YP v2 summary from published aggregate seed metrics.

This does not reconstruct per-event accuracy without the user-owned test data.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'benchmarks/2026-10-09-dbyp-v2'


def reproduce(output):
    manifest=json.loads((SOURCE/'source_manifest.json').read_text())
    for name,sha in manifest['sha256'].items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==sha,name
    rows=list(csv.DictReader((SOURCE/'seed_metrics.csv').open()))
    summary=list(csv.DictReader((SOURCE/'method_summary.csv').open()))
    protocol=json.loads((SOURCE/'protocol.json').read_text())
    expected={(m,s) for m in protocol['methods'] for s in protocol['training']['seeds']}
    assert len(rows)==len(expected)==30
    assert {(r['method'],int(r['seed'])) for r in rows}==expected
    keys=['accuracy','macro_f1','balanced_accuracy','good_auprc','good_precision','good_recall','bad_recall','auroc']
    for row in summary:
        subset=[r for r in rows if r['method']==row['method']]
        assert len(subset)==3
        for key in keys:
            v=[float(r[key]) for r in subset]
            assert abs(statistics.mean(v)-float(row[key+'_mean']))<1e-12
            assert abs(statistics.stdev(v)-float(row[key+'_sample_sd']))<1e-12
    assert {r['method'] for r in summary}==set(protocol['methods'])
    pairs=list(csv.DictReader((SOURCE/'single_multi_paired.csv').open()))
    for pair in pairs:
        items={r['method']:r for r in rows if r['seed']==pair['seed']}
        for key in ['accuracy','macro_f1','balanced_accuracy','good_auprc']:
            delta=100*(float(items['reference_multifilter'][key])-float(items['reference_ag3'][key]))
            assert abs(delta-float(pair[key+'_difference_pp']))<1e-12
    metrics=['accuracy','macro_f1','balanced_accuracy','good_precision','good_recall']
    lines=['# DB/YP 修正数据 v2：已完成训练结果','',
           '10 个配置 × 3 个种子，32 个留出台站、27,823 条测试记录。数值为百分数；± 为三个训练种子的样本标准差，不是泛化置信区间。每列最大均值加粗。','',
           '| 方法 | 准确率 | Macro-F1 | 平衡准确率 | good Precision | good Recall |',
           '|---|---:|---:|---:|---:|---:|']
    maximum={k:max(float(r[k+'_mean']) for r in summary) for k in metrics}
    for row in sorted(summary,key=lambda r:-float(r['accuracy_mean'])):
        values=[]
        for k in metrics:
            value=f"{float(row[k+'_mean'])*100:.2f} ± {float(row[k+'_sample_sd'])*100:.2f}"
            if float(row[k+'_mean'])==maximum[k]:value='**'+value+'**'
            values.append(value)
        lines.append('| `'+row['method']+'` | '+' | '.join(values)+' |')
    delta=[float(p['accuracy_difference_pp']) for p in pairs]
    lines+=['',f'同种子多频减 AG3：{statistics.mean(delta):+.3f} ± {statistics.stdev(delta):.3f} 个百分点（种子差值的均值 ± 样本标准差）；三次差值为 '+', '.join(f'{v:+.3f}' for v in delta)+'。',
            '', '默认调用使用固定种子 20260928，既不是最优种子挑选，也不是三模型集成。',
            '', '这些数据来自更新后的 DB/YP 主数据；不能与旧数据的历史分数混算。小震级导出数据无人工标签，未用于训练或计算此表准确率。',
            '', '脚本从每个种子的汇总指标复算均值、样本标准差和配对差值，并核对来源 SHA256。逐样本指标已在实验主机核验；此公开目录不含波形、标签、样本目录或逐样本预测。','']
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines))
    print(json.dumps(dict(status='passed',runs=len(rows),methods=len(summary),output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=SOURCE/'COMPARISON.md')
    reproduce(p.parse_args().output)
