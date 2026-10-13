"""Export external manual stacks on the host, or plot every small-event station.

Exported stacks are private observational derivatives, not a data publication.
The plot stage needs only the collected small stacks and exported references.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plot_dbyp_revision import GROUPS,METHODS,LABELS,COLORS,save

def export(root,out):
    names=[];raw=[];peak=[];counts=[]
    for path in sorted((root/'results/small').glob('*/plot_input.npz')):
        station=path.parent.name
        with np.load(root/'cache/complete'/f'{station}.npz') as z:
            keep=z['common_labels']==1;w=z['waveforms'][np.arange(len(keep)),(z['gaussians']==3).argmax(1)][keep].astype(float)
            names.append(station);counts.append(len(w));raw.append(w.mean(0));peak.append((w/np.maximum(np.abs(w).max(1,keepdims=True),1e-100)).mean(0))
    np.savez_compressed(out,stations=names,raw=raw,peak=peak,counts=counts,time=np.arange(501)*.1-10)

def plot(packet,external,review,summary):
    with np.load(external) as f:q=dict(f)
    lookup={s:i for i,s in enumerate(q['stations'])};dest=review/'small';dest.mkdir(exist_ok=True)
    metrics=pd.read_csv(summary);index=['<meta charset="utf-8"><title>Small-event review</title><h1>All 102 unlabelled small-event stations</h1><p>Waveform similarity is an external-population comparison, not classification accuracy. No back-azimuth or H–κ is inferred.</p><table><tr><th>Station</th><th>AG3 n / r</th><th>Multi n / r</th><th>Plots</th></tr>']
    with PdfPages(review/'all_small_stations_stacks.pdf') as book:
        for p in sorted((packet/'results/small').glob('*/stacks.npz')):
            station=p.parent.name;i=lookup[station]
            with np.load(p) as f:t=f['time'];s=dict(zip(f['keys'],f['waveforms']))
            v=metrics[(metrics.station==station)&(metrics.normalization=='peak')].set_index('method');fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
            for ax,group in zip(axes,GROUPS):
                ax.plot(t,q['peak'][i],color='black',lw=1.4,label=f'Main manual (n={q["counts"][i]})')
                for m in group:
                    n=int(v.loc[m,'n_selected']);key=m+'|peak'
                    if key in s:ax.plot(t,s[key],color=COLORS[m],lw=1,label=f'{LABELS[m]} (n={n})')
                    else:ax.plot([],[],color=COLORS[m],label=f'{LABELS[m]} (n=0; no stack)')
                ax.set_xlim(-2.5,20);ax.set_xlabel('Time after direct P (s)');ax.grid(alpha=.2);ax.legend(fontsize=6.5)
            axes[0].set_ylabel('Mean peak-normalized RF');fig.suptitle(station+' — small-event catalogue review; unlabelled');fig.tight_layout();book.savefig(fig);save(fig,dest/station)
            entries=[]
            for m in ['reference_ag3','reference_multifilter']:entries.append(f'<td>{int(v.loc[m,"n_selected"])} / {v.loc[m,"correlation"]:.4f}</td>')
            index.append('<tr><td>'+station+'</td>'+''.join(entries)+f'<td><a href="{station}.pdf">PDF</a> <a href="{station}.png">PNG</a></td></tr>')
    index.append('</table>');(dest/'index.html').write_text('\n'.join(index));print('All 102 small-event station plots complete')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--export',type=Path);p.add_argument('--packet',type=Path);p.add_argument('--external',type=Path);p.add_argument('--review',type=Path);p.add_argument('--metrics',type=Path);a=p.parse_args()
    if a.export:export(a.root,a.export)
    else:plot(a.packet,a.external,a.review,a.metrics)
