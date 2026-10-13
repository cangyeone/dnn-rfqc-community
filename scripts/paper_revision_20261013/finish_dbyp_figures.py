"""Additional aggregate plots and tables for the October diagnostic revision."""
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from plot_dbyp_revision import METHODS,LABELS,COLORS,EXAMPLES,save,baz_figure,stacks

def main(a):
    source=a.paper/'SourceData/revision_20261013';figs=a.paper/'figures';tables=a.paper/'tables'
    d=pd.read_csv(source/'test_stack_metrics.csv');full=d[(d.fraction==1)&d.eligible];summary=pd.read_csv(source/'stack_summary.csv')
    lines=['\\begin{tabular}{lrrrrrr}\\toprule','Configuration & Retained & $r$ (shape) & Raw NRMSE & Max amp. error (\\%) & Ps amp. error (\\%) & $|\\Delta t_{\\mathrm{Ps}}|$ (s)\\\\\\midrule']
    seeds=pd.read_csv(source/'seed_metrics.csv');rows=[]
    for m in METHODS:
        p=full[(full.method==m)&(full.normalization=='peak')];r=full[(full.method==m)&(full.normalization=='raw')];counts=seeds[(seeds.method==m)&(seeds.seed==20260928)].iloc[0]
        amp=100*np.abs(r.delta_maximum_amplitude)/np.abs(r.maximum_amplitude-r.delta_maximum_amplitude)
        row=dict(method=m,retained=int(counts.tp+counts.fp),shape_correlation=float(p.correlation.median()),raw_nrmse=float(r.nrmse.median()),maximum_amplitude_error_pct=float(amp.median()),maximum_abs_time_error_s=float(np.abs(r.delta_maximum_time).median()),p_amplitude_error_pct=float(r.p_amplitude_error_pct.median()),p_abs_time_error_s=float(np.abs(r.delta_p_time).median()),ps_amplitude_error_pct=float(r.ps_candidate_amplitude_error_pct.median()),ps_abs_time_error_s=float(np.abs(r.delta_ps_candidate_time).median()),n_stations=len(r));rows.append(row)
        lines.append(LABELS[m].replace('–','--')+f' & {row["retained"]:,} & {row["shape_correlation"]:.4f} & {row["raw_nrmse"]:.3f} & {row["maximum_amplitude_error_pct"]:.2f} & {row["ps_amplitude_error_pct"]:.2f} & {row["ps_abs_time_error_s"]:.2f}\\\\')
    lines.append('\\bottomrule\\end{tabular}');(tables/'dbyp_catalogues.tex').write_text('\n'.join(lines)+'\n');pd.DataFrame(rows).to_csv(source/'catalogue_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,5.5))
    for ax,key,title in zip(axes,['delta_maximum_amplitude','delta_maximum_time'],['|Maximum-amplitude difference| (raw RF units)','|Maximum-time difference| (s)']):
        values=[np.abs(full[(full.method==m)&(full.normalization=='raw')][key].dropna().to_numpy()) for m in METHODS]
        ax.boxplot(values,orientation='horizontal',tick_labels=[LABELS[m] for m in METHODS],showfliers=True);ax.invert_yaxis();ax.set_xlabel(title);ax.tick_params(labelsize=8);ax.grid(axis='x',alpha=.2)
        if key=='delta_maximum_amplitude':ax.set_xscale('symlog',linthresh=.01);ax.set_xlim(left=0)
    fig.tight_layout();save(fig,figs/'dbyp_maximum_differences')
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,norm,key,title in [(axes[0],'peak','correlation_median','Stack shape correlation'),(axes[1],'raw','nrmse_median','Raw-stack NRMSE')]:
        for m in METHODS:
            g=summary[(summary.method==m)&(summary.normalization==norm)].sort_values('fraction',ascending=False);ax.plot(g.fraction,g[key],'o-',color=COLORS[m],label=LABELS[m])
        ax.set_xticks([1,.5,.25],['Full','Half','Quarter']);ax.invert_xaxis();ax.set_xlabel('Shared candidate pool');ax.set_ylabel(title);ax.grid(alpha=.2)
    axes[1].legend(fontsize=7,loc='upper left',bbox_to_anchor=(1,1));fig.tight_layout();save(fig,figs/'dbyp_input_subsampling')
    # Coherence: aggregate azimuth bins inside station before station medians.
    cross=pd.read_csv(source/'test_cross_filter.csv');az=pd.read_csv(source/'test_azimuth.csv');values=[]
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for ax,data,key,title in [(axes[0],cross,'correlation','Cross-filter association [−2.5,20] s'),(axes[1],az[az.width_deg==30],'correlation','Within 30° bin correlation [−2.5,20] s')]:
        if ax==axes[1]:data=data.groupby(['station','method'])[['correlation','count_matched_random_correlation']].mean().reset_index()
        x=[];controls=[]
        for m in METHODS:
            v=data[data.method==m];x.append(v[key].dropna().to_numpy());controls.append(float(v.count_matched_random_correlation.median()));values.append(dict(diagnostic='cross_filter' if ax==axes[0] else 'azimuth_30deg',method=m,station_median=float(v[key].median()),matched_random_median=controls[-1],n_stations=int(v[key].notna().sum())))
        ax.boxplot(x,orientation='horizontal',tick_labels=[LABELS[m] for m in METHODS],showfliers=True);ax.scatter(controls,np.arange(1,11),marker='x',color='.4',label='Count-matched random median');ax.invert_yaxis();ax.set_xlabel(title);ax.tick_params(labelsize=8);ax.grid(axis='x',alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();save(fig,figs/'dbyp_coherence');pd.DataFrame(values).to_csv(source/'coherence_summary.csv',index=False)
    h=pd.read_csv(source/'hk_method_summary.csv');lines=['\\begin{tabular}{lrrrrrr}\\toprule','Configuration & Paired stations & $|\\Delta H|$ (km) & $|\\Delta\\kappa|$ & Boundary peaks & Multi-peak flags & Half $|\\Delta H|$ (km)\\\\\\midrule']
    for m in METHODS:
        r=h[h.method==m].iloc[0];lines.append(LABELS[m].replace('–','--')+f' & {r.valid_manual_pairs:.0f} & {r.median_abs_delta_h_km:.2f} & {r.median_abs_delta_kappa:.3f} & {r.boundary_peaks:.0f} & {r.multiple_peak_flags:.0f} & {r.half_median_abs_delta_h_km:.2f}\\\\')
    lines.append('\\bottomrule\\end{tabular}');(tables/'dbyp_hk.tex').write_text('\n'.join(lines)+'\n')
    # The named stations were requested before this revision; keep both even
    # when one method has a lower retention count or lower stack correlation.
    fig,axes=plt.subplots(2,3,figsize=(15,7),sharex=True)
    from plot_dbyp_revision import GROUPS
    for i,station in enumerate(['YP_NE8A','YP_NE69']):
        with np.load(a.packet/'results/small'/station/'plot_input.npz') as f:q=dict(f)
        with np.load(a.packet.parent/'small_external_stacks.npz') as f:
            iref=np.flatnonzero(f['stations']==station)[0];ref=f['peak'][iref];n=int(f['counts'][iref])
        s=stacks(q);counts=dict(zip(q['methods'],q['selected'].sum(1)))
        for j,group in enumerate(GROUPS):
            ax=axes[i,j];ax.plot(q['time'],ref,color='black',lw=1.5,label=f'Main-cohort manual (n={n})')
            for m in group:
                if m in s:ax.plot(q['time'],s[m],color=COLORS[m],lw=1,label=f'{LABELS[m]} (n={counts[m]})')
                else:ax.plot([],[],color=COLORS[m],lw=1,label=f'{LABELS[m]} (n={counts[m]}; no stack)')
            ax.set_title(station+' — unlabelled small events',fontsize=10);ax.set_xlim(-2.5,20);ax.grid(alpha=.2);ax.legend(fontsize=6.5)
            if j==0:ax.set_ylabel('Mean peak-normalized RF')
            if i==1:ax.set_xlabel('Time after direct P (s)')
    fig.tight_layout();save(fig,figs/'dbyp_small_examples')
    # False acceptances/rejections use actual test truth and actual baz.
    for station in EXAMPLES:
        with np.load(a.packet/'results/test'/station/'plot_input.npz') as f:q=dict(f)
        masks=dict(zip(q['methods'],q['selected']));manual=masks['manual']
        for name,truth in [('false_acceptance',~manual),('false_rejection',manual)]:
            selected=[]
            for method in q['methods']:
                if method=='all':selected.append(truth)
                elif method=='manual':selected.append(manual)
                else:selected.append(truth&(masks[method] if name=='false_acceptance' else ~masks[method]))
            z=dict(q);z['selected']=np.array(selected)
            old_all,old_manual=LABELS['all'],LABELS['manual']
            LABELS['all']='Manual bad RFs' if name=='false_acceptance' else 'Manual good RFs'
            LABELS['manual']='Manual good reference'
            title='manual bad predicted good' if name=='false_acceptance' else 'manual good predicted bad'
            fig=baz_figure(z,station+' — '+title);save(fig,a.review/station/name)
            LABELS['all'],LABELS['manual']=old_all,old_manual
    print('Additional figures/tables complete')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--paper',type=Path,required=True);p.add_argument('--review',type=Path,required=True);main(p.parse_args())
