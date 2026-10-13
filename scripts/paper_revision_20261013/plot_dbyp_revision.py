"""Verify current predictions and generate the manuscript and station atlas.

Inputs are the private, hash-verified packet from collect_dbyp_revision_20261013.
Public outputs contain aggregate tables, diagnostic figures and provenance only.
"""
import argparse, csv, hashlib, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score

METHODS=['li2021_cnn','gan2021_cnn','gong_cnn_bilstm','gan_cnn','deeprfqc','hegaz_capsule','chen2026_image','xiong2025_fcm','reference_ag3','reference_multifilter']
LABELS=dict(zip(METHODS,['Li-CNN','Gan-CNN (2021)','Gong-CNN–BiLSTM','Gan-CNN (2023)','DeepRFQC','RF-Capsule','Chen-AlexNet','Xiong-FCM','Reference–AG3','Reference–multi-filter']))
LABELS.update(all='All inputs',manual='Manual good')
COLORS=dict(zip(METHODS,['#2166ac','#b2182b','#4d9221','#762a83','#e08214','#35978f','#9970ab','#a6611a','#d73027','#4575b4']))
COLORS.update(all='#999999',manual='black')
GROUPS=[METHODS[:4],METHODS[4:8],METHODS[8:]]
EXAMPLES=['DB_EW27','YP_NE8A','YP_NE69']
SEEDS=[20260928,20260929,20260930]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def metrics(y,p,pred):
    tn,fp,fn,tp=np.bincount(2*y.astype(int)+pred.astype(int),minlength=4);n=len(y)
    ratio=lambda a,b:float(a/b) if b else 0.
    order=np.argsort(-p,kind='stable');s=p[order];truth=y[order];ends=np.r_[np.flatnonzero(np.diff(s)),n-1];cum=truth.cumsum()[ends];counts=ends+1
    ap=float(np.sum((cum/counts)*np.diff(np.r_[0,cum])/y.sum())) if y.sum() else None
    return dict(accuracy=ratio(tp+tn,n),macro_f1=ratio(tp,2*tp+fp+fn)+ratio(tn,2*tn+fp+fn),balanced_accuracy=.5*(ratio(tp,tp+fn)+ratio(tn,tn+fp)),good_precision=ratio(tp,tp+fp),good_recall=ratio(tp,tp+fn),bad_recall=ratio(tn,tn+fp),good_auprc=ap,auroc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,n=n,tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def primary(packet,source,figures,tables):
    receipt=json.loads((packet/'collection_verification.json').read_text())
    for name,digest in receipt['sha256'].items():assert sha(packet/name)==digest,name
    ids=np.load(packet/'cache/supervised/test/sample_ids.npy');y=np.load(packet/'cache/supervised/test/labels.npy');stations=np.load(packet/'cache/supervised/test/stations.npy')
    assert len(ids)==27823 and len(set(ids))==len(ids) and y.sum()==2784 and len(set(stations))==32
    rows=[];stationrows=[];preds={};bundlemeta=[]
    for method in METHODS:
        for seed in SEEDS:
            folder=packet/'runs'/f'{method}_seed{seed}';c=json.loads((folder/'completion.json').read_text());assert c['method']==method and c['seed']==seed
            for name,digest in c['sha256'].items():
                if name!='weights.safetensors':assert sha(folder/name)==digest
            d=pd.read_csv(folder/'test_predictions.csv');assert np.array_equal(d.sample_id,ids) and np.array_equal(d.label,y) and np.array_equal(d.station,stations)
            p=d.p_good.to_numpy();pred=d.prediction.to_numpy();b=json.loads((folder/'bundle.json').read_text());threshold=b['threshold'];assert np.array_equal(pred,p>=threshold)
            r=metrics(y,p,pred);old=json.loads((folder/'test_metrics.json').read_text())
            for key in old:
                if key in r and old[key] is not None:assert abs(r[key]-old[key])<1e-10,(method,seed,key,r[key],old[key])
            rows.append(dict(method=method,seed=seed,threshold=threshold,**r));preds[(method,seed)]=pred
            bundlemeta.append(dict(method=method,seed=seed,threshold=threshold,bundle_sha256=sha(folder/'bundle.json'),weights_sha256=b.get('weights_sha256'),preprocessing=b['preprocessing']))
            for station in sorted(set(stations)):
                keep=stations==station;stationrows.append(dict(method=method,seed=seed,station=station,**metrics(y[keep],p[keep],pred[keep])))
    pd.DataFrame(rows).to_csv(source/'seed_metrics.csv',index=False);pd.DataFrame(stationrows).to_csv(source/'station_metrics.csv',index=False)
    d=pd.DataFrame(rows);keys=['accuracy','macro_f1','balanced_accuracy','good_auprc','good_precision','good_recall','bad_recall','auroc'];summary=[]
    for method in METHODS:
        v=d[d.method==method];r=dict(method=method,input='seven filter views' if method=='reference_multifilter' else 'AG3',seeds=3)
        for key in keys:r[key+'_mean']=float(v[key].mean());r[key+'_sample_sd']=float(v[key].std(ddof=1))
        summary.append(r)
    s=pd.DataFrame(summary).set_index('method');s.to_csv(source/'method_summary.csv')
    text=['\\begin{tabular}{llrrrrrr}\\toprule','Configuration & Input & Accuracy & Macro-$F_1$ & BA & Good AP & Good precision & Good recall\\\\\\midrule']
    for method in METHODS:
        cells=[]
        for key in keys[:6]:
            cell=f"{100*s.loc[method,key+'_mean']:.2f}\\,$\\pm$\\,{100*s.loc[method,key+'_sample_sd']:.2f}"
            if method==s[key+'_mean'].idxmax():cell='\\textbf{'+cell+'}'
            cells.append(cell)
        text.append(LABELS[method].replace('–','--')+' & '+('7 views' if method=='reference_multifilter' else 'AG3')+' & '+' & '.join(cells)+'\\\\')
    text.append('\\bottomrule\\end{tabular}');(tables/'dbyp_primary.tex').write_text('\n'.join(text)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,5.7),sharey=True)
    for ax,key,title in zip(axes,['macro_f1','good_auprc'],['Macro-F1 (%)','Good average precision (%)']):
        for i,method in enumerate(METHODS):ax.errorbar(100*s.loc[method,key+'_mean'],i,xerr=100*s.loc[method,key+'_sample_sd'],fmt='o',color=COLORS[method],capsize=3)
        ax.set_xlabel(title);ax.grid(axis='x',alpha=.2);ax.set_xlim(35,100)
    axes[0].set_yticks(range(10),[LABELS[m]+(' [7 views]' if m=='reference_multifilter' else ' [AG3]') for m in METHODS]);axes[0].invert_yaxis();fig.tight_layout();save(fig,figures/'dbyp_primary')
    paired=[]
    for seed in SEEDS:
        a=d[(d.method=='reference_multifilter')&(d.seed==seed)].iloc[0];b=d[(d.method=='reference_ag3')&(d.seed==seed)].iloc[0]
        paired.append(dict(seed=seed,**{key+'_delta_pp':100*(a[key]-b[key]) for key in keys}))
    pd.DataFrame(paired).to_csv(source/'single_multi_paired.csv',index=False)
    # Cluster resampling: pooled counts, fixed weights and thresholds.
    rng=np.random.default_rng(20261013);unique=np.unique(stations);draws=rng.integers(0,len(unique),(10000,len(unique)));mult=np.stack([(draws==i).sum(1) for i in range(len(unique))],1);boot=[]
    for first,second in [('reference_multifilter','reference_ag3'),('gong_cnn_bilstm','reference_multifilter')]:
        for seed in SEEDS:
            scores=[]
            for method in [first,second]:
                codes=2*y+preds[(method,seed)];counts=np.array([np.bincount(codes[stations==st],minlength=4) for st in unique]);tn,fp,fn,tp=(mult@counts).T
                scores.append(np.divide(tp,2*tp+fp+fn,out=np.zeros(10000),where=(2*tp+fp+fn)>0)+np.divide(tn,2*tn+fp+fn,out=np.zeros(10000),where=(2*tn+fp+fn)>0))
            lo,median,hi=np.percentile(100*(scores[0]-scores[1]),[2.5,50,97.5]);boot.append(dict(first=first,second=second,training_seed=seed,lower_pp=lo,median_pp=median,upper_pp=hi,resampling_seed=20261013,draws=10000,stations=32))
    pd.DataFrame(boot).to_csv(source/'station_bootstrap.csv',index=False)
    for name in ['protocol.json','station_metadata.json']:shutil.copy2(packet/name,source/name)
    dump(source/'training_verification.json',dict(status='passed',runs=30,remote_verified_source_files=receipt['source_files_verified'],same_test_ids_labels_order=True,test_records=len(y),test_good=int(y.sum()),stations=32,independent_metrics='Confusion counts and grouped-threshold AP; AUROC via sklearn',bundle_provenance=bundlemeta,collection_verification_sha256=sha(packet/'collection_verification.json')))
    return s

def save(fig,path):
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight');plt.close(fig)

def stacks(q,norm='peak'):
    w=q['waveforms'].astype(float)
    if norm=='peak':w/=np.maximum(np.abs(w).max(1,keepdims=True),1e-100)
    return {str(m):w[mask].mean(0) for m,mask in zip(q['methods'],q['selected']) if mask.any()}

def stack_figure(q,station,norm='peak'):
    s=stacks(q,norm);counts={str(m):int(mask.sum()) for m,mask in zip(q['methods'],q['selected'])};fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for ax,group in zip(axes,GROUPS):
        for method in ['manual']+group:
            if method in s:ax.plot(q['time'],s[method],lw=1.3 if method=='manual' else 1,color=COLORS[method],label=f'{LABELS[method]} (n={counts[method]})')
        ax.set_xlim(-2.5,20);ax.axvline(0,color='.6',ls=':',lw=.6);ax.grid(alpha=.15);ax.set_xlabel('Time after direct P (s)');ax.legend(fontsize=7,loc='best')
    axes[0].set_ylabel('Mean peak-normalized RF' if norm=='peak' else 'Mean raw RF amplitude');fig.suptitle(station+' — '+('per-record peak normalization' if norm=='peak' else 'original amplitudes'));fig.tight_layout();return fig

def wiggle(ax,w,t,baz):
    use=(t>=-2.5-1e-7)&(t<=20+1e-7);t=t[use];x=w[:,use].astype(float);x/=np.maximum(np.abs(w).max(1,keepdims=True),1e-100);x*=4.5
    order=np.argsort(baz,kind='stable');x=x[order];baz=baz[order]
    line=[];positive=[];negative=[]
    for y,base in zip(x,baz):
        line.append(np.column_stack([t,y+base]));positive.append(np.column_stack([np.r_[t,t[::-1]],np.r_[base+np.maximum(y,0),np.repeat(base,len(t))]]));negative.append(np.column_stack([np.r_[t,t[::-1]],np.r_[base+np.minimum(y,0),np.repeat(base,len(t))]]))
    for polygons,color in [(positive,'black'),(negative,'#cb181d')]:ax.add_collection(PolyCollection(polygons,facecolors=color,edgecolors='none',rasterized=True))
    ax.add_collection(LineCollection(line,colors='black',linewidths=.22,rasterized=True));ax.set_xlim(-2.5,20);ax.set_ylim(-5,365);ax.set_yticks(np.arange(0,361,60));ax.grid(alpha=.18);ax.axvline(0,color='.5',ls=':',lw=.5)

def baz_figure(q,station,paper=False):
    methods=['all','manual']+METHODS;mask=dict(zip(q['methods'],q['selected']));s=stacks(q)
    cols=3 if paper else 6;rows=4 if paper else 2
    fig=plt.figure(figsize=(10,12) if paper else (18,12));outer=fig.add_gridspec(rows,cols,hspace=.55 if paper else .28,wspace=.2)
    for i,method in enumerate(methods):
        sub=outer[i//cols,i%cols].subgridspec(2,1,height_ratios=[1,8],hspace=.04);top=fig.add_subplot(sub[0]);ax=fig.add_subplot(sub[1],sharex=top)
        keep=mask[method];valid=keep&np.isfinite(q['baz']);n=valid.sum()
        top.set_title(f'{chr(65+i)}  {LABELS[method]}\nn={keep.sum()}',fontsize=11 if paper else 9)
        if method in s:top.plot(q['time'],s[method],color='black',lw=.9)
        else:top.text(.5,.5,'No retained RF',ha='center',transform=top.transAxes,fontsize=7)
        top.set_xlim(-2.5,20);top.tick_params(labelbottom=False);top.set_yticks([])
        if n:wiggle(ax,q['waveforms'][valid],q['time'],q['baz'][valid])
        else:ax.set_xlim(-2.5,20);ax.set_ylim(-5,365);ax.text(.5,.5,'No valid RF',transform=ax.transAxes,ha='center')
        if i%cols==0:ax.set_ylabel('Back-azimuth (°)')
        else:ax.tick_params(labelleft=False)
        if i>=(rows-1)*cols:ax.set_xlabel('Time (s)')
        ax.tick_params(labelsize=8);ax.set_xticks([0,5,10,15,20])
    fig.suptitle(station+' — AG3 catalogues ordered by back-azimuth' if paper else station+' — AG3; all traces in each retained catalogue, true SAC back-azimuth',fontsize=14,y=.995);return fig

def hk_figure(packet,station,paper=False):
    methods=['all','manual']+METHODS;p=packet/'results/test'/station
    with np.load(p/'hk_surfaces.npz') as f:h=f['h'];k=f['kappa'];surfaces=dict(zip(f['keys'],f['surfaces']))
    rows_data=pd.read_csv(p/'hk_results.csv');rows_data=rows_data[(rows_data.vp==6.3)&(rows_data.amplitude=='raw')&(rows_data.kernel=='linear')&(rows_data.fraction==1)].set_index('method')
    cols=3 if paper else 4;rows=4 if paper else 3
    fig,axes=plt.subplots(rows,cols,figsize=(10,12) if paper else (14,11),sharex=True,sharey=True)
    for i,(method,ax) in enumerate(zip(methods,axes.flat)):
        r=rows_data.loc[method];ax.set_title(f'{chr(65+i)}  {LABELS[method]}\nn={int(r.n_rf)}',fontsize=11 if paper else 9);key=method+'|6.3|raw|linear'
        if key in surfaces:
            surface=surfaces[key];spread=surface.max()-surface.min();display=(surface-surface.min())/spread if spread else surface*0
            ax.pcolormesh(h,k,display.T,cmap='viridis',vmin=0,vmax=1,shading='auto',rasterized=True);ax.plot(r.h_km,r.kappa,'+',color='red',ms=10,mew=2)
            ax.text(.02,.03,f'H={r.h_km:.1f} km, κ={r.kappa:.2f}'+('\nBoundary peak' if r.boundary==True else ''),transform=ax.transAxes,fontsize=8,color='white',bbox=dict(facecolor='black',alpha=.5,edgecolor='none'))
        else:ax.text(.5,.5,'Fewer than 10 valid RFs\nNo H–κ estimate',ha='center',transform=ax.transAxes,fontsize=9)
        ax.set_xlim(20,60);ax.set_ylim(1.5,2);ax.set_xticks([20,30,40,50,60]);ax.set_yticks([1.5,1.6,1.7,1.8,1.9,2]);ax.tick_params(labelsize=8)
        if i%cols==0:ax.set_ylabel('κ = Vp/Vs')
        if i>=(rows-1)*cols:ax.set_xlabel('H (km)')
    fig.suptitle(station+' — conditional H–κ, Vp=6.3 km/s, weights 0.7/0.2/0.1\nEach surface independently displayed from its minimum (0) to maximum (1); colors are not absolute amplitudes',fontsize=12);fig.tight_layout(rect=[0,0,1,.95]);return fig

def atlas(packet,review,figures):
    review.mkdir(parents=True,exist_ok=True);index=['<meta charset="utf-8"><title>DB/YP station review</title><h1>All 32 held-out stations</h1><p>Frozen seed 20260928; waveforms only, scratch training. No station excluded by diagnostic quality. H–κ requires geological review.</p><table><tr><th>Station</th><th>Stacks</th><th>Raw stacks</th><th>Back-azimuth</th><th>H–κ</th></tr>']
    with PdfPages(review/'all_stations_stacks.pdf') as sp, PdfPages(review/'all_stations_backazimuth.pdf') as bp, PdfPages(review/'all_stations_hk.pdf') as hp:
        for p in sorted((packet/'results/test').glob('*/plot_input.npz')):
            station=p.parent.name;dest=review/station;dest.mkdir(exist_ok=True)
            with np.load(p) as f:q=dict(f)
            fs=stack_figure(q,station);sp.savefig(fs);save(fs,dest/'stacks_peak');save(stack_figure(q,station,'raw'),dest/'stacks_raw')
            fb=baz_figure(q,station);bp.savefig(fb);save(fb,dest/'backazimuth')
            fh=hk_figure(packet,station);hp.savefig(fh);save(fh,dest/'hk')
            if station in EXAMPLES:
                save(baz_figure(q,station,paper=True),figures/f'dbyp_baz_{station}');save(hk_figure(packet,station,paper=True),figures/f'dbyp_hk_{station}')
            shutil.copy2(p.parent/'hk_results.csv',dest/'hk_results.csv');index.append('<tr><td>'+station+'</td>'+''.join(f'<td><a href="{station}/{name}.pdf">PDF</a> <a href="{station}/{name}.png">PNG</a></td>' for name in ['stacks_peak','stacks_raw','backazimuth','hk'])+'</tr>');print('atlas',station,flush=True)
    index.append('</table>');(review/'index.html').write_text('\n'.join(index))
    fig,axes=plt.subplots(3,3,figsize=(15,11),sharex=True)
    for row,station in enumerate(EXAMPLES):
        with np.load(packet/'results/test'/station/'plot_input.npz') as f:q=dict(f)
        s=stacks(q);counts=dict(zip(q['methods'],q['selected'].sum(1)))
        for col,group in enumerate(GROUPS):
            ax=axes[row,col]
            for method in ['manual']+group:
                if method in s:ax.plot(q['time'],s[method],color=COLORS[method],lw=1.4 if method=='manual' else 1,label=f'{LABELS[method]} (n={counts[method]})')
            ax.set_xlim(-2.5,20);ax.grid(alpha=.2);ax.legend(fontsize=6.5);ax.set_title(f'{chr(65+3*row+col)}  {station}',loc='left');ax.axvline(0,color='.6',ls=':',lw=.5)
            if col==0:ax.set_ylabel('Mean peak-normalized RF')
            if row==2:ax.set_xlabel('Time after direct P (s)')
    fig.tight_layout();save(fig,figures/'dbyp_stack_examples')

def dataset(packet,figures):
    with np.load(packet/'dataset_examples.npz') as f:q=dict(f)
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for i,ax in enumerate(axes.flat):
        w=q['waveforms'][i].astype(float);gain=.75/max(np.abs(w).max(),1e-100)
        for j,(trace,g) in enumerate(zip(w,q['gaussians'][i])):ax.plot(q['time'],trace*gain+j,color='black',lw=.7)
        ax.set_yticks(range(7),[f'a={g:g}' for g in q['gaussians'][i]]);ax.set_xlim(-10,40);ax.axvline(0,color='red',ls=':',lw=.8);ax.set_title(f'{chr(65+i)}  {q["stations"][i]} — '+('manual good' if q['labels'][i] else 'manual bad'),fontsize=11);ax.text(.98,.98,f'max |RF|={np.abs(w).max():.3g}',ha='right',va='top',transform=ax.transAxes,fontsize=8);ax.grid(alpha=.15)
        if i>=2:ax.set_xlabel('Time after direct P (s)')
    fig.tight_layout();save(fig,figures/'dbyp_dataset_examples')

def summaries(packet,diag,source,figures,tables,review):
    for name in ['test_stack_metrics','test_cross_filter','test_cross_filter_pairs','test_azimuth','small_stack_metrics']:
        shutil.copy2(diag/(name+'.csv'),source/(name+'.csv'))
    shutil.copy2(diag/'verification.json',source/'diagnostic_verification.json')
    d=pd.read_csv(source/'test_stack_metrics.csv');full=d[(d.fraction==1)&d.eligible];summ=[]
    for (method,norm,fraction),g in d[d.eligible].groupby(['method','normalization','fraction']):
        # Repeated draws receive one station-level median, not 30 independent stations.
        fields=['correlation','nrmse','delta_maximum_amplitude','delta_maximum_time','delta_p_amplitude','delta_p_time','delta_ps_candidate_amplitude','delta_ps_candidate_time','p_amplitude_error_pct','ps_candidate_amplitude_error_pct']
        v=g.groupby('station')[fields].median();r=dict(method=method,normalization=norm,fraction=fraction,n_stations=len(v))
        for key in fields:r[key+'_median']=float(v[key].median());r[key+'_q25']=float(v[key].quantile(.25));r[key+'_q75']=float(v[key].quantile(.75))
        summ.append(r)
    pd.DataFrame(summ).to_csv(source/'stack_summary.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(14,8));fields=['correlation','nrmse','delta_maximum_time','delta_p_time','delta_ps_candidate_time','ps_candidate_amplitude_error_pct'];norms=['peak','raw','raw','raw','raw','raw'];titles=['Shape correlation [−2.5,20] s','Raw-stack NRMSE','|Maximum-time difference| (s)','|P-window time difference| (s)','|Ps-candidate time difference| (s)','Ps-candidate amplitude error (%)']
    for ax,key,norm,title in zip(axes.flat,fields,norms,titles):
        values=[]
        for m in METHODS:
            v=full[(full.method==m)&(full.normalization==norm)][key].dropna().to_numpy();values.append(v if key in ['correlation','nrmse'] else np.abs(v))
        ax.boxplot(values,orientation='horizontal',tick_labels=[LABELS[m] for m in METHODS],showfliers=True);ax.invert_yaxis();ax.tick_params(labelsize=7);ax.set_xlabel(title,fontsize=9);ax.grid(axis='x',alpha=.2)
        if key in ['nrmse','ps_candidate_amplitude_error_pct']:
            ax.set_xscale('symlog',linthresh=1);ax.set_xlim(left=0)
    fig.tight_layout();save(fig,figures/'dbyp_stack_diagnostics')
    hk=[]
    for p in sorted((packet/'results/test').glob('*/hk_results.csv')):
        v=pd.read_csv(p);v.insert(0,'station',p.parent.name);hk.append(v)
    hk=pd.concat(hk,ignore_index=True);hk.to_csv(source/'hk_all_results.csv',index=False)
    current=hk[(hk.vp==6.3)&(hk.amplitude=='raw')&(hk.kernel=='linear')&(hk.fraction==1)];current.to_csv(source/'hk_primary_review.csv',index=False);current.to_csv(review/'hk_primary_review.csv',index=False)
    summaries=[]
    for m in METHODS:
        v=current[(current.method==m)&(current.status=='ok')&current.delta_h_vs_manual.notna()];half=hk[(hk.method==m)&(hk.vp==6.3)&(hk.amplitude=='raw')&(hk.kernel=='linear')&(hk.fraction==.5)&hk.delta_h_vs_manual.notna()]
        medhalf=half.groupby('station')[['delta_h_vs_manual','delta_kappa_vs_manual']].agg(lambda x:np.median(np.abs(x)))
        summaries.append(dict(method=m,valid_manual_pairs=len(v),median_abs_delta_h_km=float(np.median(np.abs(v.delta_h_vs_manual))),median_abs_delta_kappa=float(np.median(np.abs(v.delta_kappa_vs_manual))),boundary_peaks=int(v.boundary.sum()),multiple_peak_flags=int((v.separated_peaks_95pct>1).sum()),half_paired_stations=len(medhalf),half_median_abs_delta_h_km=float(medhalf.delta_h_vs_manual.median()),half_median_abs_delta_kappa=float(medhalf.delta_kappa_vs_manual.median())))
    pd.DataFrame(summaries).to_csv(source/'hk_method_summary.csv',index=False)
    small=pd.read_csv(source/'small_stack_metrics.csv');small=small[(small.fraction==1)&(small.normalization=='peak')];small.to_csv(review/'small_station_review.csv',index=False)
    summary=[]
    for m,g in small[small.method.isin(METHODS)].groupby('method'):
        summary.append(dict(method=m,n_input=int(g.n_input.sum()),retained=int(g.n_selected.sum()),stations=len(g),stations_with_3_rf=int((g.n_selected>=3).sum()),median_correlation=float(g[g.eligible].correlation.median()),n_eligible_stations=int(g.eligible.sum())))
    pd.DataFrame(summary).to_csv(source/'small_method_summary.csv',index=False)
    dump(source/'diagnostic_summary.json',dict(full_test_stations=32,manual_stack_eligible=int(full[(full.method=='manual')&(full.normalization=='raw')].station.nunique()),hk_manual_valid=int(((current.method=='manual')&(current.status=='ok')).sum()),window=[-2.5,20],small_n=int(small[small.method=='all'].n_input.sum()),small_stations=int(small.station.nunique())))

def timing(timing,source,figures,tables,summary):
    if not timing.exists():return
    target=source/'timing';shutil.copytree(timing,target,dirs_exist_ok=True);manifest=json.loads((target/'verification.json').read_text());assert manifest['rounds']==30
    for name,digest in manifest['sha256'].items():assert sha(target/name)==digest
    rows=[json.loads((target/name).read_text()) for name in manifest['sha256']];assert len({(r['method'],r['round']) for r in rows})==30
    independent=[r for r in rows if r['method']!='xiong2025_fcm'];assert len({r['sample_ids_sha256'] for r in independent})==1
    values=[]
    for method in METHODS:
        group=[r for r in rows if r['method']==method];r=dict(method=method,parameters=group[0]['parameters'])
        for key in ['latency_p50_ms','latency_p95_ms','batch_records_per_second','forward_batch_records_per_second','pipeline_peak_gpu_allocated_bytes','pool_seconds','pool_records_per_second']:
            v=[x[key] for x in group if x.get(key) is not None]
            if v:r[key]=float(np.median(v));r[key+'_min']=float(min(v));r[key+'_max']=float(max(v))
        values.append(r)
    d=pd.DataFrame(values).set_index('method');d.to_csv(source/'inference_summary.csv');text=['\\begin{tabular}{lrrrrr}\\toprule','Configuration & Params & p50 (ms) & p95 (ms) & API (RF/s) & GPU MiB\\\\\\midrule']
    for method in METHODS:
        r=d.loc[method]
        cells=[f'{r.parameters:,.0f}']+(['---']*4 if method=='xiong2025_fcm' else [f'{r.latency_p50_ms:.2f}',f'{r.latency_p95_ms:.2f}',f'{r.batch_records_per_second:,.0f}',f'{r.pipeline_peak_gpu_allocated_bytes/2**20:.1f}'])
        text.append(LABELS[method].replace('–','--')+' & '+' & '.join(cells)+'\\\\')
    text.append('\\bottomrule\\end{tabular}');(tables/'dbyp_inference.tex').write_text('\n'.join(text)+'\n')
    fig,ax=plt.subplots(figsize=(8,5))
    for method in METHODS:
        if method=='xiong2025_fcm':continue
        ax.errorbar(d.loc[method,'batch_records_per_second'],100*summary.loc[method,'macro_f1_mean'],yerr=100*summary.loc[method,'macro_f1_sample_sd'],fmt='o',color=COLORS[method],label=LABELS[method])
        right=d.loc[method,'batch_records_per_second']>10000
        ax.annotate(LABELS[method],(d.loc[method,'batch_records_per_second'],100*summary.loc[method,'macro_f1_mean']),xytext=(-6,5) if right else (4,3),ha='right' if right else 'left',textcoords='offset points',fontsize=7)
    ax.set_xscale('log');ax.set_xlabel('Batch-32 prediction API throughput (RF/s)');ax.set_ylabel('Three-seed mean macro-F1 (%)');ax.grid(alpha=.2);fig.tight_layout();save(fig,figures/'dbyp_inference')

def main(a):
    source=a.paper/'SourceData/revision_20261013';figures=a.paper/'figures';tables=a.paper/'tables';source.mkdir(parents=True,exist_ok=True)
    summary=primary(a.packet,source,figures,tables);dataset(a.packet,figures)
    a.review.mkdir(parents=True,exist_ok=True);summaries(a.packet,a.diagnostics,source,figures,tables,a.review)
    timing(a.timing,source,figures,tables,summary)
    if not a.skip_atlas:atlas(a.packet,a.review,figures)
    dump(source/'analysis_verification.json',dict(status='passed',script_sha256=sha(Path(__file__)),current_method_count=10,literature_methods=8,reference_methods=2,seed_count=3,training_runs=30,predictions_checked=834690,review_path=str(a.review),source_path=str(a.packet),diagnostics_path=str(a.diagnostics)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--diagnostics',type=Path,required=True);p.add_argument('--timing',type=Path,required=True);p.add_argument('--paper',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--skip-atlas',action='store_true');main(p.parse_args())
