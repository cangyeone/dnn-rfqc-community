"""Recompute frozen-catalogue diagnostics; never fit models or thresholds.

Run on the DB/YP v2 experiment host. Output is aggregate source data. RF arrays
and identifiers remain in the private frozen experiment directory.
"""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from threadpoolctl import threadpool_limits

WINDOW=(-2.5,20.)
SEED=20261009

def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def corr(a,b):
    a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(a@b/den,-1,1)) if den>0 else None

def measurements(w,t):
    out={}
    for prefix,limits in [('maximum',WINDOW),('p',(-.3,1.6)),('ps_candidate',(2.5,7.5))]:
        use=np.flatnonzero((t>=limits[0]-1e-7)&(t<=limits[1]+1e-7))
        fallback=False
        if prefix=='ps_candidate':
            v=w[use];peaks=np.unique(np.r_[find_peaks(v)[0],find_peaks(-v)[0]])
            if len(peaks):use=use[peaks]
            else:fallback=True
        i=use[np.argmax(w[use])]
        out[prefix+'_amplitude']=float(w[i]);out[prefix+'_time']=float(t[i])
        if prefix=='ps_candidate':out['ps_candidate_fallback']=fallback
    return out

def stack_rows(station,waves,t,ids,selections,external=None,fcm=None):
    from hk_core import _ranks
    window=(t>=WINDOW[0]-1e-7)&(t<=WINDOW[1]+1e-7)
    peak=waves/np.maximum(np.abs(waves).max(1,keepdims=True),1e-100)
    rows=[]
    for fraction in ([1.,.5,.25] if external is None else [1.]):
        for draw in range(1 if fraction==1 else 30):
            indices=np.argsort(_ranks(ids,SEED+draw,station),kind='stable')[:int(len(ids)*fraction)]
            masks={k:v[indices] for k,v in selections.items()}
            if fraction!=1 and fcm is not None:
                masks['xiong2025_fcm']=fcm(waves[indices],station)
            for norm,data in [('raw',waves),('peak',peak)]:
                manual=data[indices[masks['manual']]].mean(0) if 'manual' in masks and masks['manual'].any() else (external or {}).get(norm)
                nmanual=int(masks['manual'].sum()) if 'manual' in masks else (external or {}).get('n',0)
                ref=measurements(manual,t) if manual is not None else {}
                for method,mask in masks.items():
                    keep=indices[mask];r=dict(station=station,method=method,normalization=norm,fraction=fraction,draw=draw,n_input=len(indices),n_selected=len(keep),n_manual=nmanual,window_start=WINDOW[0],window_end=WINDOW[1],eligible=len(keep)>=3 and nmanual>=3)
                    if len(keep):
                        stack=data[keep].mean(0);r.update(measurements(stack,t))
                        if manual is not None:
                            a=stack[window];b=manual[window];den=np.sqrt(np.mean(b*b))
                            r.update(correlation=corr(a,b),nrmse=float(np.sqrt(np.mean((a-b)**2))/den) if den>0 else None)
                            for prefix in ['maximum','p','ps_candidate']:
                                for field in ['amplitude','time']:r['delta_'+prefix+'_'+field]=r[prefix+'_'+field]-ref[prefix+'_'+field]
                                r[prefix+'_amplitude_error_pct']=100*abs(r['delta_'+prefix+'_amplitude'])/abs(ref[prefix+'_amplitude']) if ref[prefix+'_amplitude']!=0 else None
                    rows.append(r)
    return rows

def coherence(station,z,ids,selected):
    from hk_core import _ranks
    t=np.arange(501)*.1-10;win=(t>=WINDOW[0]-1e-7)&(t<=WINDOW[1]+1e-7)
    x=z['waveforms'][...,win].astype(float);x/=np.maximum(np.abs(x).max(-1,keepdims=True),1e-100);x-=x.mean(-1,keepdims=True)
    norms=np.linalg.norm(x,axis=-1);x/=np.maximum(norms[...,None],1e-100)
    valid=norms>1e-15;pairs=[];pairrows=[];rank=_ranks(ids,SEED,station);order=np.argsort(rank,kind='stable')
    for i in range(7):
        for j in range(i+1,7):
            v=np.clip((x[:,i]*x[:,j]).sum(-1),-1,1);v[~(valid[:,i]&valid[:,j])]=np.nan;pairs.append(v)
            for method,mask in selected.items():
                use=v[mask];pairrows.append(dict(station=station,method=method,left=float(z['gaussians'][0,i]),right=float(z['gaussians'][0,j]),n=int(np.isfinite(use).sum()),correlation=float(np.nanmean(use)) if np.isfinite(use).any() else None))
    per_record=np.nanmean(pairs,axis=0);cross=[]
    for method,mask in selected.items():
        n=int(mask.sum());v=per_record[mask];control=per_record[order[:n]]
        cross.append(dict(station=station,method=method,n_selected=n,correlation=float(np.nanmean(v)) if n else None,count_matched_random_correlation=float(np.nanmean(control)) if n else None,window_start=WINDOW[0],window_end=WINDOW[1]))
    ag3=(z['gaussians']==3).argmax(1);signal=x[np.arange(len(x)),ag3];baz=np.mod(z['baz'],360);az=[]
    def meanpair(use):
        n=len(use)
        return float(np.clip((np.sum(signal[use].sum(0)**2)-n)/(n*(n-1)),-1,1)) if n>1 else None
    for width in [15,30,45]:
        for low in range(0,360,width):
            pool=np.flatnonzero(np.isfinite(baz)&valid[np.arange(len(x)),ag3]&(baz>=low)&(baz<low+width));ordered=pool[np.argsort(rank[pool],kind='stable')]
            for method,mask in selected.items():
                keep=pool[mask[pool]];az.append(dict(station=station,method=method,width_deg=width,baz_low=low,n_input=len(pool),n_selected=len(keep),correlation=meanpair(keep),count_matched_random_correlation=meanpair(ordered[:len(keep)])))
    return cross,pairrows,az

def verify_hk(station,z,ids,selections,source):
    from hk_core import HKData,_contributions,_peak,_ranks
    p=z['p'];valid=np.isfinite(p)&(p>0)&(p<1/6.6)&np.isfinite(z['original_ag3']).all(1)&np.any(z['original_ag3']!=0,axis=1)
    data=HKData(z['original_ag3'][valid],z['original_time'],p[valid],ids[valid],np.repeat(station,valid.sum()),'radial','P','uncorrected','SAC user4 s/km, provider confirmed')
    h,k,terms=_contributions(data,np.arange(20.,60.01,.5),np.arange(1.5,2.001,.01),6.3,(.7,.2,.1),'linear','raw')
    with np.load(source/'hk_surfaces.npz') as f:surfaces=dict(zip(f['keys'],f['surfaces']))
    with (source/'hk_results.csv').open() as f:old=list(csv.DictReader(f))
    old={(r['method'],float(r['fraction']),int(r['draw'])):r for r in old if float(r['vp'])==6.3 and r['amplitude']=='raw' and r['kernel']=='linear'}
    descriptors=[];matrix=[];max_error=0.
    for fraction in [1.,.5]:
        for draw in range(1 if fraction==1 else 100):
            ranks=_ranks(ids[valid],SEED+draw,station)
            for method,mask in selections.items():
                eligible=np.flatnonzero(mask[valid]);n=int(len(eligible)*fraction);r=old[(method,fraction,draw)];assert int(r['n_rf'])==n
                if n<10:assert r['status']=='insufficient_rf';continue
                keep=eligible[np.argsort(ranks[eligible],kind='stable')[:n]];m=np.zeros(valid.sum());m[keep]=1/n
                matrix.append(m);descriptors.append((method,fraction,draw,r))
    values=(np.asarray(matrix)@terms.reshape(valid.sum(),-1)).reshape(-1,len(h),len(k))
    for s,(method,fraction,draw,r) in zip(values,descriptors):
        peak=_peak(s,h,k,int(r['n_rf']));assert peak['h_km']==float(r['h_km']) and abs(peak['kappa']-float(r['kappa']))<1e-12
        if fraction==1:
            error=float(np.max(np.abs(s-surfaces[method+'|6.3|raw|linear'])));assert error<1e-10;max_error=max(max_error,error)
    return dict(station=station,status='passed',primary_peaks_verified=len(descriptors),maximum_surface_error=max_error,n_valid_p=int(valid.sum()),source_grid='Original AG3 SAC samples; per-record ray parameters')

def main(root,out):
    sys.path.insert(0,str(root/'scripts'));sys.path.insert(0,str(root/'runtime'))
    # Import equations directly: no model/network initialization is required.
    import importlib.util
    spec=importlib.util.spec_from_file_location('frozen_fcm',root/'runtime/rfqc_bench/fcm.py')
    # fcm.py has a relative io import: load through the frozen package instead.
    from rfqc_bench.fcm import features,membership
    bundle=json.loads((root/'runs/xiong2025_fcm_seed20260928/bundle.json').read_text())
    def fcm(waves,station):
        f=features(waves,np.repeat(station,len(waves)));prep=bundle['preprocessing'];par=bundle['parameters']
        x=np.clip((f-np.array(prep['minimum']))/np.array(prep['scale']),0,1)
        return membership(x,np.array(par['centers']))[:,par['good_cluster']]>=bundle['threshold']
    out.mkdir(parents=True,exist_ok=True);stacks=[];cross=[];pair=[];az=[];hkverify=[]
    for path in sorted((root/'results/test').glob('*/plot_input.npz')):
        station=path.parent.name;target=out/'stations'/station;target.mkdir(parents=True,exist_ok=True)
        if (target/'verified.json').exists():print('reused',station,flush=True);continue
        with np.load(path) as f:q=dict(f)
        selections=dict(zip(q['methods'],q['selected']));assert np.array_equal(fcm(q['waveforms'],station),selections['xiong2025_fcm'])
        rows=stack_rows(station,q['waveforms'].astype(float),q['time'],q['sample_ids'],selections,fcm=fcm);write(target/'stack_metrics.csv',rows)
        with np.load(root/'cache/complete'/f'{station}.npz') as f:
            lookup={s:i for i,s in enumerate(f['sample_ids'])};index=np.array([lookup[s] for s in q['sample_ids']])
            z={name:(f[name][index] if name!='original_time' and np.ndim(f[name])>0 and len(f[name])==len(f['lengths']) else f[name]) for name in f.files}
        c,p,a=coherence(station,z,q['sample_ids'],selections);write(target/'cross_filter.csv',c);write(target/'cross_filter_pairs.csv',p);write(target/'azimuth.csv',a)
        verification=verify_hk(station,z,q['sample_ids'],selections,path.parent)
        (target/'verified.json').write_text(json.dumps(verification,indent=2)+'\n');print('completed',station,flush=True)
    small=[]
    for path in sorted((root/'results/small').glob('*/plot_input.npz')):
        station=path.parent.name;target=out/'small'/station
        if (target/'stack_metrics.csv').exists():continue
        with np.load(path) as f:q=dict(f)
        with np.load(root/'cache/complete'/f'{station}.npz') as f:
            keep=f['common_labels']==1;w=f['waveforms'][np.arange(len(f['lengths'])),(f['gaussians']==3).argmax(1)][keep].astype(float)
            external={'n':len(w),'raw':w.mean(0),'peak':(w/np.maximum(np.abs(w).max(1,keepdims=True),1e-100)).mean(0)} if len(w) else {}
        rows=stack_rows(station,q['waveforms'].astype(float),q['time'],q['sample_ids'],dict(zip(q['methods'],q['selected'])),external=external)
        write(target/'stack_metrics.csv',rows);print('small',station,flush=True)
    for cohort in ['stations','small']:
        for name in (['stack_metrics','cross_filter','cross_filter_pairs','azimuth'] if cohort=='stations' else ['stack_metrics']):
            rows=[]
            for file in sorted((out/cohort).glob('*/'+name+'.csv')):
                with file.open() as f:rows.extend(csv.DictReader(f))
            write(out/(('test_' if cohort=='stations' else 'small_')+name+'.csv'),rows)
    checks=[json.loads(p.read_text()) for p in sorted((out/'stations').glob('*/verified.json'))]
    assert len(checks)==32
    (out/'verification.json').write_text(json.dumps(dict(status='passed',window=WINDOW,model_seed=20260928,subsample_seed=SEED,draws=30,hk_half_draws=100,test_station_checks=checks,small_stations=len(list((out/'small').iterdir())),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=4):main(a.root,a.output)
