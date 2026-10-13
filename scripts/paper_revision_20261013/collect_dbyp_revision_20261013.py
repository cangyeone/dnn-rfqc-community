"""Collect verified completed DB/YP v2 results for the October figure revision.

Run on the experiment host. The packet is private analysis input, not a public
dataset deposit. No training, fitting or threshold selection is performed.
"""
import argparse, csv, hashlib, json, shutil, sqlite3, tarfile
from pathlib import Path
import numpy as np

METHODS=['li2021_cnn','gan2021_cnn','gong_cnn_bilstm','gan_cnn','deeprfqc',
         'hegaz_capsule','chen2026_image','reference_ag3','reference_multifilter','xiong2025_fcm']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main(root,out):
    out.mkdir(parents=True,exist_ok=False);packet=out/'inputs';packet.mkdir()
    def copy(p,target=None):
        q=packet/(target or p.relative_to(root));q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    protocol=json.loads((root/'protocol.json').read_text())
    assert protocol['version']=='dbyp_complete_20261009_v2' and protocol['methods']==METHODS
    manifest=json.loads((root/'cache/supervised/manifest.json').read_text())
    assert protocol['data_manifest_sha256']==sha(root/'cache/supervised/manifest.json')
    stations=sorted(x['station'] for x in manifest['stations'] if x['split']=='test')
    assert len(stations)==32
    expected={(m,s) for m in METHODS for s in protocol['training']['seeds']}
    assert len(expected)==30
    count=0;receipts=[]
    for method,seed in sorted(expected):
        p=root/'runs'/f'{method}_seed{seed}'
        c=json.loads((p/'completion.json').read_text())
        assert c['status']=='completed' and c['protocol_sha256']==sha(root/'protocol.json')
        for name,digest in c['sha256'].items():
            assert sha(p/name)==digest,(p,name);count+=1
            if name!='weights.safetensors':copy(p/name)
        copy(p/'completion.json');receipts.append(dict(method=method,seed=seed,completion_sha256=sha(p/'completion.json')))
    for name in ['protocol.json','training_status.json','audit/setup_validation.json','audit/inventory_summary.json',
                 'audit/data_revision_comparison.json','cache/supervised/manifest.json','cache/supervised/station_split.csv']:
        copy(root/name)
    for p in (root/'summary').iterdir():
        if p.is_file():copy(p)
    for name in ['sample_ids.npy','labels.npy','stations.npy']:
        copy(root/'cache/supervised/test'/name)
    n=0
    for station in stations:
        p=root/'results/test'/station;c=json.loads((p/'completion.json').read_text())
        assert c['seed']==20260928 and c['station']==station
        for name,digest in c['files'].items():
            if name=='hk_primary_contributions.npz':continue
            assert sha(p/name)==digest,(station,name);copy(p/name)
        copy(p/'completion.json');n+=c['n']
    assert n==27823
    # Small-data scalar diagnostics are copied for all stations. Only the two
    # previously requested held-out stations need waveform arrays for examples.
    for p in sorted((root/'results/small').iterdir()):
        if not (p/'completion.json').exists():continue
        c=json.loads((p/'completion.json').read_text());copy(p/'completion.json')
        for name in ['stack_metrics.csv','catalog_overlap.csv','cross_filter_correlation.csv','stacks.npz']:
            if (p/name).exists():assert sha(p/name)==c['files'][name];copy(p/name)
        if p.name in ['YP_NE69','YP_NE8A']:
            for name in ['plot_input.npz','predictions.npz']:
                assert sha(p/name)==c['files'][name];copy(p/name)
    # Representative good/bad examples from two training stations, selected by
    # the median observed AG3 shape measurements, never classifier performance.
    examples=[]
    for station in ['DB_EW50','YP_NE3A']:
        assert next(x for x in manifest['stations'] if x['station']==station)['split']=='train'
        with np.load(root/'cache/complete'/f'{station}.npz') as z:
            waves=z['waveforms'];ix=(z['gaussians']==3).argmax(1);w=waves[np.arange(len(ix)),ix].astype(float)
            normalized=w/np.maximum(np.abs(w).max(1,keepdims=True),1e-30)
            time=np.arange(501)*.1-10;pwin=(time>=-.3)&(time<=1.6);swin=(time>=2.5)&(time<=7.5)
            f=np.column_stack([normalized[:,pwin].max(1),normalized[:,swin].max(1),np.sqrt(np.mean(normalized[:,:71]**2,axis=1))])
            for label in [1,0]:
                eligible=np.flatnonzero((z['common_labels']==label)&np.any(w!=0,axis=1))
                v=f[eligible];centre=np.median(v,axis=0);scale=np.quantile(v,.75,axis=0)-np.quantile(v,.25,axis=0)
                selected=eligible[np.argmin(np.sum(((v-centre)/np.maximum(scale,1e-8))**2,axis=1))]
                examples.append(dict(station=station,label=label,sample_id=str(z['sample_ids'][selected]),
                                     waves=waves[selected],gaussians=z['gaussians'][selected],length=int(z['lengths'][selected])))
    np.savez_compressed(packet/'dataset_examples.npz',waveforms=np.array([e['waves'] for e in examples]),
        gaussians=np.array([e['gaussians'] for e in examples]),lengths=np.array([e['length'] for e in examples]),
        stations=np.array([e['station'] for e in examples]),labels=np.array([e['label'] for e in examples]),
        sample_ids=np.array([e['sample_id'] for e in examples]),time=np.arange(501)*.1-10)
    # Coordinates and acquisition ranges come from actual SAC headers.
    db=sqlite3.connect(root/'audit/archive_index.sqlite')
    rows=list(db.execute('SELECT station,archive,offset FROM sac WHERE gaussian=3 GROUP BY station ORDER BY station'))
    coords=[];streams={}
    for station,archive,offset in rows:
        if archive not in streams:streams[archive]=(root/'archives'/archive).open('rb')
        f=streams[archive];f.seek(offset);raw=f.read(632)
        order='<' if np.frombuffer(raw,'<i4',40,280)[6] in [6,7] else '>'
        values=np.frombuffer(raw,order+'f4',70)
        lat,lon=map(float,values[[31,32]])
        coords.append(dict(station=station,latitude=lat if -90<=lat<=90 else None,longitude=lon if -180<=lon<=180 else None))
    for f in streams.values():f.close()
    acquisition={name:dict(zip(['n','minimum','maximum'],db.execute(f'SELECT count({column}),min({column}),max({column}) FROM sac WHERE gaussian=3').fetchone())) for name,column in [('gcarc','gcarc'),('magnitude','mag'),('ray_parameter','p')]}
    db.close();dump(packet/'station_metadata.json',dict(coordinates=coords,acquisition=acquisition))
    dump(packet/'collection_verification.json',dict(status='passed',source_version=protocol['version'],
         training_runs=30,source_files_verified=count,test_stations=stations,n_test=n,receipts=receipts,
         private_packet_contains_observations=True,not_a_public_dataset=True,
         sha256={str(p.relative_to(packet)):sha(p) for p in packet.rglob('*') if p.is_file()}))
    with tarfile.open(out/'inputs.tar.gz','w:gz',compresslevel=1) as t:t.add(packet,arcname='inputs')
    print(json.dumps(dict(status='passed',archive=str(out/'inputs.tar.gz'),bytes=(out/'inputs.tar.gz').stat().st_size)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
