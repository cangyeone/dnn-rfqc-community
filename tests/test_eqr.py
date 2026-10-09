import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from fastapi.testclient import TestClient
import numpy as np
import pytest

from rfqc_bench import RFQCPredictor, create_model, screen_eqr, synthetic_data
from rfqc_bench.api import create_app
from rfqc_bench.cli import main
from rfqc_bench.eqr import read_eqr
from rfqc_bench.predictor import Prediction
from rfqc_bench.preprocessing import Preprocessor
from rfqc_bench.registry import get_spec
from rfqc_bench.training import save_bundle


def sac(path, *, value=1., endian='<', version=6, dt=.1, begin=-15., n=601,
        station='EW27', baz=40., nan=False, uneven=False, signal=None):
    """Synthetic binary SAC fixture; no observational data in the test suite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = np.full(70, -12345., dtype=endian+'f4')
    f[[0,5,6,44,52,53]] = [dt,begin,begin+(n-1)*dt,.065,baz,70.]
    i = np.full(40, -12345, dtype=endian+'i4')
    i[[6,9,15,35]] = [version,n,1,0 if uneven else 1]
    chars = bytearray(b' '*192)
    chars[:8] = station.encode().ljust(8)[:8]
    chars[160:168] = b'EQR     '
    chars[168:176] = b'DB      '
    wave = np.full(n,value,dtype=endian+'f4')
    if signal is not None:wave=np.asarray(signal(begin+np.arange(n)*dt),dtype=endian+'f4')
    if nan:wave[100] = np.nan
    footer = np.full(22, -12345., dtype=endian+'f8')
    footer[:3] = [dt,begin,begin+(n-1)*dt]
    path.write_bytes(f.tobytes()+i.tobytes()+chars+wave.tobytes()+(footer.tobytes() if version==7 else b''))
    return path


class Dummy:
    def __init__(self, model='reference_multifilter'):
        self.spec=get_spec(model);self.seed=12;self.threshold=.6;self.device='cpu'
        self.bundle={'input_grid':{},'method':model}
        self.calls=[]

    def predict(self, data, batch_size=32):
        self.calls.append(data)
        p=np.where(data.waveforms[:,0,100]>0,.9,.1)
        return Prediction(p,(p>=self.threshold).astype(int),self.threshold,self.spec.name,self.seed)

    def screen_eqr(self,*args,**kwargs):
        return screen_eqr(*args,predictor=self,**kwargs)


@pytest.mark.parametrize('endian',['<','>'])
@pytest.mark.parametrize('version',[6,7])
def test_sac_matches_obspy_grid(tmp_path,endian,version):
    path=sac(tmp_path/'test.eqr',endian=endian,version=version)
    wave,metadata=read_eqr(path)
    assert wave.shape==(501,) and wave.dtype==np.float32
    np.testing.assert_array_equal(wave,np.ones(501))
    assert metadata['station']=='EW27' and metadata['baz']==40.
    assert len(metadata['sha256'])==64
    # Independent optional oracle for the SAC header and exact waveform slice.
    obspy=pytest.importorskip('obspy')
    # ObsPy's size check predates the v7 double footer; use it as a waveform
    # oracle only for v7. The reader's own exact file-size checks remain on.
    trace=obspy.read(str(path),format='SAC',fsize=(version==6))[0]
    np.testing.assert_array_equal(wave,trace.data[50:551])


def test_v7_uses_double_footer_time_grid(tmp_path):
    path=sac(tmp_path/'v7.eqr',version=7)
    raw=bytearray(path.read_bytes())
    raw[:4]=np.array([.2],dtype='<f4').tobytes()
    path.write_bytes(raw)
    assert read_eqr(path)[0].shape==(501,)


@pytest.mark.parametrize('options,match',[
    ({'dt':.2},'delta'),({'begin':0},'cover'),({'begin':-15.01},'cover'),
    ({'n':500},'cover'),({'nan':True},'Nonfinite'),({'value':0},'All-zero'),
    ({'uneven':True},'evenly sampled')])
def test_invalid_sac(tmp_path,options,match):
    with pytest.raises(ValueError,match=match):read_eqr(sac(tmp_path/'x.eqr',**options))


def test_truncated_and_extra_bytes(tmp_path):
    path=sac(tmp_path/'x.eqr')
    raw=path.read_bytes()
    for bad in [raw[:200],raw[:-4],raw+b'junk']:
        path.write_bytes(bad)
        with pytest.raises(ValueError):read_eqr(path)


def test_grouping_missing_view_and_no_folder_label_leakage(tmp_path):
    root=tmp_path/'input'
    # A good prediction must still be possible under a folder named bad.
    sac(root/'A/AG1/bad/same.eqr')
    sac(root/'A/AG3/same.eqr')
    sac(root/'A/AG3/other.eqr',value=-1)
    sac(root/'B/AG3/same.eqr',value=-1,station='B')
    before={p:p.read_bytes() for p in root.rglob('*.eqr')}
    predictor=Dummy()
    report=screen_eqr(root,predictor=predictor)
    assert (root/'record').read_text().splitlines()==['same.eqr']
    assert report['record_entries']==1
    assert report['events_screened']==3 and report['good_events']==1 and report['retained_files']==2
    assert [len(d) for d in predictor.calls]==[2,1]
    assert sorted(predictor.calls[0].lengths)==[1,2]
    assert all(p.read_bytes()==b for p,b in before.items())
    rows=list(csv.DictReader((root/'record.predictions.csv').open()))
    assert len({r['sample_id'] for r in rows})==3
    assert report==json.loads((root/'record.json').read_text())
    with pytest.raises(FileExistsError):screen_eqr(root,predictor=predictor)
    screen_eqr(root,predictor=predictor,overwrite=True)


def test_record_deduplicates_basenames_globally_and_preserves_source_paths(tmp_path):
    root=tmp_path/'input'
    name='XZ_NAQ_2022219_214001.eqr'
    retained=[f'A/AG1/bad/{name}',f'A/AG3/{name}',f'A/AG5/{name}',f'B/AG3/{name}']
    for path in retained:
        sac(root/path,station=path[0])
    sac(root/'A/AG3/rejected.eqr',value=-1,station='A')
    report=screen_eqr(root,predictor=Dummy())
    assert (root/'record').read_text()==name+'\n'
    assert report['record_entries']==1 and report['retained_files']==4
    assert report['good_events']==2 and report['bad_events']==1
    rows=list(csv.DictReader((root/'record.predictions.csv').open()))
    paths=[path for row in rows if row['prediction']=='1' for path in json.loads(row['files'])]
    assert sorted(paths)==retained
    assert all(set(json.loads(row['files']))==set(json.loads(row['source_sha256'])) for row in rows)
    screen_eqr(root,predictor=Dummy(),overwrite=True)
    assert (root/'record').read_text()==name+'\n'


def test_single_filter_and_root_itself_ag(tmp_path):
    root=tmp_path/'STA'
    sac(root/'AG1/event.eqr');sac(root/'AG3/event.eqr')
    report=screen_eqr(root,predictor=Dummy('reference_ag3'))
    assert (root/'record').read_text()=='event.eqr\n'
    assert report['rejected_or_unused_entries']==1
    report=screen_eqr(root/'AG3',predictor=Dummy())
    assert (root/'AG3/record').read_text()=='event.eqr\n'
    assert report['good_events']==1


def test_duplicates_and_header_conflicts_are_not_chosen(tmp_path):
    root=tmp_path/'input'
    sac(root/'STA/AG3/duplicate.eqr')
    sac(root/'STA/AG3/bad/duplicate.eqr')
    sac(root/'STA/AG1/conflict.eqr')
    sac(root/'STA/AG3/conflict.eqr',baz=70)
    report=screen_eqr(root,predictor=Dummy())
    assert report['events_screened']==0 and report['status']=='no_valid_events'
    assert (root/'record').read_bytes()==b''
    reasons=(root/'record.rejected.csv').read_text()
    assert 'Ambiguous duplicate' in reasons and 'Conflicting' in reasons


def test_flat_gaussian_required_and_fcm_station_pool(tmp_path):
    root=tmp_path/'flat'
    for i in range(5):sac(root/f'{i}.eqr',station='A' if i<3 else 'B')
    report=screen_eqr(root,tmp_path/'missing',predictor=Dummy())
    assert report['status']=='no_valid_events'
    predictor=Dummy('xiong2025_fcm')
    report=screen_eqr(root,predictor=predictor,gaussian=3.,batch_size=1)
    assert sorted(len(d) for d in predictor.calls)==[2,3]
    assert report['retained_files']==5


def test_rejections_limits_and_failed_inference_preserve_outputs(tmp_path):
    root=tmp_path/'input'
    sac(root/'AG3/valid.eqr');sac(root/'AG3/invalid.eqr',nan=True)
    predictor=Dummy()
    report=screen_eqr(root,predictor=predictor)
    assert report['retained_files']==1 and report['rejected_or_unused_entries']==1
    before={p:p.read_bytes() for p in root.glob('record*')}
    def fail(*args,**kwargs):raise RuntimeError('inference failed')
    predictor.predict=fail
    with pytest.raises(RuntimeError):screen_eqr(root,predictor=predictor,overwrite=True)
    assert all(p.read_bytes()==data for p,data in before.items())
    with pytest.raises(ValueError,match='limit'):
        screen_eqr(root,tmp_path/'limited',predictor=Dummy(),max_files=1)
    assert not (tmp_path/'limited').exists()
    with pytest.raises(ValueError,match='waveform models'):
        screen_eqr(root,tmp_path/'feature',predictor=Dummy('combined_ag3'))
    with pytest.raises(ValueError,match='EQR'):
        screen_eqr(root,root/'AG3/valid.eqr',predictor=Dummy())
    (root/'.busy.rfqc-lock').mkdir()
    with pytest.raises(FileExistsError,match='Publication lock'):
        screen_eqr(root,root/'busy',predictor=Dummy())
    assert not (root/'busy').exists()


def test_http_and_symlinks(tmp_path):
    root=tmp_path/'root';sac(root/'input/AG3/ok.eqr')
    outside=tmp_path/'outside';sac(outside/'secret.eqr')
    (root/'outside').symlink_to(outside,target_is_directory=True)
    (root/'input/AG3/link.eqr').symlink_to(outside/'secret.eqr')
    disabled=TestClient(create_app(Dummy()))
    assert disabled.post('/screen-eqr',json={'directory':str(root)}).status_code==403
    client=TestClient(create_app(Dummy(),eqr_root=root))
    for directory in ['../outside',str(outside),'outside']:
        assert client.post('/screen-eqr',json={'directory':directory}).status_code==422
    assert client.post('/screen-eqr',json={'directory':'input','output':'../oops'}).status_code==422
    response=client.post('/screen-eqr',json={'directory':'input'})
    assert response.status_code==200 and response.json()['retained_files']==1
    assert response.json()['rejected_or_unused_entries']==1
    assert client.post('/screen-eqr',json={'directory':'input'}).status_code==409
    assert not (outside/'record').exists()
    assert '/screen-eqr' in client.get('/openapi.json').json()['paths']


def test_actual_model_python_and_cli(tmp_path,capsys):
    training=synthetic_data(4)
    bundle=tmp_path/'model'
    save_bundle(bundle,'reference_multifilter',123,.5,Preprocessor.fit(training).state,
                create_model('reference_multifilter').eval())
    root=tmp_path/'input'
    for gaussian in [1,3,5]:sac(root/f'STA/AG{gaussian}/test.eqr')
    predictor=RFQCPredictor.from_directory(bundle)
    expected=predictor.predict(waveforms=np.ones((1,3,501)),gaussians=[1,3,5])
    result=predictor.screen_eqr(root)
    rows=list(csv.DictReader((root/'record.predictions.csv').open()))
    assert float(rows[0]['p_good'])==expected.p_good[0]
    main(['screen-eqr',str(root),'--model-dir',str(bundle),'--output',str(tmp_path/'cli-record')])
    assert json.loads(capsys.readouterr().out)['events_screened']==1
    assert (tmp_path/'cli-record').read_bytes()==(root/'record').read_bytes()
    assert result['model_weights_sha256']==predictor.bundle['weights_sha256']


def test_call_threshold_does_not_mutate_model_or_scores(tmp_path):
    root=tmp_path/'input';sac(root/'AG3/a.eqr');sac(root/'AG3/b.eqr',value=-1)
    model=Dummy()
    for threshold,expected in [(0,2),(.1,2),(.9,1),(.95,0),(1,0),(None,1)]:
        result=screen_eqr(root,tmp_path/f'record_{threshold}',predictor=model,threshold=threshold)
        assert result['good_events']==expected and model.threshold==.6
        assert result['threshold_source']==('model' if threshold is None else 'user')
        rows=list(csv.DictReader(open(result['outputs']['predictions'])))
        assert [float(r['p_good']) for r in rows]==[.9,.1]
        assert all(float(r['threshold'])==result['threshold'] for r in rows)
    for bad in [-.01,1.01,float('nan'),float('inf'),True]:
        with pytest.raises(ValueError,match='threshold'):
            screen_eqr(root,tmp_path/'bad',predictor=model,threshold=bad)


@pytest.mark.parametrize('dt,begin,n',[(.02,-15,3001),(.05,-15,1201),(.2,-15,301),(.025,-15.025,2403),(.1,-15.03,602),(.03,-10.01,1668)])
@pytest.mark.parametrize('version,endian',[(6,'<'),(7,'>')])
def test_resampling_preserves_low_frequency_time_and_amplitude(tmp_path,dt,begin,n,version,endian):
    signal=lambda t: np.exp(-((t-4.)/.7)**2)+.2*np.sin(2*np.pi*.2*t)
    path=sac(tmp_path/'signal.eqr',dt=dt,begin=begin,n=n,signal=signal,version=version,endian=endian)
    before=path.read_bytes();wave,metadata=read_eqr(path,resample=True)
    truth=signal(np.arange(501)*.1-10.)
    assert wave.shape==(501,) and metadata['resampled']
    # Include P/Ps timing, not merely a copy of the implementation's algorithm.
    assert np.max(np.abs(wave[20:-20]-truth[20:-20]))<.02
    assert abs(np.argmax(wave)-np.argmax(truth))<=1
    assert path.read_bytes()==before


def test_downsampling_rejects_alias_energy_and_preserves_exact_grid(tmp_path):
    path=sac(tmp_path/'alias.eqr',dt=.02,n=3001,
             signal=lambda t: np.sin(2*np.pi*t)+np.sin(2*np.pi*8*t))
    wave,metadata=read_eqr(path,resample=True)
    truth=np.sin(2*np.pi*(np.arange(501)*.1-10.))
    assert np.sqrt(np.mean((wave[20:-20]-truth[20:-20])**2))<.01
    path=sac(tmp_path/'aligned.eqr',signal=lambda t:np.cos(t))
    np.testing.assert_array_equal(read_eqr(path)[0],read_eqr(path,resample=True)[0])
    assert not read_eqr(path,resample=True)[1]['resampled']


@pytest.mark.parametrize('options',[{'dt':0},{'dt':-.1},{'dt':.05,'n':501},{'dt':.2,'begin':-9}])
def test_resampling_does_not_fabricate_time_coverage(tmp_path,options):
    with pytest.raises(ValueError):read_eqr(sac(tmp_path/'invalid.eqr',**options),resample=True)


def test_select_views_and_audit_mixed_sampling_rates(tmp_path):
    root=tmp_path/'mixed'
    sac(root/'STA/AG1/event.eqr',dt=.05,n=1201)
    sac(root/'STA/AG3/event.eqr')
    sac(root/'STA/AG5/event.eqr',dt=.2,n=301)
    predictor=Dummy()
    result=screen_eqr(root,predictor=predictor,filters=[1,5],resample=True)
    assert result['files_resampled']==2 and result['retained_files']==2
    assert predictor.calls[0].gaussians.tolist()==[[1,5]]
    assert (root/'record').read_text().splitlines()==['event.eqr']
    assert result['record_entries']==1
    inputs=list(csv.DictReader((root/'record.inputs.csv').open()))
    assert len(inputs)==2 and all(r['resampled']=='True' for r in inputs)
    assert all(r['source_delta_s'] for r in inputs)
    with pytest.raises(ValueError,match='requires AG3'):
        screen_eqr(root,tmp_path/'wrong',predictor=Dummy('reference_ag3'),filters=[1,5])
    for filters in [[],[8],[float('nan')]]:
        with pytest.raises(ValueError,match='filters'):
            screen_eqr(root,tmp_path/'bad_filter',predictor=predictor,filters=filters)


def test_http_threshold_isolation_resample_and_filter_selection(tmp_path):
    root=tmp_path/'root';sac(root/'input/AG1/a.eqr',dt=.05,n=1201);sac(root/'input/AG3/a.eqr')
    predictor=Dummy();client=TestClient(create_app(predictor,eqr_root=root))
    request=dict(directory='input',output='strict',threshold=.95,resample=True,filters=[1])
    r=client.post('/screen-eqr',json=request)
    assert r.status_code==200 and r.json()['good_events']==0 and r.json()['files_resampled']==1
    r=client.post('/screen-eqr',json={'directory':'input','output':'default'})
    assert r.status_code==200 and r.json()['good_events']==1 and r.json()['threshold']==.6
    assert predictor.threshold==.6
    assert client.post('/screen-eqr',json={**request,'threshold':1.1}).status_code==422


def test_cli_passes_threshold_resample_filters(tmp_path,capsys,monkeypatch):
    root=tmp_path/'input';sac(root/'AG1/event.eqr',dt=.05,n=1201)
    monkeypatch.setattr('rfqc_bench.cli._predictor',lambda a:Dummy())
    main(['screen-eqr',str(root),'--threshold','.95','--resample','--filters','1'])
    report=json.loads(capsys.readouterr().out)
    assert report['good_events']==0 and report['files_resampled']==1
    assert report['threshold']==.95 and report['requested_filters']==[1]


@pytest.mark.skipif(shutil.which('bash') is None,reason='Bash launcher is for Linux/macOS/WSL; Windows uses the CLI')
def test_portable_launcher_from_another_users_directory(tmp_path):
    project=Path(__file__).resolve().parents[1]
    home=tmp_path/'other user 中文';home.mkdir()
    launcher=home/'screen_eqr.sh';shutil.copy2(project/'screen_eqr.sh',launcher)
    source=home/'input with spaces'
    sac(source/'ST/AG1/event.eqr',dt=.05,n=1201)
    sac(source/'ST/AG5/event.eqr',dt=.2,n=301)
    output=home/'my results'/'record'
    env=dict(os.environ,RFQC_PYTHON=sys.executable,PYTHONPATH=str(project/'src'),HOME=str(home),
             RFQC_CACHE=str(home/'model_cache'))
    result=subprocess.run(['bash',str(launcher),str(source),'--threshold','0','--resample',
                           '--filters','1','5','--output',str(output)],env=env,cwd=tmp_path,
                           text=True,capture_output=True,check=True,timeout=60)
    report=json.loads(result.stdout)
    assert report['retained_files']==2 and report['files_resampled']==2
    assert output.read_text().splitlines()==['event.eqr']
    assert report['record_entries']==1
    assert not (home/'model_cache').exists(), 'Bundled model should require no shared writable cache'
