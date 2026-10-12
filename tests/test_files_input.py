import csv
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from rfqc_bench import RFQCPredictor, screen_eqr
from rfqc_bench.api import create_app
from rfqc_bench.cli import main
from rfqc_bench.files_input import read_files_row
from test_eqr import Dummy, sac


def row(name, value=1., dt=.1, start=-100, end=401, waveform=None):
    values=np.full(end-start,value) if waveform is None else waveform
    return f'{name} {dt} {start} {end} '+' '.join(f'{x:.17g}' for x in values)+'\n'


def export(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(rows),encoding='utf-8')
    return path


def test_text_indices_and_resampling_match_sac(tmp_path):
    signal=lambda t:np.sin(2*np.pi*.4*t)
    for dt,start,end in [(.1,-100,401),(.05,-200,801),(.2,-50,201)]:
        raw=signal(np.arange(start,end)*dt).astype(np.float32)
        name,wave,meta=read_files_row(row('x.eqr',dt=dt,start=start,end=end,waveform=raw),resample=True)
        # The v7 double footer expresses the same grid as the text header;
        # v6 stores delta in float32 and can introduce a small time-grid drift.
        path=sac(tmp_path/'x.eqr',dt=dt,begin=start*dt,n=end-start,signal=signal,version=7)
        from rfqc_bench.eqr import read_eqr
        np.testing.assert_allclose(wave,read_eqr(path,resample=True)[0],atol=2e-7,rtol=0)
        assert name=='x.eqr' and wave.shape==(501,) and meta['source_end_index_exclusive']==end
    assert not read_files_row(row('x.eqr'))[2]['resampled']
    with pytest.raises(ValueError,match='delta'):
        read_files_row(row('x.eqr',dt=.05,start=-200,end=801))


@pytest.mark.parametrize('text,match',[
    ('x.eqr .1 -100 401 1','count'),
    ('x.eqr 0 -100 401 '+'1 '*501,'positive'),
    ('x.eqr .1 -100.0 401 '+'1 '*501,'integer'),
    ('x.eqr .1 0 501 '+'1 '*501,'cover'),
    ('x.eqr .1 -100 401 '+'nan '*501,'Nonfinite'),
    ('x.eqr .1 -100 401 '+'0 '*501,'All-zero'),
    ('x.sac .1 -100 401 '+'1 '*501,'.eqr'),
])
def test_invalid_rows_are_not_fabricated(text,match):
    with pytest.raises(ValueError,match=match):read_files_row(text,resample=True)


def test_align_by_filename_independent_of_order_and_preserve_lines(tmp_path):
    root=tmp_path/'YP_NE3A'
    paths=[export(root/'AG1/good_2',[row('same.eqr'),row('bad.eqr',value=-1)]),
           export(root/'AG3/bad_2',[row('bad.eqr',value=-1),row('same.eqr')])]
    before={p:p.read_bytes() for p in paths}
    model=Dummy();result=screen_eqr(root,predictor=model,input_type='files')
    assert result['input_type']=='files' and result['events_screened']==2 and result['good_events']==1
    assert result['record_entries']==1 and result['retained_files']==2
    assert (root/'record').read_text()=='same.eqr\n'
    assert model.calls[0].lengths.tolist()==[2,2]
    rows=list(csv.DictReader((root/'record.predictions.csv').open()))
    good=next(r for r in rows if r['event']=='same.eqr')
    assert json.loads(good['source_lines'])=={'AG1/good_2':1,'AG3/bad_2':2}
    assert all(p.read_bytes()==b for p,b in before.items())
    with pytest.raises(FileExistsError):screen_eqr(root,predictor=model,input_type='files')
    screen_eqr(root,predictor=model,input_type='files',overwrite=True)
    def fail(*args,**kwargs):raise RuntimeError('inference failed')
    old={p:p.read_bytes() for p in root.glob('record*')};model.predict=fail
    with pytest.raises(RuntimeError):screen_eqr(root,predictor=model,input_type='files',overwrite=True)
    assert all(p.read_bytes()==b for p,b in old.items())


def test_duplicates_skip_entire_event_and_missing_views_remain(tmp_path):
    root=tmp_path/'input'
    export(root/'A/AG1/good_2',[row('duplicate.eqr'),row('keep.eqr')])
    export(root/'A/AG3/bad_2',[row('duplicate.eqr'),row('duplicate.eqr')])
    export(root/'B/AG3/good_1',[row('keep.eqr')])
    result=screen_eqr(root,predictor=Dummy(),input_type='files')
    assert result['events_screened']==2 and result['good_events']==2 and result['retained_files']==2
    assert (root/'record').read_text()=='keep.eqr\n'
    assert 'Ambiguous duplicate' in (root/'record.rejected.csv').read_text()


def test_explicit_export_single_filter_limits_and_source_protection(tmp_path):
    root=tmp_path/'STA'
    path=export(root/'AG3/good_1',[row('event.eqr')])
    result=screen_eqr(path,predictor=Dummy('reference_ag3'),input_type='files')
    assert result['record_entries']==1 and (path.parent/'record').read_text()=='event.eqr\n'
    with pytest.raises(ValueError,match='source'):
        screen_eqr(path,path,predictor=Dummy(),input_type='files',overwrite=True)
    export(root/'AG1/good_1',[row('event.eqr')])
    with pytest.raises(ValueError,match='limit'):
        screen_eqr(root,tmp_path/'limited',predictor=Dummy(),input_type='files',max_files=1)
    assert not (tmp_path/'limited').exists()
    with pytest.raises(ValueError,match='input_type'):
        screen_eqr(root,predictor=Dummy(),input_type='unknown')


def test_cli_and_http_use_selected_input_type(tmp_path,monkeypatch,capsys):
    root=tmp_path/'server';export(root/'input/AG3/good_1',[row('event.eqr')])
    predictor=Dummy()
    monkeypatch.setattr('rfqc_bench.cli._predictor',lambda a:predictor)
    main(['screen-eqr',str(root/'input'),'--input-type','files','--threshold','.95'])
    assert json.loads(capsys.readouterr().out)['good_events']==0
    client=TestClient(create_app(predictor,eqr_root=root))
    response=client.post('/screen-eqr',json={'directory':'input','output':'http_record','input_type':'files'})
    assert response.status_code==200 and response.json()['record_entries']==1
    assert (root/'http_record').read_text()=='event.eqr\n'
    assert client.post('/screen-eqr',json={'directory':'input','input_type':'zip'}).status_code==422
    assert client.post('/screen-eqr',json={'directory':'../outside','input_type':'files'}).status_code==422
    export(root/'another/AG3/good_1',[row('single.eqr')])
    response=client.post('/screen-eqr',json={'directory':'another/AG3/good_1','input_type':'files'})
    assert response.status_code==200 and (root/'another/AG3/record').read_text()=='single.eqr\n'


def test_real_bundle_text_and_sac_predictions_agree(tmp_path):
    root=tmp_path/'files';eqr=tmp_path/'eqr'
    signal=lambda t:np.exp(-(t/.4)**2)+.2*np.exp(-((t-4)/.7)**2)
    for g in [1,3,5]:
        wave=signal(np.arange(501)*.1-10).astype(np.float32)
        export(root/f'STA/AG{g}/good_1',[row('event.eqr',waveform=wave)])
        sac(eqr/f'STA/AG{g}/event.eqr',n=501,begin=-10.,signal=signal)
    model=RFQCPredictor.from_pretrained('reference_multifilter')
    a=model.screen_eqr(root,input_type='files');b=model.screen_eqr(eqr)
    def scores(r):
        rows=list(csv.DictReader(Path(r['outputs']['predictions']).open()))
        return [float(x['p_good']) for x in rows]
    np.testing.assert_array_equal(scores(a),scores(b))
    assert Path(a['outputs']['record']).read_bytes()==Path(b['outputs']['record']).read_bytes()
