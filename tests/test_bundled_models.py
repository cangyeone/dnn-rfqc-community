from pathlib import Path
import numpy as np
import pytest

from rfqc_bench import RFQCPredictor, available_weights, download_model, synthetic_data
import rfqc_bench.zoo as zoo

BUNDLED=[(m['name'],m['seed']) for m in available_weights() if m.get('bundled_path')]


@pytest.mark.parametrize('name,seed',BUNDLED)
def test_real_reference_weights_work_offline(name,seed,monkeypatch,tmp_path):
    def no_network(*args,**kwargs):raise AssertionError('Bundled inference must not download')
    monkeypatch.setattr(zoo,'urlopen',no_network)
    path=download_model(name,seed,cache_dir=tmp_path/'unused')
    assert path.is_relative_to(Path(zoo.__file__).resolve().parent/'pretrained')
    assert not (tmp_path/'unused').exists()
    model=RFQCPredictor.from_pretrained(name,seed=seed)
    result=model.predict(synthetic_data(2))
    assert result.p_good.shape==(2,) and np.isfinite(result.p_good).all()
    assert result.seed==seed and result.method==name


def test_bundled_catalog_has_both_regimes_and_community_urls():
    assert len(BUNDLED)==8
    assert {name for name,seed in BUNDLED}=={'reference_ag1','reference_ag3','reference_ag5','reference_multifilter'}
    assert len(available_weights())==53
    entries=available_weights()
    fresh=[m for m in entries if m['dataset_version']=='dbyp_complete_20261009_v2']
    historical=[m for m in entries if m['dataset_version']=='historical_benchmark_v1']
    assert len(fresh)==30 and len(historical)==23
    assert len({m['name'] for m in fresh})==10
    assert all('/cangyeone/dnn-rfqc-community/releases/download/v0.2.0/' in m['url'] for m in fresh)
    assert all('/cangyeone/dnn-rfqc-community/releases/download/v0.1.3/' in m['url'] for m in historical)
    assert {(m['name'],m['seed']) for m in fresh if m.get('bundled_path')}=={
        (name,seed) for name in ['reference_ag3','reference_multifilter'] for seed in [20260928,20260929,20260930]}


def test_corrupt_bundled_metadata_fails_without_network(monkeypatch):
    entry=next(m for m in available_weights() if m.get('bundled_path'))
    entry['files']['bundle.json']='0'*64
    monkeypatch.setattr(zoo,'available_weights',lambda:[entry])
    with pytest.raises(ValueError,match='Bundled model missing or corrupt'):
        download_model(entry['name'],entry['seed'])
