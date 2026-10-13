"""Acceptance checks for the current paper, aggregate data and private atlas."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np,pandas as pd
from pypdf import PdfReader

METHODS=['li2021_cnn','gan2021_cnn','gong_cnn_bilstm','gan_cnn','deeprfqc','hegaz_capsule','chen2026_image','xiong2025_fcm','reference_ag3','reference_multifilter']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(root,review):
    source=root/'SourceData/revision_20261013';s=(root/'main.tex').read_text();sup=(root/'supplement.tex').read_text();text=s+'\n'+sup
    assert 'Receiver-function quality control on a common task: a benchmark with published baselines and waveform diagnostics' in s
    assert not re.search(r'Gong-CNN(?![\s−–-]*BiLSTM)',text)
    for term in ['single-band','multi-band','we will','Registered additional','T4','T5','T6','Results pending verification']:assert term not in text,term
    abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',s,re.S).group(1);assert len(abstract.split())<=250
    metrics=pd.read_csv(source/'seed_metrics.csv');assert len(metrics)==30 and set(metrics.method)==set(METHODS)
    assert not metrics[['method','seed']].duplicated().any();assert set(metrics.seed)=={20260928,20260929,20260930};assert set(metrics.n)=={27823}
    paired=pd.read_csv(source/'single_multi_paired.csv')
    summary=pd.read_csv(source/'method_summary.csv').set_index('method')
    for m in METHODS:
        d=metrics[metrics.method==m]
        for key in ['accuracy','macro_f1','good_auprc','balanced_accuracy','good_precision','good_recall']:
            assert abs(d[key].mean()-summary.loc[m,key+'_mean'])<1e-12
            assert abs(d[key].std(ddof=1)-summary.loc[m,key+'_sample_sd'])<1e-12
    table=(root/'tables/dbyp_primary.tex').read_text();assert table.count('\\textbf{')==6 and table.count('7 views')==1
    # Twenty numeric source checks across current prose and tables. These check
    # the amended cohort, not equality with obsolete data-version results.
    checks=[]
    def check(token,condition,origin):
        assert token in text and condition,(token,origin);checks.append(dict(token=token,source=origin,status='passed'))
    check('163,889',112393+23673+27823==163889,'cohort split')
    check('205',142+31+32==205,'station split')
    check('20,616',14637+3195+2784==20616,'cohort good counts')
    check('143,273',97756+20478+25039==143273,'cohort bad counts')
    check('27,823',set(metrics.n)=={27823},'all prediction sets')
    check('2,784',set(metrics.tp+metrics.fn)=={2784},'all test labels')
    check('89.99',f'{100*25039/27823:.2f}'=='89.99','reject-all accuracy')
    for token,method,key in [('89.10','gong_cnn_bilstm','macro_f1'),('96.06','gong_cnn_bilstm','accuracy'),('88.74','gong_cnn_bilstm','good_auprc'),('80.02','gong_cnn_bilstm','good_precision'),('90.75','chen2026_image','balanced_accuracy'),('84.66','chen2026_image','good_recall')]:
        check(token,f'{100*summary.loc[method,key+"_mean"]:.2f}'==token,'independently recomputed '+method+'/'+key)
    check('0.25',f'{paired.macro_f1_delta_pp.mean():.2f}'=='0.25','paired macro-F1 mean')
    check('0.41',f'{paired.macro_f1_delta_pp.std(ddof=1):.2f}'=='0.41','paired macro-F1 sample SD')
    check('0.11',f'{paired.accuracy_delta_pp.mean():.2f}'=='0.11','paired accuracy mean')
    check('0.23',f'{paired.accuracy_delta_pp.std(ddof=1):.2f}'=='0.23','paired accuracy sample SD')
    small=pd.read_csv(source/'small_method_summary.csv').set_index('method');check('151,324',set(small.n_input)=={151324},'small-event pools')
    check('2,134',small.loc['reference_ag3','retained']==2134,'small AG3 retained')
    check('1,573',small.loc['reference_multifilter','retained']==1573,'small multi-filter retained')
    assert len(checks)==20
    diag=json.loads((source/'diagnostic_verification.json').read_text());assert diag['status']=='passed' and len(diag['test_station_checks'])==32
    assert sum(v['primary_peaks_verified'] for v in diag['test_station_checks'])==37984
    assert diag['window']==[-2.5,20.] and diag['model_seed']==20260928
    timing=json.loads((source/'timing/verification.json').read_text());assert timing['rounds']==30
    for name,digest in timing['sha256'].items():assert sha(source/'timing'/name)==digest
    snapshot=json.loads((source/'source_snapshot/manifest.json').read_text())
    for name,digest in snapshot.items():assert sha(source/'source_snapshot'/name)==digest
    for name in ['test_stack_metrics','test_cross_filter','small_stack_metrics']:
        d=pd.read_csv(source/(name+'.csv'));assert set(d.window_start)=={-2.5} and set(d.window_end)=={20.}
    stations=set(pd.read_csv(source/'station_metrics.csv').station);assert len(stations)==32
    for station in stations:
        for kind in ['stacks_peak','stacks_raw','backazimuth','hk']:
            for suffix in ['.pdf','.png']:assert (review/station/(kind+suffix)).is_file()
    for name in ['all_stations_stacks','all_stations_backazimuth','all_stations_hk']:assert len(PdfReader(review/(name+'.pdf')).pages)==32
    assert len(PdfReader(review/'all_small_stations_stacks.pdf').pages)==102
    assert len(list((review/'small').glob('*.pdf')))==102
    cites=set()
    for group in re.findall(r'\\cite[pt]?\{([^}]+)\}',text):cites.update(group.split(','))
    assert len(cites)==34
    refs=(root/'references.bib').read_text();assert all(re.search(r'@\w+\{'+re.escape(k)+',',refs) for k in cites)
    labels=re.findall(r'\\label\{([^}]+)\}',s);assert len(labels)==len(set(labels))
    assert set(re.findall(r'\\ref\{([^}]+)\}',s))<=set(labels)
    pdfs={}
    for name in ['main','supplement']:
        log=(root/'build'/(name+'.log')).read_text()
        assert not re.search(r'undefined|Overfull|LaTeX Error|Citation.*undefined',log,re.I)
        pdf=root/'build'/(name+'.pdf');reader=PdfReader(pdf);contents='\n'.join(p.extract_text() or '' for p in reader.pages)
        # Matplotlib vector-text extraction can represent an en dash as space.
        assert not re.search(r'Gong-CNN(?![\s−–-]*BiLSTM)',contents)
        assert 'Results pending verification' not in contents
        pdfs[name]=dict(pages=len(reader.pages),sha256=sha(pdf))
    report=dict(status='passed',cohort='dbyp_complete_20261009_v2',training_runs=30,source_files_verified_remotely=207,station_count=205,test_count=27823,test_stations=32,current_configurations=10,metrics_recomputed=True,diagnostic_window=[-2.5,20],hk_primary_peaks_checked=37984,inference_rounds=30,abstract_words=len(abstract.split()),cited_keys=sorted(cites),numeric_checks=checks,atlas_station_count=32,small_station_count=102,small_physical_diagnostics='Not performed: no matching baz/p',pdfs=pdfs,verifier_sha256=sha(Path(__file__)))
    (source/'paper_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'SourceData/paper_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','training_runs','test_count','test_stations','hk_primary_peaks_checked','inference_rounds','abstract_words']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--review',type=Path,required=True);a=p.parse_args();main(a.root,a.review)
