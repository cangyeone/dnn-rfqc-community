"""Three sequential timing rounds for the ten current DB/YP v2 bundles.

Fresh process per model/round; no training. Supply the released community
benchmarking.py next to this script so its source checksum is recorded.
"""
import argparse, hashlib, importlib.util, json, os, subprocess, sys
from pathlib import Path

METHODS=['li2021_cnn','gan2021_cnn','gong_cnn_bilstm','gan_cnn','deeprfqc','hegaz_capsule','chen2026_image','reference_ag3','reference_multifilter','xiong2025_fcm']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main(a):
    sys.path.insert(0,str(a.root/'runtime'))
    from rfqc_bench import RFData, RFQCPredictor
    code=Path(__file__).with_name('benchmarking.py')
    spec=importlib.util.spec_from_file_location('rfqc_bench.benchmarking',code);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if a.method:
        import torch
        torch.set_num_interop_threads(1)
        folder=a.root/'runs'/f'{a.method}_seed20260928'
        p=RFQCPredictor.from_directory(folder,'cpu' if a.method=='xiong2025_fcm' else 'cuda:0')
        data=RFData.load(a.root/'cache/supervised/test')
        report=module.benchmark_inference(p,data,sample_size=512,complete_views=True,batch_size=32,warmup=10,latency_calls=100,throughput_calls=32,threads=4,round_index=a.round)
        report.update(bundle_sha256=sha(folder/'bundle.json'),protocol_sha256=sha(a.root/'protocol.json'),timing_code_sha256=sha(code),runner_sha256=sha(Path(__file__)))
        a.output.mkdir(parents=True,exist_ok=True);(a.output/f'{a.method}_round{a.round}.json').write_text(json.dumps(report,indent=2)+'\n')
        print(a.method,a.round,report.get('batch_records_per_second',report.get('pool_records_per_second')),flush=True);return
    a.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(methods=METHODS,rounds=3,seed=20260928,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],classification_source=str(a.root),weights_fixed=True,sample_size=512,FCM_pool=27823,threads=4,timing_code_sha256=sha(code),runner_sha256=sha(Path(__file__)),scope='Resident arrays to prediction API, single GPU sequential, FCM full pool CPU')
    config=a.output/'protocol.json'
    if config.exists():assert json.loads(config.read_text())==protocol
    else:config.write_text(json.dumps(protocol,indent=2)+'\n')
    hashes={};manifest=a.output/'completed_rounds.json'
    if manifest.exists():hashes=json.loads(manifest.read_text())
    for repeat in range(3):
        for method in sorted(METHODS,key=lambda m:hashlib.sha256(f'{repeat}/{m}'.encode()).hexdigest()):
            name=f'{method}_round{repeat}.json';target=a.output/name
            if target.exists():assert hashes.get(name)==sha(target);continue
            subprocess.run([sys.executable,str(Path(__file__)), '--root',str(a.root),'--output',str(a.output),'--method',method,'--round',str(repeat)],check=True)
            hashes[name]=sha(target);manifest.write_text(json.dumps(hashes,indent=2)+'\n')
    (a.output/'verification.json').write_text(json.dumps(dict(status='passed',rounds=len(hashes),sha256=hashes),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--method',choices=METHODS);p.add_argument('--round',type=int,default=0);main(p.parse_args())
