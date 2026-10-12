"""Screen filename-bearing waveform exports; no SAC reconstruction or labels."""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from .data import GAUSSIANS, RFData
from .eqr import _location, _prepare_waveform
from .io import digest


def read_files_row(raw, *, resample=False):
    """Parse filename, dt, start_index, exclusive_end_index, then samples.

    The indices are sample offsets relative to P=0, not seconds. For example,
    ``event.eqr 0.1 -100 401`` introduces 501 samples spanning -10..40 s.
    """
    fields = raw.decode('utf-8-sig').split() if isinstance(raw, bytes) else raw.split()
    if len(fields) < 5:
        raise ValueError('Expected filename, delta, start/end sample indices and waveform')
    name = fields[0].replace('\\','/').rsplit('/',1)[-1]
    if not name or Path(name).suffix.lower() != '.eqr' or '\x00' in name:
        raise ValueError('The first field must name an .eqr file')
    try:
        dt = float(fields[1].replace('D','E').replace('d','e'))
        start, end = int(fields[2]), int(fields[3])
    except ValueError:
        raise ValueError('Invalid delta or integer sample indices') from None
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('delta must be finite and positive')
    n = end-start
    if not 1 <= n <= 10_000_000 or len(fields)-4 != n:
        raise ValueError('Waveform count must equal exclusive_end_index - start_index')
    try:
        wave = np.asarray([float(x.replace('D','E').replace('d','e')) for x in fields[4:]], dtype=np.float32)
    except ValueError:
        raise ValueError('Invalid numeric waveform') from None
    if not np.isfinite(wave).all():
        raise ValueError('Nonfinite waveform')
    begin = start*dt
    last = (end-1)*dt
    if not np.isfinite(begin) or not np.isfinite(last):
        raise ValueError('Nonfinite time grid')
    wave, conversion = _prepare_waveform(wave, dt, begin, resample=resample)
    return name, wave, dict(source_samples=n, source_delta_s=dt, source_rate_hz=1./dt,
                            source_begin_s=begin, source_end_s=last,
                            source_start_index=start, source_end_index_exclusive=end, **conversion)


def _candidate(path):
    return (re.fullmatch(r'(?:good|bad)_\d+(?:\.txt)?', path.name, re.IGNORECASE) is not None
            or path.suffix.lower() in {'.txt','.files','.dat'} or path.name.lower() == 'files')


def screen_files(directory, output=None, *, predictor=None, model='reference_multifilter',
                 seed=None, model_dir=None, device='cpu', cache_dir=None, gaussian=None,
                 batch_size=32, overwrite=False, max_files=None, threshold=None,
                 filters=None, resample=False):
    """Read one export file or station/AG directories, then write unique names.

    ``good_*``/``bad_*`` names never provide labels. Events align by basename,
    independently of row order. max_files bounds nonempty waveform rows here.
    Source file hashes and line numbers remain in CSV audit outputs.
    """
    from .predictor import RFQCPredictor
    source = Path(directory).expanduser().absolute()
    if source.is_symlink():
        raise ValueError('Input must not be a symlink')
    source = source.resolve(strict=True)
    if not source.is_dir() and not source.is_file():
        raise ValueError('Input must be a waveform export file or directory')
    root = source if source.is_dir() else source.parent
    if gaussian is not None and gaussian not in GAUSSIANS:
        raise ValueError('Unsupported Gaussian coefficient (not Hz)')
    if threshold is not None:
        if isinstance(threshold,bool) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('threshold must be a finite number in [0, 1]')
        threshold = float(threshold)
    if filters is not None:
        filters = tuple(sorted(set(float(g) for g in filters)))
        if not filters or any(g not in GAUSSIANS for g in filters):
            raise ValueError('filters must select Gaussian coefficients from 1, 1.5, 2, 2.5, 3, 4, 5 (not Hz)')
        if gaussian is not None and gaussian not in filters:
            raise ValueError('--gaussian is not included in --filters')
    if batch_size < 1 or (max_files is not None and max_files < 1):
        raise ValueError('batch_size and max_files must be positive')
    target = Path(output).expanduser().absolute() if output is not None else root/'record'
    if any(p.is_symlink() for p in (target,*target.parents)):
        raise ValueError('Output paths must not contain symlinks')
    if target.suffix.lower() == '.eqr':
        raise ValueError('The record output must not overwrite an EQR file')
    destinations = dict(record=target, predictions=Path(str(target)+'.predictions.csv'),
                        rejected=Path(str(target)+'.rejected.csv'), inputs=Path(str(target)+'.inputs.csv'),
                        summary=Path(str(target)+'.json'))
    for path in destinations.values():
        if path.is_symlink() or (path.exists() and (not overwrite or not path.is_file())):
            raise FileExistsError(f'Output exists or is not a regular file: {path}; choose a new output or use --overwrite')
    if predictor is None:
        predictor = (RFQCPredictor.from_directory(model_dir,device) if model_dir is not None else
                     RFQCPredictor.from_pretrained(model,seed,device,cache_dir))
    elif model_dir is not None:
        raise ValueError('Pass predictor or model_dir, not both')
    if predictor.spec.mode != 'waveform':
        raise ValueError('Files screening supports waveform models only; supplied descriptors are required otherwise')
    if filters is not None and predictor.spec.gaussian is not None and predictor.spec.gaussian not in filters:
        raise ValueError(f'Model requires AG{predictor.spec.gaussian:g}, which is not in --filters; choose a matching model')
    effective_threshold = predictor.threshold if threshold is None else threshold
    rejected, pools = [], defaultdict(list)
    def reject(path, line, reason):
        rejected.append((path.relative_to(root).as_posix(), line, reason))
    if source.is_file():
        candidates = [source]
    else:
        candidates = []
        def walk_error(error): raise error
        for folder, subdirs, names in os.walk(root, followlinks=False, onerror=walk_error):
            subdirs.sort()
            for name in list(subdirs):
                p = Path(folder)/name
                if p.is_symlink():
                    reject(p,'','Skipped symlink directory; contents not scanned')
                    subdirs.remove(name)
            candidates.extend(Path(folder)/name for name in sorted(names) if _candidate(Path(name)))
    # Prevent an export or a source-containing directory from being replaced
    # by any output, even when the caller explicitly asks for overwrite.
    for path in destinations.values():
        if any(path == p or path in p.parents for p in candidates):
            raise ValueError('Output must not overwrite a source waveform export or its directory')
    if not candidates:
        raise ValueError('No waveform export files found; supply good_N/bad_N, *.txt/*.files/*.dat, or an explicit file')
    for path in candidates:
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError('Symlink or non-regular export file is not read')
            anchor, view, _ = _location(path, root, gaussian)
            pools[(str(anchor),anchor.name)].append((view,path))
        except (ValueError,OSError) as exc:
            reject(path,'',str(exc))
    totals = dict(files_found=len(candidates), waveform_rows_found=0, events_screened=0,
                  good_events=0, bad_events=0, retained_files=0, record_entries=0,
                  stations_screened=0, files_resampled=0)
    recorded, source_hashes = set(), {}
    started = datetime.now(timezone.utc).isoformat()
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.rfqc-screen-',dir=target.parent) as temporary:
        staging = {key:Path(temporary)/key for key in destinations}
        with staging['record'].open('w',encoding='utf-8',newline='\n') as records, \
             staging['predictions'].open('w',encoding='utf-8',newline='') as predictions, \
             staging['inputs'].open('w',encoding='utf-8',newline='') as inputs:
            writer = csv.writer(predictions)
            writer.writerow(['sample_id','station','event','gaussians','p_good','prediction','threshold',
                             'method','seed','files','source_lines','source_sha256'])
            fields = ['path','line','event','gaussian','source_samples','source_delta_s','source_rate_hz',
                      'source_begin_s','source_end_s','source_start_index','source_end_index_exclusive',
                      'resampled','resampling_up','resampling_down','resampling_method','sha256']
            input_writer = csv.DictWriter(inputs,fieldnames=fields); input_writer.writeheader()
            for (anchor,station), files in sorted(pools.items()):
                events, seen, ambiguous = defaultdict(dict), set(), set()
                for view,path in sorted(files):
                    relative = path.relative_to(root).as_posix()
                    file_hash = hashlib.sha256()
                    with path.open('rb') as stream:
                        for line, raw in enumerate(stream,1):
                            file_hash.update(raw)
                            if not raw.strip(): continue
                            totals['waveform_rows_found'] += 1
                            if max_files is not None and totals['waveform_rows_found'] > max_files:
                                raise ValueError(f'Input exceeds the configured limit of {max_files} waveform rows')
                            if filters is not None and view not in filters:
                                reject(path,line,'Excluded by requested --filters selection'); continue
                            if predictor.spec.gaussian is not None and view != predictor.spec.gaussian:
                                reject(path,line,f'Unused by single-filter model requiring AG{predictor.spec.gaussian:g}'); continue
                            try:
                                # Count repeated event/view names even if one row is malformed.
                                tokens = raw.decode('utf-8-sig').split()
                                if not tokens:
                                    raise ValueError('Expected filename and waveform fields')
                                event = tokens[0].replace('\\','/').rsplit('/',1)[-1]
                                key = (event,view)
                                if key in seen:
                                    ambiguous.add(event)
                                    raise ValueError('Ambiguous duplicate event/filter view; entire event skipped')
                                seen.add(key)
                                event,wave,metadata = read_files_row(raw,resample=resample)
                                metadata.update(path=relative,line=line,event=event,gaussian=view,
                                                sha256=hashlib.sha256(raw).hexdigest())
                                events[event][view] = (wave,metadata)
                            except (ValueError,UnicodeError) as exc:
                                reject(path,line,str(exc))
                    source_hashes[relative] = file_hash.hexdigest()
                for event in ambiguous:
                    for _,meta in events.pop(event,{}).values():
                        rejected.append((meta['path'],meta['line'],'Ambiguous duplicate event/filter view; entire event skipped'))
                if not events: continue
                names = sorted(events); n=len(names); k=max(map(len,events.values()))
                waves=np.zeros((n,k,501),dtype=np.float32); gaussians=np.zeros((n,k),dtype=np.float32)
                lengths=np.array([len(events[e]) for e in names])
                for i,event in enumerate(names):
                    for j,(view,(wave,_)) in enumerate(sorted(events[event].items())):
                        waves[i,j]=wave; gaussians[i,j]=view
                data=RFData(waves,gaussians,lengths,stations=np.repeat(station,n))
                prediction=predictor.predict(data,batch_size=batch_size)
                totals['stations_screened'] += 1
                for i,event in enumerate(names):
                    entries=[meta for _,(_,meta) in sorted(events[event].items())]
                    paths=[meta['path'] for meta in entries]
                    lines={meta['path']:meta['line'] for meta in entries}
                    hashes={path:source_hashes[path] for path in paths}
                    decision=int(prediction.p_good[i]>=effective_threshold)
                    sample_id=json.dumps([os.path.relpath(anchor,root),station,event],ensure_ascii=False)
                    writer.writerow([sample_id,station,event,';'.join(f'{g:g}' for g in sorted(events[event])),
                                     float(prediction.p_good[i]),decision,effective_threshold,predictor.spec.name,
                                     predictor.seed,json.dumps(paths,ensure_ascii=False),json.dumps(lines,ensure_ascii=False),
                                     json.dumps(hashes,ensure_ascii=False,sort_keys=True)])
                    for meta in entries:
                        input_writer.writerow(meta)
                        totals['files_resampled'] += int(meta['resampled'])
                    totals['events_screened'] += 1
                    totals['good_events' if decision else 'bad_events'] += 1
                    if decision:
                        totals['retained_files'] += len(entries)
                        if event not in recorded:
                            records.write(event+'\n'); recorded.add(event); totals['record_entries'] += 1
        with staging['rejected'].open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream); writer.writerow(['path','line','reason']); writer.writerows(rejected)
        bundle=getattr(predictor,'bundle',{})
        summary=dict(schema=3,input_type='files',status='complete' if totals['events_screened'] else 'no_valid_events',
                     started_utc=started,completed_utc=datetime.now(timezone.utc).isoformat(),directory=str(source),
                     outputs={key:str(path) for key,path in destinations.items()},model=predictor.spec.name,seed=predictor.seed,
                     threshold=effective_threshold,model_threshold=predictor.threshold,
                     threshold_source='model' if threshold is None else 'user',requested_filters=list(filters) if filters is not None else None,
                     resampling_enabled=bool(resample),target_grid=dict(start_s=-10.,end_s=40.,delta_s=.1,samples=501),
                     device=str(predictor.device),model_weights_sha256=bundle.get('weights_sha256'),
                     model_bundle_sha256=hashlib.sha256(json.dumps(bundle,sort_keys=True,allow_nan=False).encode()).hexdigest(),
                     **totals,rejected_or_unused_entries=len(rejected),source_files_sha256=source_hashes,
                     source_manifest_sha256=hashlib.sha256(json.dumps(source_hashes,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                     output_sha256={key:digest(staging[key]) for key in ['record','predictions','rejected','inputs']},
                     record_format='UTF-8; one unique retained EQR basename per line; no directory prefixes; no header',
                     input_contract='filename delta start_index exclusive_end_index waveform; P=0; observed -10..40 s; no SAC metadata inferred',
                     count_units='files_found: export files; waveform_rows_found: candidate rows; retained_files/files_resampled: waveform views',
                     selection='Joint event decision; align filter views by filename; good/bad export filenames are not labels')
        staging['summary'].write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        lock=target.parent/('.'+target.name+'.rfqc-lock')
        try: lock.mkdir()
        except FileExistsError:
            raise FileExistsError(f'Publication lock exists: {lock}; inspect outputs before removing a stale lock') from None
        try:
            for path in destinations.values():
                if path.is_symlink() or (path.exists() and (not overwrite or not path.is_file())):
                    raise FileExistsError(f'Output appeared during inference: {path}')
            for key in ['predictions','rejected','inputs','summary','record']: os.replace(staging[key],destinations[key])
        finally: lock.rmdir()
    return summary
