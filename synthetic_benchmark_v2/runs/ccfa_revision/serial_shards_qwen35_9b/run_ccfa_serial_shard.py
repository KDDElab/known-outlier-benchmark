#!/usr/bin/env python3
"""Run a disjoint, frozen shard with one-query greedy inference and raw logs.

This changes placement/scheduling only. A worker checks exact raw-text agreement
on six previously completed cases before producing publication predictions.
"""
import argparse
import json
import os
import time
import uuid
from pathlib import Path
import torch
import transformers
from prepare_ccfa_revision import OUT,PROTOCOL,read,sha
from annotate_v2 import ANNOTATORS,load_model,generate_one,parse_prediction,append_jsonl,utc_now,find_model_manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--inputs-path',type=Path,required=True)
    parser.add_argument('--output-path',type=Path,required=True)
    parser.add_argument('--fidelity-records',type=Path,required=True)
    args=parser.parse_args()
    cases=read(args.inputs_path);canonical={x['case_id']:x for x in read(OUT/'inputs_all_conditions.jsonl')}
    assert len(cases)==len({x['case_id'] for x in cases})
    assert all(x['prompt_sha256']==canonical[x['case_id']]['prompt_sha256'] for x in cases)
    existing=read(args.output_path) if args.output_path.exists() else []
    done={x['case_id'] for x in existing};assert len(done)==len(existing)
    spec=ANNOTATORS[args.model];path=Path(spec['model_path']);run_id=f'{args.model}_serial_shard_{uuid.uuid4().hex[:10]}'
    log=args.output_path.with_suffix('.runs.jsonl')
    metadata={'event':'started','run_id':run_id,'annotator_id':args.model,'model_id':spec['model_id'],
      'protocol_id':PROTOCOL,'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
      'inputs_sha256':sha(args.inputs_path),'runner_sha256':sha(__file__),
      'shared_loader_sha256':sha(Path(__file__).resolve().parent.parent/'annotate.py'),
      'parser_sha256':sha(Path(__file__).resolve().parent/'annotate_v2.py'),
      'inference_transformers_version':transformers.__version__,'torch_version':torch.__version__,
      'model_files':find_model_manifest(args.model,path),'model_path':str(path),
      'decoding':{'do_sample':False,'num_beams':1,'batch_size':1,'max_input_tokens':16384,'max_new_tokens':'frozen per case'},
      'n_cases':len(cases),'existing_records':len(existing),'started_at_utc':utc_now()}
    append_jsonl(log,metadata)
    model,tokenizer,config,dtype=load_model(path)
    mapping={str(k):str(v) for k,v in getattr(model,'hf_device_map',{}).items()}
    assert not any(v in {'cpu','disk'} for v in mapping.values()),'GPU-only execution required'
    append_jsonl(log,{'event':'loaded','run_id':run_id,'dtype':dtype,'device_map':mapping})
    prior=[r for r in read(args.fidelity_records) if r.get('raw_output') and not (r.get('parse_or_inference_error') or '').startswith('inference failed')]
    probes=sorted(prior,key=lambda r:r['case_id'])[:6];assert len(probes)==6
    fidelity=[]
    for r in probes:
        c=canonical[r['case_id']];raw,_,_=generate_one(model,tokenizer,c)
        fidelity.append({'case_id':r['case_id'],'same_raw_text':raw==r['raw_output'],'raw_output':raw,'reference_raw_output':r['raw_output']})
    args.output_path.with_suffix('.fidelity.json').write_text(json.dumps({'run_id':run_id,'cases':fidelity},indent=2)+'\n')
    assert all(r['same_raw_text'] for r in fidelity),'Replica raw fidelity differs; publication shard was not run'
    for i,c in enumerate(cases):
        if c['case_id'] in done:continue
        start=time.perf_counter();raw='';parsed=None;prompt_tokens=None;output_tokens=None
        try:
            raw,prompt_tokens,output_tokens=generate_one(model,tokenizer,c)
            parsed,error=parse_prediction(raw,set(c['feature_ids']),c.get('output_contract','full'))
        except torch.cuda.OutOfMemoryError:
            append_jsonl(log,{'event':'oom_stopped','run_id':run_id,'case_id':c['case_id'],'utc':utc_now()})
            raise
        except ValueError as exc:
            if 'exceeding 16384' not in str(exc):raise
            error=f'inference failed: ValueError: {exc}'
        record={'event':'annotation','run_id':run_id,'protocol_id':PROTOCOL,'annotator_id':args.model,'model_id':spec['model_id'],
          **{k:c.get(k) for k in ['case_id','source_case_id','generator_seed','condition','display_order_seed','split','prompt_sha256']},
          'timestamp_utc':utc_now(),'temperature':0.,'do_sample':False,'batch_size':1,'max_input_tokens':16384,
          'max_new_tokens':c.get('max_new_tokens',512),'output_contract':c.get('output_contract','full'),
          'prompt_tokens':prompt_tokens,'output_tokens':output_tokens,'output_limit_reached':output_tokens==c.get('max_new_tokens',512),
          'latency_seconds':time.perf_counter()-start,'raw_output':raw,'parsed_prediction':parsed,'parse_or_inference_error':error}
        append_jsonl(args.output_path,record)
        if i%10==0:print(f'{args.model} shard: {i+1}/{len(cases)}',flush=True)
    append_jsonl(log,{'event':'completed','run_id':run_id,'n_cases':len(cases),'completed_at_utc':utc_now()})

if __name__=='__main__':main()
