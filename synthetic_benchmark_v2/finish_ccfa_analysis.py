#!/usr/bin/env python3
"""Finish analysis/figures only after the complete frozen experiment queue."""
import json
import os
import subprocess
import sys
import time
from prepare_ccfa_revision import OUT,HERE,MODELS

def main():
    status=OUT/'analysis_queue_status.json'
    while True:
        counts={}
        for model in MODELS:
            p=OUT/f'new_{model}.jsonl'
            counts[model]=sum(1 for line in p.open() if line.strip()) if p.exists() else 0
        complete=all(v==1055 for v in counts.values()) and (OUT/'m2oe/predictions_m2oe_choice.manifest.json').exists()
        status.write_text(json.dumps({'state':'ready' if complete else 'waiting_for_complete_runs','new_record_counts':counts,'required_each':1055,'updated_unix':time.time()},indent=2)+'\n')
        if complete:break
        time.sleep(30)
    commands=[[sys.executable,str(HERE/name)] for name in ['analyze_ccfa_revision.py','analyze_ccfa_supplements.py','analyze_ccfa_secondary.py']]
    commands += [[sys.executable,str(HERE.parent/'figures/make_manuscript_figures.py')],
                 [sys.executable,str(HERE.parent/'figures/make_manuscript_result_figures.py')]]
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
    for command in commands:
        status.write_text(json.dumps({'state':'running','command':command,'updated_unix':time.time()},indent=2)+'\n')
        name=command[-1].split('/')[-1]
        with (OUT/f'completed_{name}.log').open('w') as log:
            r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        if r.returncode:
            status.write_text(json.dumps({'state':'failed','command':command,'returncode':r.returncode},indent=2)+'\n')
            raise SystemExit(r.returncode)
    status.write_text(json.dumps({'state':'complete_analysis_and_figures','updated_unix':time.time()},indent=2)+'\n')

if __name__=='__main__':main()
