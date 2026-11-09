#!/usr/bin/env python3
"""Wait for two sufficiently free GPUs, then run the frozen DeepSeek queue.

This process never stops or changes another user's GPU worker.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from prepare_ccfa_revision import HERE, OUT

def main():
    status=OUT/'deepseek_queue_status.json'
    while True:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free=[tuple(map(int,line.split(','))) for line in raw.strip().splitlines()]
        eligible=sorted([(mem,index) for index,mem in free if mem>=20000],reverse=True)
        status.write_text(json.dumps({'state':'waiting_for_two_free_gpus','time_unix':time.time(),'free_mib':dict(free),'minimum_mib_each':20000},indent=2)+'\n')
        if len(eligible)>=2:break
        time.sleep(30)
    chosen=sorted([x[1] for x in eligible[:2]])
    env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,chosen)),
         'CCFA_GPU_MAX_MEMORY_GIB':'17','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
    # Preserve serial inference: actual Qwen/Llama diagnostics found batching
    # changed localization outputs, so batch inference was rejected.
    cmd=[sys.executable,str(HERE/'annotate_v2.py'),'--model','deepseek_v2_lite_chat',
         '--inputs-path',str(OUT/'pending_deepseek_v2_lite_chat.jsonl'),
         '--annotations-path',str(OUT/'new_deepseek_v2_lite_chat.jsonl'),
         '--run-log-path',str(OUT/'model_runs_deepseek_v2_lite_chat.jsonl'),
         '--protocol-id','known_outlier_ccfa_repaired_v1','--resume']
    status.write_text(json.dumps({'state':'running','gpus':chosen,'command':cmd,'time_unix':time.time()},indent=2)+'\n')
    with (OUT/'deepseek_v2_lite_chat.log').open('a') as log:
        process=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    status.write_text(json.dumps({'state':'finished' if process.returncode==0 else 'failed','returncode':process.returncode,
                                 'gpus':chosen,'command':cmd,'time_unix':time.time()},indent=2)+'\n')
    raise SystemExit(process.returncode)

if __name__=='__main__':main()
