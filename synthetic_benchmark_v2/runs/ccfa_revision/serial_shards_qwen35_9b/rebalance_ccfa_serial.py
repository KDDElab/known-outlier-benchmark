#!/usr/bin/env python3
"""Resume only unfinished queries on disjoint serial replicas after GPUs free.

Completed records, including failures, are preserved. Replica diagnostics never
enter predictions. This runner changes scheduling, not prompts or batch size.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import pandas as pd
from prepare_ccfa_revision import OUT,HERE,read,write,sha

def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split(') ',1)[1][0]!='Z'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--stop-pid',required=True,type=int)
    ap.add_argument('--wait-for-model',required=True)
    ap.add_argument('--wait-for-pid',type=int)
    ap.add_argument('--gpu-groups',nargs=2,required=True)
    ap.add_argument('--memory-gib',type=int,required=True)
    a=ap.parse_args()
    folder=OUT/f'serial_shards_{a.model}'
    folder.mkdir(exist_ok=False)
    status=folder/'status.json'
    def update(state,**kw):
        status.write_text(json.dumps({'state':state,'time_unix':time.time(),**kw},indent=2)+'\n')
    def count(path):
        return sum(1 for x in path.open() if x.strip()) if path.exists() else 0
    waiting=OUT/f'new_{a.wait_for_model}.jsonl'
    canonical=OUT/f'new_{a.model}.jsonl'
    while count(waiting)!=1055 or (a.wait_for_pid and alive(a.wait_for_pid)):
        update('waiting_for_serial_replica_capacity',dependency=a.wait_for_model,
               dependency_records=count(waiting),current_records=count(canonical))
        time.sleep(15)
    if count(canonical)==1055:
        update('already_complete_without_rebalancing');return
    assert alive(a.stop_pid),'Original worker missing; inspect before changing execution'
    cmd=Path(f'/proc/{a.stop_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert a.model in cmd and ('run_ccfa_batched_inference.py' in cmd or 'annotate_v2.py' in cmd),cmd
    # Record the current partial generation as an infrastructure interruption.
    # Its unfinished output is not a completed benchmark observation.
    interruption={'pid':a.stop_pid,'command':cmd,'completed_records_before_signal':count(canonical),
                  'reason':'Disjoint batch-size-one replicas after another model completed; preserve all completed outputs',
                  'signal':'SIGTERM','time_unix':time.time()}
    (folder/'interruption.json').write_text(json.dumps(interruption,indent=2)+'\n')
    os.kill(a.stop_pid,signal.SIGTERM)
    for _ in range(60):
        if not alive(a.stop_pid):break
        time.sleep(1)
    assert not alive(a.stop_pid),'Original worker did not exit; do not start overlapping writers'
    prefix=read(canonical);assert len(prefix)==len({x['case_id'] for x in prefix})
    shutil.copy2(canonical,folder/'prefix.jsonl');prefix_sha=sha(canonical)
    pending=read(OUT/f'pending_{a.model}.jsonl')
    ids={x['case_id'] for x in prefix}
    remaining=[x for x in pending if x['case_id'] not in ids]
    assert len(prefix)+len(remaining)==1055
    lengths=pd.read_csv(OUT/'token_lengths_all_conditions.csv')
    lengths=lengths[lengths.method.eq(a.model)].set_index('case_id').prompt_tokens.to_dict()
    weight=lambda x:lengths[x['case_id']]+10*x.get('max_new_tokens',512)
    shards=[[],[]];loads=[0,0]
    for x in sorted(remaining,key=lambda x:(-weight(x),x['case_id'])):
        i=min(range(2),key=lambda i:loads[i]);shards[i].append(x);loads[i]+=weight(x)
    contract={'model':a.model,'prefix_records':len(prefix),'prefix_sha256':prefix_sha,
              'shard_sizes':list(map(len,shards)),'gpu_groups':a.gpu_groups,
              'partition':'Descending prompt_tokens + 10*output_cap, greedily balanced; independent of prediction outcomes',
              'batch_size':1,'fidelity':'Six exact raw-text reproductions before official predictions on each replica',
              'scheduler_sha256':sha(__file__),'runner_sha256':sha(HERE/'run_ccfa_serial_shard.py')}
    (folder/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    for file in ['rebalance_ccfa_serial.py','run_ccfa_serial_shard.py','annotate_v2.py']:
        shutil.copy2(HERE/file,folder/file)
    children=[];logs=[]
    for i,(shard,gpus) in enumerate(zip(shards,a.gpu_groups)):
        ip=folder/f'inputs_{i}.jsonl';op=folder/f'predictions_{i}.jsonl';write(ip,shard)
        log=(folder/f'worker_{i}.log').open('w');logs.append(log)
        command=[sys.executable,str(HERE/'run_ccfa_serial_shard.py'),'--model',a.model,
                 '--inputs-path',str(ip),'--output-path',str(op),'--fidelity-records',str(folder/'prefix.jsonl')]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpus,'CCFA_GPU_MAX_MEMORY_GIB':str(a.memory_gib),
             'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
        children.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env))
    while any(c.poll() is None for c in children):
        update('serial_replicas_running',prefix_records=len(prefix),
               shard_counts=[count(folder/f'predictions_{i}.jsonl') for i in range(2)],
               shard_sizes=list(map(len,shards)),pids=[c.pid for c in children],
               returncodes=[c.poll() for c in children])
        time.sleep(15)
    for log in logs:log.close()
    if any(c.returncode for c in children):
        update('replica_failed_outputs_preserved',returncodes=[c.returncode for c in children])
        raise RuntimeError('Inspect replica logs; no automatic quality-based retry or partial publication merge')
    combined=prefix[:]
    for i,shard in enumerate(shards):
        output=read(folder/f'predictions_{i}.jsonl')
        assert len(output)==len(shard) and {x['case_id'] for x in output}=={x['case_id'] for x in shard}
        assert all(x['same_raw_text'] for x in json.loads((folder/f'predictions_{i}.fidelity.json').read_text())['cases'])
        combined+=output
    assert len(combined)==len({x['case_id'] for x in combined})==1055
    assert {x['case_id'] for x in combined}=={x['case_id'] for x in pending}
    assert sha(canonical)==prefix_sha,'Canonical prefix changed while disjoint replicas ran'
    tmp=canonical.with_suffix('.merge.tmp');write(tmp,combined);tmp.replace(canonical)
    update('completed_and_merged',records=1055,canonical_sha256=sha(canonical),
           retained_prefix_records=len(prefix),new_shard_records=len(remaining))

if __name__=='__main__':main()
