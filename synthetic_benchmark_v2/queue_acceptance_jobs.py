"""Wait for a preceding worker to exit, then run a frozen inference batch."""
import argparse,json,os,subprocess,time
from pathlib import Path
import psutil
def main():
 p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int,required=True);p.add_argument('--extra-wait-pid',type=int,nargs='*',default=[]);p.add_argument('--model',required=True);p.add_argument('--gpus',required=True);p.add_argument('--batch',required=True);p.add_argument('--runner',default='annotate_v2.py');p.add_argument('--protocol-id',default='known_outlier_new_compositions_v1');a=p.parse_args()
 here=Path(__file__).resolve().parent;out=here/'runs/ccfa_acceptance_revision'
 for preceding_pid in [a.wait_pid,*a.extra_wait_pid]:
  while psutil.pid_exists(preceding_pid):
   try:
    if psutil.Process(preceding_pid).status()==psutil.STATUS_ZOMBIE:break
   except psutil.NoSuchProcess:break
   time.sleep(15)
 env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpus,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
 if len(a.gpus.split(','))>1:env['CCFA_GPU_MAX_MEMORY_GIB']='20'
 cmd=['/opt/conda/bin/python',str(here/a.runner),'--model',a.model,'--inputs-path',str(out/f'inputs_{a.batch}.jsonl'),
  '--annotations-path',str(out/f'annotations_{a.batch}_{a.model}.jsonl'),'--run-log-path',str(out/f'model_runs_{a.batch}_{a.model}.jsonl'),
  '--protocol-id',a.protocol_id,'--resume']
 with (out/f'{a.model}_{a.batch}.log').open('a') as log:
  worker=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
  (out/f'worker_{a.batch}_{a.model}.json').write_text(json.dumps({'pid':worker.pid,'command':cmd,'previous_pid':a.wait_pid})+'\n')
  rc=worker.wait()
 print(json.dumps({'model':a.model,'batch':a.batch,'exit_code':rc}),flush=True)
if __name__=='__main__':main()
