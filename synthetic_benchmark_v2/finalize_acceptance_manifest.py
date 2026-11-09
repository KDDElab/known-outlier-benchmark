"""Record complete cohorts, protocol timing, runtime settings and release hashes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,collections
HERE=Path(__file__).resolve().parent;OUT=HERE/'runs/ccfa_acceptance_revision'
MODELS=['qwen35_9b','deepseek_v2_lite_chat','llama31_8b_instruct']
PROTOCOLS={'prior3':'protocol.json','new_rules':'new_rules_protocol.json',
 'structured':'structured_protocol.json','real_cells':'real_cells_protocol.json',
 'real_cells_structured':'real_structured_protocol.json'}
def read(p):return [json.loads(x) for x in p.open() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 summary=json.loads((OUT/'summary.json').read_text());assert summary['complete']
 assert json.loads((OUT/'reconstruction_verification.json').read_text())['all_checks_passed']
 batches=[];total=0
 for batch,filename in PROTOCOLS.items():
  protocol=json.loads((OUT/filename).read_text());frozen=protocol.get('frozen_at_utc',protocol.get('created_at_utc'))
  assert frozen,(filename,list(protocol))
  inputs={r['case_id']:r for r in read(OUT/f'inputs_{batch}.jsonl')}
  for model in MODELS:
   path=OUT/f'annotations_{batch}_{model}.jsonl';rows=read(path)
   starts=[r for r in read(OUT/f'model_runs_{batch}_{model}.jsonl') if r.get('event')=='started']
   assert starts and all(datetime.fromisoformat(r['started_at_utc'])>=datetime.fromisoformat(frozen) for r in starts)
   assert all(r['inputs_sha256']==sha(OUT/f'inputs_{batch}.jsonl') for r in starts)
   assert len(rows)==len(inputs) and len({r['case_id'] for r in rows})==len(inputs)
   assert min(datetime.fromisoformat(r['timestamp_utc']) for r in rows)>=datetime.fromisoformat(frozen)
   for r in rows:
    assert r['prompt_sha256']==inputs[r['case_id']]['prompt_sha256']
    assert r['do_sample'] is False and r['max_input_tokens']==16384 and r['max_new_tokens']==512
    assert r['protocol_id']==protocol['protocol_id']
   batches.append({'batch':batch,'model':model,'records':len(rows),
    'actual_generations':sum(bool(r['raw_output']) for r in rows),
    'errors':dict(collections.Counter(r.get('parse_or_inference_error') or 'none' for r in rows)),
    'first_record_utc':min(r['timestamp_utc'] for r in rows),'last_record_utc':max(r['timestamp_utc'] for r in rows),
    'first_run_started_utc':min(r['started_at_utc'] for r in starts),
    'protocol_frozen_at_utc':frozen,'predictions_sha256':sha(path)})
   total+=len(rows)
 assert total==1368
 files=[p for p in OUT.iterdir() if p.is_file() and not p.name.startswith(('partial_','queue_','worker_'))
        and p.suffix in ['.csv','.json','.jsonl','.tex'] and p.name!='analysis_manifest.json']
 result={'complete':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),
  'model_condition_records':total,'cohorts':batches,
  'native_m2oe_records':len(read(OUT/'native_m2oe_results.jsonl')),
  'native_sod_records':len(read(OUT/'native_sod_results.jsonl')),
  'current_files_sha256':{p.name:sha(p) for p in sorted(files)},
  'analysis_code_sha256':{p.name:sha(p) for p in sorted(HERE.glob('*acceptance*.py'))},
  'not_claimed':['new human annotations or observed analyst benefit','public reviewer access','independent-machine reproduction'],
  'scope':'Completed reviewer-requested cohorts. Original frozen protocols and declared post-hoc corrections retain their own chronology. Output failures remain in all-query denominators.'}
 (OUT/'analysis_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'complete':True,'records':total,'batches':len(batches)}))
if __name__=='__main__':main()
