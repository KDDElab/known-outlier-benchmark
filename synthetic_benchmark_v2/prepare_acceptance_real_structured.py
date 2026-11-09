"""Freeze paired schema decoding on the identical 72 public correction queries."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'runs/ccfa_acceptance_revision'
OUT=SOURCE
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if (OUT/'real_structured_protocol.json').exists():raise RuntimeError('Already frozen')
    rows=[]
    for line in (SOURCE/'inputs_real_cells.jsonl').open():
        old=json.loads(line);c=copy.deepcopy(old)
        c['paired_case_id']=old['case_id'];c['case_id']='structured_'+old['case_id']
        c['condition']='real_cells_structured';c['protocol_id']='public_corrected_cells_structured_v1'
        rows.append(c)
    assert len(rows)==72
    path=OUT/'inputs_real_cells_structured.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    protocol={'protocol_id':'public_corrected_cells_structured_v1','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
      'status':'Additional descriptive paired condition, fixed after observing malformed unconstrained public-cell outputs; not a preregistered comparison.',
      'comparison':'All 72 original public-cell queries, all three models; identical prompts, references, targets, schemas, order, 512-token cap and greedy decoding. Original strict results retained.',
      'schema':'The previously fixed LM Format Enforcer schema, with each candidate features array restricted to exactly one legal ID, matching the existing prompt. Zero to four candidate objects; abstain boolean. Original parser unchanged; repeated candidate IDs deduplicated in both conditions.',
      'selection':'No case selection, prompt tuning, repaired-output credit or model-specific configuration.',
      'metrics':'Same FieldHit@1/3, Recall@3, answer availability and offline first-corrected-field inspection position as the original public-cell protocol; no significance family.',
      'n_cases':72,'models':['qwen35_9b','deepseek_v2_lite_chat','llama31_8b_instruct'],
      'inputs_sha256':sha(path),'source_inputs_sha256':sha(SOURCE/'inputs_real_cells.jsonl'),
      'files':{n:sha(HERE/n) for n in ['annotate_v2.py','annotate_acceptance_structured.py','annotate_acceptance_real_structured.py','prepare_acceptance_real_structured.py']}}
    (OUT/'real_structured_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('frozen structured real-cell inputs',len(rows))
if __name__=='__main__':main()
