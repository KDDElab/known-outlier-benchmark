"""Freeze output-schema intervention and audit archived malformed answers."""
from pathlib import Path
from datetime import datetime,timezone
import json,copy,hashlib,re,collections
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=HERE/'runs/ccfa_acceptance_revision'
def read(p):return [json.loads(x) for x in p.open() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if (OUT/'structured_protocol.json').exists():raise RuntimeError('Already frozen')
 rows=[]
 for old in read(HERE/'runs/ccfa_revision/inputs_all_conditions.jsonl'):
  if old.get('condition')!='minimal512':continue
  c=copy.deepcopy(old);c['paired_case_id']=old['case_id'];c['case_id']='structured_'+old['case_id'];c['condition']='structured_minimal512';c['protocol_id']='known_outlier_structured_v1';rows.append(c)
 assert len(rows)==96
 (OUT/'inputs_structured.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 protocol={'protocol_id':'known_outlier_structured_v1','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
 'n_cases':96,'comparison':'Same original minimal512 prompt, refs, target, order, model checkpoints and 512-token cap; add JSON Schema token filtering only. No size-three prior in either condition.',
 'decoder':'lm-format-enforcer==0.11.3; interegular==0.3.3; Transformers prefix_allowed_tokens_fn; greedy batch1; no sampling; default whitespace/key-order settings.',
 'schema':'object: explanations array 0..4 of {features: array 1..D of legal feature IDs}, abstain boolean; no additional properties. Uniqueness still validated by the original strict parser; grammar alone does not force unique entries.',
 'compatibility':'In-process alias from transformers.tokenization_utils_base.PreTrainedTokenizerBase to the old tokenization_utils location required by LMFE; no package downgrade or weight change.',
 'metrics':'Strict parse coverage, non-abstaining answered coverage, all-query Exact@1/Hit@4; conditional accuracy separately. Keep all 96 including input-limit rejections, and report tokenizer-common-feasible sensitivity.',
 'audit':'Archived errors inspected without replacing old raw outputs or scoring repaired strings. A diagnostic single missing brace insertion records syntactic recoverability only, not new accuracy.',
 'files':{n:sha(HERE/n) for n in ['annotate_v2.py','annotate_acceptance_structured.py','prepare_acceptance_structured.py']},'inputs_sha256':sha(OUT/'inputs_structured.jsonl')}
 (OUT/'structured_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 # Syntactic failure classification, never used to change original predictions.
 records=[r for r in read(HERE/'runs/ccfa_revision/new_qwen35_9b.jsonl') if r.get('condition')=='minimal512']
 details=[]
 for r in records:
  raw=r.get('raw_output','').replace('Ġ',' ').replace('Ċ','\n');obj=raw[raw.find('{'):] if '{' in raw else raw
  valid=False;repairable=False
  try:json.JSONDecoder().raw_decode(obj);valid=True
  except Exception:pass
  fixed,n=re.subn(r'(\])\s*(\]\s*,\s*"abstain"\s*:)',r'\1}\2',obj,count=1)
  if n and not valid:
   try:json.JSONDecoder().raw_decode(fixed);repairable=True
   except Exception:pass
  details.append({'case_id':r['case_id'],'generated':bool(raw),'json_valid':valid,'single_missing_final_candidate_brace':repairable,
    'output_tokens':r.get('output_tokens'),'limit_reached':r.get('output_limit_reached'), 'error':r.get('parse_or_inference_error')})
 (OUT/'qwen_minimal_syntax_audit.json').write_text(json.dumps({'n':len(details),'generated':sum(r['generated'] for r in details),
  'json_valid':sum(r['json_valid'] for r in details),'missing_final_candidate_brace':sum(r['single_missing_final_candidate_brace'] for r in details),
  'limit_reached':sum(bool(r['limit_reached']) for r in details),'details':details},indent=2)+'\n')
 print('structured inputs',len(rows),'brace pattern',sum(r['single_missing_final_candidate_brace'] for r in details))
if __name__=='__main__':main()
