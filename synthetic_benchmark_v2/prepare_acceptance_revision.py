"""Freeze reviewer-requested diagnostics, retaining the previous experiment files."""
from pathlib import Path
import copy,hashlib,json
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
OUT=HERE/'runs/ccfa_acceptance_revision'
PROTOCOL='known_outlier_acceptance_diagnostics_v1'
def read(p):return [json.loads(l) for l in Path(p).open() if l.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,rows):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def stamp(c,condition):
 c['condition']=condition;c['source_case_id']=c.get('source_case_id',c['case_id'])
 c['prompt_sha256']=hashlib.sha256((c['system_prompt']+'\0'+c['user_prompt']).encode()).hexdigest()
 c['protocol_id']=PROTOCOL;return c
CAP='\nShared candidate rule for this condition: every inclusion-minimal explanation has at most three fields. Return at most four DISTINCT candidate subsets, each containing one to three distinct feature IDs. Rank the candidates from most to least plausible. The same one-to-three-field search space and four-candidate budget are used by the statistical baselines. This statement gives no field identities or labels.\n\n'
def main():
 OUT.mkdir(exist_ok=True)
 if (OUT/'protocol.json').exists():raise RuntimeError('Protocol already frozen')
 src=HERE/'runs/ccfa_revision/inputs_panel.jsonl'
 rows=[]
 for old in read(src):
  c=copy.deepcopy(old);c['paired_case_id']=old['case_id'];c['case_id']='prior3_'+old['case_id']
  loc=c['user_prompt'].find('"feature_schema"');start=c['user_prompt'].rfind('{',0,loc)
  c['user_prompt']=c['user_prompt'][:start]+CAP+c['user_prompt'][start:]
  c['candidate_size_cap']=3;c['max_new_tokens']=512;c['output_contract']='full'
  rows.append(stamp(c,'shared_prior3'))
 write(OUT/'inputs_prior3.jsonl',rows)
 protocol={'protocol_id':PROTOCOL,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
  'review_scope':['shared size prior','complete/replicated reference evidence','invoice rank reversal','new rule compositions','structured output and adapters','auditable real cells'],
  'prior_pairing':'All 96 existing outcome-independent first-seed test targets with 32 references; same rows, field order, full output contract, checkpoints, greedy batch one, 16384 input and 512 output limits; add only an explicit size/candidate rule.',
  'prior_rules':{'max_candidate_size':3,'max_distinct_candidates':4,'oversized_or_empty_subsets':'invalid shared-contract output, all-query miss; additionally report cap compliance separately','duplicates':'collapse repeated candidates in first-occurrence order; no re-ranking'},
  'prior_comparison':'Paired with original full-contract 512-token records and same cap-3 Gower predictions. No posthoc truncation of LLM sets. All 96 targets remain in denominators; tokenizer-defined common-feasible subset is secondary.',
  'gower_small_width':'All 288 six-field main cases: compare cap 3 versus all 63 nonempty subsets, same references and k=3.',
  'invoice_mechanism':'All 96 frozen invoice targets: max-order 1/2/3/4/8, k=1/3/5/10, distractors retained/removed (oracle diagnostic only). Record target-reference zero distances, duplicate projections, ties, raw scores and gold ranks. Add within-subspace reference-rank calibration as a diagnostic, not a renamed original baseline.',
  'reference_diagnostics':'A separately frozen small finite-domain, complete-population workload and repeated iid reference samples for the same targets, preserving order/schema/gold; definitions and counts must freeze before outputs. Distinguish full visible population from a projection with distractors marginalized.',
  'new_rules':'Freeze model/prompt/parser/scoring versions before inference on new rule compositions. No method tuning on their outputs. Newly designed after review, not independently externally curated; report limited independence accurately.',
  'structured_output':'Keep original full/minimal results. Add a fixed constrained-output protocol on the 96 matched targets, with full raw logs and separate parse coverage/content metrics. Freeze schema and decoder before outputs; no repair conditioned on correctness.',
  'statistics':'Primary paired prior endpoints Exact@1/Hit@4; 24-template aggregation, 2000 template bootstraps, 20000 paired sign flips; six model-by-endpoint tests form one Holm family. Diagnostic sweeps descriptive; no selection of best setting as a new main result.',
  'source_panel_sha256':sha(src),'inputs_prior3_sha256':sha(OUT/'inputs_prior3.jsonl'),
  'frozen_method_code_sha256':{n:sha(HERE/n) for n in ['annotate_v2.py','run_gower_subspace_baseline.py','evaluate_benchmark.py','run_sod_cop_adaptations_v2.py','run_m2oe_baseline_v2.py']},
  'frozen_loader_sha256':sha(HERE.parent/'annotate.py'),'generator_sha256':sha(__file__),
  'models':['qwen35_9b','deepseek_v2_lite_chat','llama31_8b_instruct'],
  'human_annotations':'No participant annotations or time-saving outcomes may be fabricated; public reviewed cell corrections are a separate target from minimal support.'}
 (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'protocol':PROTOCOL,'prior_cases':len(rows),'output':str(OUT)}))
if __name__=='__main__':main()
