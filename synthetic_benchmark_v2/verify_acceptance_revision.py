"""Reconstruct frozen inputs and independently check labels and paired scores."""
from pathlib import Path
import json,hashlib,tempfile,itertools,importlib,csv
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;OUT=HERE/'runs/ccfa_acceptance_revision'
def read(p):return [json.loads(x) for x in p.open() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 checks={}
 with tempfile.TemporaryDirectory(prefix='known_outlier_reconstruction_') as temp:
  for module,files in [
   ('prepare_acceptance_revision',['inputs_prior3.jsonl']),
   ('prepare_acceptance_new_rules',['inputs_new_rules.jsonl','ground_truth_new_rules.jsonl','populations_new_rules.jsonl']),
   ('prepare_acceptance_structured',['inputs_structured.jsonl']),
   ('prepare_acceptance_real_cells',['inputs_real_cells.jsonl','ground_truth_real_cells.jsonl']),
   ('prepare_acceptance_real_structured',['inputs_real_cells_structured.jsonl'])]:
   m=importlib.import_module(module);m.OUT=Path(temp)/module;m.OUT.mkdir()
   m.main()
   for name in files:
    assert sha(m.OUT/name)==sha(OUT/name),name
    checks['reconstructed_'+name]=sha(OUT/name)
 # Separate brute-force projection-set calculation, without calling generator
 # or diagnostic minimal() helpers.
 populations={r['template_id']:r['complete_normal_population'] for r in read(OUT/'populations_new_rules.jsonl')}
 cases={r['case_id']:r for r in read(OUT/'inputs_new_rules.jsonl')};n=0
 for g in read(OUT/'ground_truth_new_rules.jsonl'):
  ids=sorted(cases[g['case_id']]['feature_ids']);normal=populations[g['template_id']];target=g['target_row']
  candidates=[frozenset(s) for z in range(1,7) for s in itertools.combinations(ids,z)]
  absent=[]
  for s in candidates:
   ordered=sorted(s);projections={tuple(r[f] for f in ordered) for r in normal}
   if tuple(target[f] for f in ordered) not in projections:absent.append(s)
  expected={s for s in absent if not any(t<s for t in absent)}
  assert expected=={frozenset(s) for s in g['rule_minimal_subspaces']},g['case_id']
  support={tuple(r[f] for f in ids) for r in normal}
  assert all(tuple(r[f] for f in ids) in support for r in g['normal_reference_rows'])
  if g['condition']=='complete':assert {tuple(r[f] for f in ids) for r in g['normal_reference_rows']}==support
  n+=1
 checks['new_rule_labels_and_reference_membership']=n
 # All new k3 baseline rankings must reproduce the original archived outputs.
 for experiment,path,condition,cap,original in [
  ('invoice','invoice_sweep.csv','all',8,HERE/'runs/ccfa_constraint_transfer/baselines'),
  ('main_d6','main_d6_search.csv','main_d6',3,HERE/'runs/ccfa_revision/gower')]:
  d=pd.read_csv(OUT/path);d=d[(d.condition==condition)&(d.cap==cap)&(d.k==3)]
  for method in ['knn','lof']:
   filename=f'predictions_gower_{method}_subspace.jsonl'
   if not (original/filename).exists():
    # Existing invoice package stores these in baseline_predictions.
    alternatives=list((HERE/'runs/ccfa_constraint_transfer').rglob(filename));assert len(alternatives)==1;op=alternatives[0]
   else:op=original/filename
   archived={r['case_id']:[tuple(sorted(e['features'])) for e in r['explanations']] for r in read(op)}
   sub=d[d.method==method]
   for r in sub.to_dict('records'):assert [tuple(s) for s in json.loads(r['top4'])]==archived[r['case_id']],r['case_id']
   checks[f'{experiment}_{method}_original_rankings']=len(sub)
 # New prior contains the identical payload, with one explicit additional rule.
 from run_gower_subspace_baseline import extract_case_data
 original={r['case_id']:r for r in read(HERE/'runs/ccfa_revision/inputs_panel.jsonl')}
 for c in read(OUT/'inputs_prior3.jsonl'):
  old=original[c['paired_case_id']]
  assert extract_case_data(c['user_prompt'])==extract_case_data(old['user_prompt'])
  assert c['feature_ids']==old['feature_ids'] and c['system_prompt']==old['system_prompt']
 checks['prior_payload_pairs']=96
 real={r['case_id']:r for r in read(OUT/'inputs_real_cells.jsonl')}
 for c in read(OUT/'inputs_real_cells_structured.jsonl'):
  old=real[c['paired_case_id']]
  for field in ['user_prompt','system_prompt','prompt_sha256','feature_ids','output_contract']:
   assert old[field]==c[field],(c['case_id'],field)
 checks['public_cell_identical_prompt_pairs']=72
 # Re-read published CSV strings independently of the pandas-based preparer.
 public={}
 for dataset in ['flights','beers','rayyan']:
  public[dataset]={}
  for version in ['dirty','clean']:
   with (OUT/'public_data'/f'{dataset}_{version}.csv').open(newline='') as stream:
    public[dataset][version]=[{k.replace('beer-name','beer_name').replace('brewery-name','brewery_name'):v for k,v in r.items()} for r in csv.DictReader(stream)]
 real_gold=read(OUT/'ground_truth_real_cells.jsonl')
 held={dataset:{r['source_row_index_zero_based'] for r in real_gold if r['dataset']==dataset} for dataset in public}
 for g in real_gold:
  dataset=g['dataset'];i=g['source_row_index_zero_based'];names=g['feature_names']
  dirty=public[dataset]['dirty'][i];clean=public[dataset]['clean'][i]
  expected={f for f,name in names.items() if dirty[name]!=clean[name]}
  assert expected==set(g['corrected_fields'])
  assert {f:dirty[name] for f,name in names.items()}==g['original_target']
  assert {f:clean[name] for f,name in names.items()}==g['corrected_target']
  data=extract_case_data(real[g['case_id']]['user_prompt'])
  assert data['target_row']==g['original_target']
  assert len(g['reference_row_indices'])==32
  for j,row in zip(g['reference_row_indices'],data['normal_reference_rows']):
   assert j not in held[dataset]
   expected_row=public[dataset]['clean'][j]
   assert row=={f:expected_row[name] for f,name in names.items()}
   if dataset=='flights':assert expected_row['flight']!=clean['flight']
 checks['independent_public_csv_labels_and_reference_exclusion']=72
 checks['beers_targets_with_ounces_canonicalization']=sum(any(c['field_name']=='ounces' for c in r['corrections']) for r in real_gold if r['dataset']=='beers')
 # Recompute native selected-field and patch diagnostics from stored vectors.
 for r in read(OUT/'native_m2oe_results.jsonl'):
  truth={4,5,6};assert set(np.flatnonzero(np.asarray(r['target']).reshape(-1)>1))==truth
  for prediction in r['outputs']:
   fields=set(prediction['fields']);patch=np.array(prediction['patch'])
   assert prediction['exact']==int(fields==truth)
   assert abs(prediction['recall']-len(fields&truth)/3)<1e-12
   assert abs(prediction['precision']-(len(fields&truth)/len(fields) if fields else 0))<1e-12
   outside=float(np.linalg.norm(patch-np.clip(patch,0,1)))
   assert abs(prediction['outside_cube_reduction']-(1-outside/np.sqrt(3)))<1e-12
  assert r['any_exact']==int(any(set(x['fields'])==truth for x in r['outputs']))
 checks['native_m2oe_field_and_patch_recomputation']=20
 result={'all_checks_passed':True,'scope':'Input reconstruction, independent finite-label enumeration, unchanged original Gower rankings and prior-payload pairing; not a second-machine replication.',
  'checks':checks,'runner_sha256':sha(Path(__file__))}
 (OUT/'reconstruction_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
