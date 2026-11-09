"""Freeze new Boolean compositions and paired reference evidence conditions."""
from pathlib import Path
from datetime import datetime,timezone
import itertools,random,json,hashlib,copy
from run_acceptance_gower_diagnostics import minimal
HERE=Path(__file__).resolve().parent
OUT=HERE/'runs/ccfa_acceptance_revision'
RULES=[
 ('parity4',4,'sum(x0..x3) mod 2 = 0',lambda x:sum(x[:4])%2==0),
 ('and_xor_pairs',4,'(x0 != x1) AND (x2 != x3)',lambda x:(x[0]!=x[1]) and (x[2]!=x[3])),
 ('or_and_pairs',4,'(x0 AND x1) OR (x2 AND x3)',lambda x:(x[0] and x[1]) or (x[2] and x[3])),
 ('xor_and_pairs',4,'(x0 AND x1) XOR (x2 AND x3)',lambda x:bool(x[0] and x[1])!=bool(x[2] and x[3])),
 ('exactly_two4',4,'sum(x0..x3) = 2',lambda x:sum(x[:4])==2),
 ('threshold_three4',4,'sum(x0..x3) >= 3',lambda x:sum(x[:4])>=3),
 ('implication_chain4',4,'x0 <= x1 <= x2 <= x3',lambda x:x[0]<=x[1]<=x[2]<=x[3]),
 ('multiplexer4',4,'x3 = (x1 if x0 else x2)',lambda x:x[3]==(x[1] if x[0] else x[2])),
 ('conditional_parity5',5,'x4 = (x1 XOR x2 if x0 else x3)',lambda x:x[4]==((x[1]^x[2]) if x[0] else x[3])),
 ('parity6',6,'sum(x0..x5) mod 2 = 0',lambda x:sum(x)%2==0),
 ('count_balance6',6,'sum(x0..x2) = sum(x3..x5)',lambda x:sum(x[:3])==sum(x[3:])),
 ('majority_output6',6,'x5 = [sum(x0..x4) >= 3]',lambda x:x[5]==int(sum(x[:5])>=3)),
]
def write(p,rows):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if (OUT/'new_rules_protocol.json').exists():raise RuntimeError('Already frozen')
 old=json.loads((HERE/'runs/ccfa_revision/inputs_main.jsonl').read_text().splitlines()[0])
 loc=old['user_prompt'].find('"feature_schema"');prefix=old['user_prompt'][:old['user_prompt'].rfind('{',0,loc)]
 rng=random.Random(882031);alltuples=list(itertools.product([0,1],repeat=6));inputs=[];gold=[];populations=[];sigs=set()
 for name,d,formula,rule in RULES:
  sig=tuple(bool(rule(x)) for x in alltuples)
  assert sig not in sigs;sigs.add(sig)
  essential=[j for j in range(6) if any(bool(rule(x))!=bool(rule(tuple((1-v) if k==j else v for k,v in enumerate(x)))) for x in alltuples)]
  assert essential==list(range(d)),(name,essential)
  normal=[x for x in alltuples if rule(x)];bad=[x for x in alltuples if not rule(x)];rng.shuffle(bad)
  ids=[f'F{i:03d}' for i in range(1,7)];role_ids=ids.copy();rng.shuffle(role_ids)
  display=ids.copy();rng.shuffle(display)
  def row(x):return {fid:x[role_ids.index(fid)] for fid in display}
  population=[row(x) for x in normal]
  populations.append({'template_id':name,'normal_formula':formula,'essential_roles':essential,'role_to_feature':dict(enumerate(role_ids)),
    'complete_normal_population':population,'truth_table':list(map(int,sig))})
  for j,x in enumerate(bad[:4]):
   target=row(x);tid=f'new_{name}_{j:02d}';truth=minimal(target,population,ids)
   conditions=['draw0']+(['draw1','draw2','complete'] if j<2 else [])
   for condition in conditions:
    refs=copy.deepcopy(population) if condition=='complete' else [copy.deepcopy(rng.choice(population)) for _ in range(32)]
    # Enumeration is the full visible six-field population, including every
    # binary distractor combination. No population members are marginalized.
    if condition=='complete':rng.shuffle(refs)
    payload={'feature_schema':[{'feature_id':f,'value_type':'binary','allowed_values':[0,1]} for f in display],
      'normal_reference_rows':refs,'target_row':target}
    note=('The following reference rows enumerate the COMPLETE normal population over the declared finite domains; there are no other normal rows.\n\n' if condition=='complete' else '')
    c={'case_id':tid+'_'+condition,'target_id':tid,'source_case_id':tid,'split':'test','condition':'new_rules_'+condition,
      'feature_ids':display,'feature_count':6,'reference_count':len(refs),'generator_seed':882031,'field_name_mode':'opaque',
      'nuisance_profile':'independent_binary','system_prompt':old['system_prompt'],'user_prompt':prefix+note+json.dumps(payload,indent=2),
      'output_contract':'full','max_new_tokens':512,'protocol_id':'known_outlier_new_compositions_v1'}
    c['prompt_sha256']=hashlib.sha256((c['system_prompt']+'\0'+c['user_prompt']).encode()).hexdigest();inputs.append(c)
    sample=minimal(target,refs,ids)
    gold.append({'case_id':c['case_id'],'target_id':tid,'template_id':name,'condition':condition,'task_label':'higher_order',
      'rule_minimal_subspaces':truth,'reference_support_sufficient_minimal_subspaces':sample,'feature_count':6,
      'min_truth_size':min(map(len,truth)),'truth_alternatives':len(truth),
      'extra_sample_minima':len({tuple(s) for s in sample}-{tuple(s) for s in truth}),
      'truth_retention':len({tuple(s) for s in sample}&{tuple(s) for s in truth})/len(truth),
      'population_count':len(population),'target_row':target,'normal_reference_rows':refs,'reference_count':len(refs),
      'mechanism_feature_ids':sorted(role_ids[:d]),'normal_formula':formula})
 write(OUT/'inputs_new_rules.jsonl',inputs);write(OUT/'ground_truth_new_rules.jsonl',gold);write(OUT/'populations_new_rules.jsonl',populations)
 proto={'protocol_id':'known_outlier_new_compositions_v1','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
 'design':'12 new Boolean compositions with 4-6 essential roles in six visible binary fields. Original generator rules have at most three essential roles. These are newly designed after review, not externally blind data.',
 'targets':'4 uniformly shuffled unique invalid full rows per rule, chosen before inference; 48 targets, no outcome filtering.',
 'reference_pairs':'First 2 targets per rule (24) have three independent with-replacement 32-row draws and one complete population. All 48 have draw0; total 120 inputs.',
 'complete_condition':'All six-field normal tuples, including all nuisance combinations, explicitly identified as complete in the prompt. This estimates complete-information performance, combining complete support coverage and a completeness declaration. The unique complete populations have 16-32 rows; sampled conditions have 32 rows with replacement.',
 'frozen_methods':'Original three checkpoints, original full prompt, strict parser, greedy 512-token decode and input cap16384; Gower all63 subsets k3. No method selection/tuning on these outputs.',
 'novelty_check':'Distinct 64-bit normal truth tables; every named rule has >=4 essential coordinates. Does not claim external curation or independence from all researcher design choices.',
 'endpoints':'Exact@1/Hit@4, all-query denominator, equal weight per new rule; draw0 all48 is new-rule result. 24-target paired evidence table compares draw0/1/2/complete; report target-level answer instability and method ordering for each draw.',
 'strata':'Minimum truth size 1-2/3/4-6; number of valid minimal sets 1/2-4/5+; extra sample minima 0/1-3/4+. Report empty bins, no posthoc boundary tuning.',
 'seed':882031,'n_inputs':len(inputs),'n_targets':48,'n_templates':12,
 'code_hashes':{n:sha(HERE/n) for n in ['annotate_v2.py','run_gower_subspace_baseline.py','prepare_acceptance_new_rules.py']},
 'files':{n:sha(OUT/n) for n in ['inputs_new_rules.jsonl','ground_truth_new_rules.jsonl','populations_new_rules.jsonl']}}
 (OUT/'new_rules_protocol.json').write_text(json.dumps(proto,indent=2)+'\n')
 print(json.dumps({'inputs':len(inputs),'targets':48,'populations':[len(p['complete_normal_population']) for p in populations]}))
if __name__=='__main__':main()
