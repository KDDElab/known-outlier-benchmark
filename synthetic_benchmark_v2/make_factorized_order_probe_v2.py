#!/usr/bin/env python3
"""Create matched row-only and field-order-only variants of the 96-case probe."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
RUNS=HERE/'runs'
SOURCE=RUNS/'order_probe_original_inputs.jsonl'
GT=HERE/'ground_truth.jsonl'
SELECTION=RUNS/'order_probe_selection_manifest.json'

def payload(case):
    prompt=case['user_prompt']; marker='\n{\n  "feature_schema"'; offset=prompt.find(marker)
    if offset<0: raise ValueError(f'payload marker missing: {case["case_id"]}')
    prefix,text=prompt[:offset+1],prompt[offset+1:]
    value=json.loads(text)
    return prefix,value

def make_variant(case, kind):
    row=dict(case); prefix,p=payload(row)
    if kind=='field_order_only':
        p['feature_schema']=list(reversed(p['feature_schema']))
        p['target_row']=dict(reversed(list(p['target_row'].items())))
        p['normal_reference_rows']=[dict(reversed(list(r.items()))) for r in p['normal_reference_rows']]
        protocol='known_anomaly_subspace_v2_field_order_only_reverse_v1'
    elif kind=='reference_row_order_only':
        p['normal_reference_rows']=list(reversed(p['normal_reference_rows']))
        protocol='known_anomaly_subspace_v2_reference_row_order_only_reverse_v1'
    else: raise ValueError(kind)
    row['user_prompt']=prefix+json.dumps(p,ensure_ascii=False,indent=2)
    row['prompt_sha256']=hashlib.sha256((row['system_prompt']+'\0'+row['user_prompt']).encode()).hexdigest()
    row['protocol_id']=protocol
    return row

def write(path, rows):
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')

rows=[json.loads(x) for x in SOURCE.read_text().splitlines() if x.strip()]
byid={x['case_id']:x for x in rows}
truth={json.loads(x)['case_id']:json.loads(x) for x in GT.read_text().splitlines() if x.strip()}
ids=set(byid)
out=RUNS/'factorized_order_probe'; out.mkdir(parents=True,exist_ok=True)
write(out/'inputs_field_order_only.jsonl',[make_variant(x,'field_order_only') for x in rows])
write(out/'inputs_reference_row_order_only.jsonl',[make_variant(x,'reference_row_order_only') for x in rows])
control=[]
for x in rows:
    row=dict(x); row['protocol_id']='known_anomaly_subspace_v2_factorized_order_control_v1'; control.append(row)
write(out/'inputs_control.jsonl',control)
write(out/'ground_truth_96.jsonl',[truth[x] for x in byid])
(out/'probe_manifest.json').write_text(json.dumps({'n_cases':len(rows),'source_protocol':'known_anomaly_subspace_benchmark_v2_order_probe_original_v1','source_selection_manifest_sha256':hashlib.sha256(SELECTION.read_bytes()).hexdigest(),'variants':{'field_order_only':'Reverse schema and within-row field order; preserve reference-row order.','reference_row_order_only':'Reverse only the outer normal-reference row order; preserve schema and within-row field order.'}},indent=2)+'\n')
print(f'created factorized probes: {len(rows)} cases x 2 variants at {out}')
