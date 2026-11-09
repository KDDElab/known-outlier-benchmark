"""Exploratory correction-type audit; never changes source labels or predictions.

Row-local certificates identify explicit quantity/null/location normalization.
Time-value changes and source-specific bibliography conventions remain separate;
absence of a certificate does not prove impossibility from reference data.
"""
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import hashlib,json,re
import numpy as np
import pandas as pd
from prepare_unified_revision import OUT,HERE,read,write
from analyze_acceptance_revision import parsed

ACC=HERE/'runs/ccfa_acceptance_revision'
def time_minutes(s):
    m=re.search(r'\b(\d{1,2}):(\d{2})\s*([ap])\.?m\.?',s,re.I)
    if m:return (int(m[1])%12+(12 if m[3].lower()=='p' else 0))*60+int(m[2])
    m=re.search(r'\b(\d{1,2}):(\d{2})\b',s)
    return int(m[1])*60+int(m[2]) if m else None

def classify(g,c):
    field=c['field_name'];a=str(c['original']);b=str(c['corrected'])
    if g['dataset']=='beers':
        if field=='ounces':
            m=re.match(r'^\s*([0-9.]+)\s*(?:oz\.?|ounce)',a,re.I)
            if m and Decimal(m[1])==Decimal(b):return 'quantity_representation','row_local_quantity'
        if a=='N/A' and b=='':return 'missing_marker','row_local_null'
        names={v:k for k,v in g['feature_names'].items()}
        city=g['original_target'][names['city']];clean_city=g['corrected_target'][names['city']];state=g['corrected_target'][names['state']]
        if field in ['city','state'] and city==clean_city+' '+state and re.fullmatch('[A-Z]{2}',state):
            return 'location_split','row_local_cross_field'
        if field=='abv' and a.endswith('%'):return 'percentage_convention','source_convention_required'
    if g['dataset']=='flights':
        x,y=time_minutes(a),time_minutes(b)
        if x is not None and x==y:return 'clock_preserving_format','source_convention_required'
        return ('time_imputation' if x is None else 'time_value_change'),'external_fact_required'
    if g['dataset']=='rayyan':
        if field=='article_jcreated_at':return 'date_serialization','source_convention_required'
        if field in ['author_list','journal_title']:return 'encoding_normalization','source_convention_required'
        if a=='':return 'missing_sentinel','source_convention_required'
        return 'bibliographic_serialization','source_convention_required'
    return 'other_value_change','not_certified'

def main():
    gs=read(ACC/'ground_truth_real_cells.jsonl');cells=[];rows=[];byid={g['case_id']:g for g in gs}
    (OUT/'correction_audit_protocol.json').write_text(json.dumps({'created_at_utc':datetime.now(timezone.utc).isoformat(),
      'scope':'Posthoc descriptive source-label audit; classifications fixed in source code, no new annotations or changed gold.',
      'row_local':'Recognized ounces unit stripping with equal quantity, explicit missing-marker normalization, and matching city/state split.',
      'not_row_local':'Percentage interpretation, date/encoding conventions, and actual flight times are not independently certified.',
      'external_fact':'Flight time changes/imputation are recorded event facts, and same-flight reference records were excluded.',
      'non_ounces':'Beers targets with >=1 corrected field other than ounces; original predictions unchanged, only acceptable-target set narrowed.',
      'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    for g in gs:
        kinds=[]
        for c in g['corrections']:
            kind,cert=classify(g,c);kinds.append(cert)
            cells.append({'case_id':g['case_id'],'source':g['dataset'],**c,'correction_type':kind,'evidence_class':cert})
        local=sum(x.startswith('row_local') for x in kinds)
        group='row_local_only' if local==len(kinds) else 'mixed' if local else 'external_or_source'
        rows.append({'case_id':g['case_id'],'source':g['dataset'],'stratum':group,'n_corrections':len(kinds),'row_local_certificates':local})
    write(OUT/'correction_cells_audit.jsonl',cells);pd.DataFrame(rows).to_csv(OUT/'correction_rows_audit.csv',index=False)
    results=[];slices=[]
    models=['qwen35_9b','deepseek_v2_lite_chat','llama31_8b_instruct']
    inputs={c['source_case_id'] if 'source_case_id' in c else c['case_id']:c for c in read(ACC/'inputs_real_cells_structured.jsonl')}
    # Stable mapping uses input paired IDs when available, not output position.
    inputs_list=read(ACC/'inputs_real_cells_structured.jsonl')
    for method in models+['gower_knn_subspace','gower_lof_subspace']:
        if method in models:
            ps={r['case_id']:r for r in read(ACC/f'annotations_real_cells_structured_{method}.jsonl')}
            mapped={c.get('paired_case_id',c.get('source_case_id')):(ps[c['case_id']],c) for c in inputs_list}
        else:
            ps={r['case_id']:r for r in read(ACC/f'real_cells_baselines/predictions_{method}.jsonl')}
            cs={c['case_id']:c for c in read(ACC/'inputs_real_cells.jsonl')};mapped={cid:(p,cs[cid]) for cid,p in ps.items()}
        for row in rows:
            cid=row['case_id'];g=byid[cid];record,case=mapped[cid];p=parsed(record,case)
            choices=[e['features'][0] for e in p['explanations'] if len(e['features'])==1] if p and not p.get('abstain') else []
            choices=list(dict.fromkeys(choices));truth=set(g['corrected_fields'])
            results.append({**row,'method':method,'hit1':int(bool(choices) and choices[0] in truth),'hit3':int(bool(set(choices[:3])&truth)),
                            'recall3':len(set(choices[:3])&truth)/len(truth)})
            if g['dataset']=='beers':
                rest={f for f in truth if g['feature_names'][f]!='ounces'}
                if rest:slices.append({'case_id':cid,'method':method,'hit1':int(bool(choices) and choices[0] in rest),
                    'hit3':int(bool(set(choices[:3])&rest)),'recall3':len(set(choices[:3])&rest)/len(rest)})
    df=pd.DataFrame(results);df.to_csv(OUT/'correction_strata_per_case.csv',index=False)
    summary=df.groupby(['source','stratum','method']).agg(n=('hit1','size'),hit1=('hit1','mean'),hit1_sd=('hit1','std'),hit3=('hit3','mean'),recall3=('recall3','mean'))
    summary.to_csv(OUT/'correction_strata_summary.csv')
    pd.DataFrame(slices).to_csv(OUT/'beers_non_ounces_per_case.csv',index=False)
    pd.DataFrame(slices).groupby('method').agg(n=('hit1','size'),hit1=('hit1','mean'),hit3=('hit3','mean'),recall3=('recall3','mean')).to_csv(OUT/'beers_non_ounces_summary.csv')
    counts=pd.DataFrame(cells).groupby(['source','correction_type','evidence_class']).size().rename('n_cells')
    counts.to_csv(OUT/'correction_type_counts.csv');print(counts.to_string());print(summary.to_string())
if __name__=='__main__':main()
