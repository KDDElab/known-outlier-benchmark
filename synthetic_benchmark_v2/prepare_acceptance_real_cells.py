"""Public corrected-cell inspection targets, distinct from support explanations."""
from pathlib import Path
from datetime import datetime,timezone
import json,random,hashlib,csv
import pandas as pd
HERE=Path(__file__).resolve().parent;OUT=HERE/'runs/ccfa_acceptance_revision';DATA=OUT/'public_data'
def write(p,rows):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def main():
 if (OUT/'real_cells_protocol.json').exists():raise RuntimeError('Already frozen')
 rng=random.Random(930761);inputs=[];gold=[];audit=[];dataset_info=[]
 for name,drop in [('flights',['tuple_id']),('beers',['index','id']),('rayyan',['id'])]:
  dirty=pd.read_csv(DATA/f'{name}_dirty.csv',dtype=str,keep_default_na=False)
  clean=pd.read_csv(DATA/f'{name}_clean.csv',dtype=str,keep_default_na=False)
  assert dirty.shape==clean.shape
  original_clean_cols=clean.columns.tolist()
  # The upstream Beers clean header uses beer-name/brewery-name while dirty
  # uses underscores. Align these two documented equivalent names only.
  clean=clean.rename(columns={'beer-name':'beer_name','brewery-name':'brewery_name'})
  assert dirty.columns.tolist()==clean.columns.tolist(),name
  idcol='tuple_id' if name=='flights' else 'id'
  assert (dirty[idcol].to_numpy()==clean[idcol].to_numpy()).all()
  cols=[c for c in dirty.columns if c not in drop];diff=dirty[cols].ne(clean[cols]);eligible=diff.any(axis=1)
  buckets=[[int(i) for i in dirty.index[eligible & (diff.sum(axis=1)==k)]] for k in [1,2]]
  buckets.append([int(i) for i in dirty.index[eligible & (diff.sum(axis=1)>=3)]])
  chosen=[]
  for b in buckets:rng.shuffle(b);chosen.extend(b[:8])
  remainder=[int(i) for i in dirty.index[eligible] if int(i) not in chosen];rng.shuffle(remainder);chosen+=remainder[:24-len(chosen)]
  assert len(chosen)==24
  # All held-out rows are removed before drawing any clean references.
  refpool=[int(i) for i in clean.index if int(i) not in chosen]
  ids=[f'F{i+1:03d}' for i in range(len(cols))];mapping=dict(zip(cols,ids))
  for rank,i in enumerate(chosen):
   pool=[j for j in refpool if name!='flights' or clean.at[j,'flight']!=clean.at[i,'flight']]
   refids=rng.sample(pool,32);display=ids.copy();rng.shuffle(display)
   def row(df,j):return {f:str(df.at[j,cols[ids.index(f)]]) for f in display}
   refs=[row(clean,j) for j in refids];target=row(dirty,i)
   payload={'feature_schema':[{'feature_id':f,'field_name':cols[ids.index(f)],'value_type':'categorical'} for f in display],
    'normal_reference_rows':refs,'target_row':target}
   prompt=('The target row is known to contain at least one erroneous or noncanonical field value. '
    'Prioritize fields for a human to inspect using the schema and the clean reference rows. '
    'Return up to four DISTINCT field IDs, most suspicious first. Each candidate must be a SINGLE field. '
    'Do not repair values. A field may be erroneous even when its value appears elsewhere. '
    'Return one JSON object with exactly explanations and abstain. explanations is an ordered array of objects, '
    'each containing only features, a one-element array with a legal feature ID. abstain is boolean. '
    'Do not include analysis or a preamble. Example: {"explanations":[{"features":["F001"]}],"abstain":false}.\n\n')
   cid=f'realcell_{name}_{rank:03d}'
   c={'case_id':cid,'source_case_id':cid,'split':'test','condition':'public_corrected_cells','feature_ids':display,'feature_count':len(ids),
    'reference_count':32,'system_prompt':'Prioritize fields for inspection in a known dirty table row. Use the supplied schema and clean references. Return one JSON object only.',
    'user_prompt':prompt+json.dumps(payload,ensure_ascii=False,indent=2),'output_contract':'minimal','max_new_tokens':512,'protocol_id':'public_corrected_cells_v1'}
   c['prompt_sha256']=hashlib.sha256((c['system_prompt']+'\0'+c['user_prompt']).encode()).hexdigest();inputs.append(c)
   errors=[mapping[col] for col in cols if diff.at[i,col]]
   g={'case_id':cid,'dataset':name,'source_row_index_zero_based':i,'source_record_id':str(dirty.at[i,idcol]),'feature_names':{v:k for k,v in mapping.items()},
    'corrected_fields':errors,'valid_single_field_inspection_targets':[[f] for f in errors],
    'original_target':target,'corrected_target':row(clean,i),'reference_row_indices':refids,'feature_count':len(ids),
    'corrections':[{'feature_id':mapping[col],'field_name':col,'original':str(dirty.at[i,col]),'corrected':str(clean.at[i,col])} for col in cols if diff.at[i,col]],
    'label_scope':'Public benchmark cell corrections, including canonical formatting; not minimal explanation sets, causal responsibility, or new participant annotations.'}
   gold.append(g);audit.append({'case_id':cid,'dataset':name,'row_index':i,'rater_id':'','valid_explanation_sets_json':'','selected_fields_json':'','basis_or_source':'','disagreement_reason':'','seconds_spent':'','confidence':''})
  dataset_info.append({'dataset':name,'rows':len(dirty),'fields_evaluated':cols,'removed_metadata_fields':drop,'clean_header_original':original_clean_cols,
   'eligible_dirty_rows':int(eligible.sum()),'selected_rows':chosen,'corrected_cell_counts_by_column':diff.sum().to_dict(),
   'sample_strata_counts':[sum(int(diff.loc[i].sum())==1 for i in chosen),sum(int(diff.loc[i].sum())==2 for i in chosen),sum(int(diff.loc[i].sum())>=3 for i in chosen)]})
 write(OUT/'inputs_real_cells.jsonl',inputs);write(OUT/'ground_truth_real_cells.jsonl',gold)
 with (OUT/'human_annotation_blank.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
 proto={'protocol_id':'public_corrected_cells_v1','frozen_at_utc':datetime.now(timezone.utc).isoformat(),'seed':930761,'n_cases':72,
 'sources':json.loads((DATA/'provenance.json').read_text()),'datasets':dataset_info,
 'selection':'24 dirty targets/dataset, up to8 with1/2/3+ corrected fields; shortages filled randomly from remaining dirty rows, before any predictions.',
 'references':'32 sampled corrected rows excluding all24 heldout target rows. Flights additionally excludes every record of the same flight as the query. Corrected version of target is hidden from every method.',
 'methods':'Three frozen checkpoints; a fixed field-inspection minimal JSON prompt, greedy512, strict parsing. Gower kNN/LOF singleton novelty k3 on same references; categorical mismatch on raw CSV strings. No learned data-cleaning baseline claim.',
 'contract':'Up to4 distinct singleton field candidates. Multi-field candidate output is invalid. No posthoc conversion of explanations into a better field ranking.',
 'metrics':'Equal dataset mean of FieldHit@1/FieldHit@3, precision among first3 returned fields, corrected-field recall within first3, number of inspected fields until first recorded correction (failure assigned D+1), answered coverage. These are offline list metrics, not observed human time savings.',
 'multiple_answers':'Each publicly corrected field is a valid individual inspection target; not an enumeration of alternative minimal support explanations.',
 'human_evidence':'Blank two-rater forms provided; no completed human ratings, disagreement statistic, or measured usability benefit claimed.',
 'limits':'Public corrected versions may include canonicalization and imperfect labels; 3 datasets and72 stratified rows, no claim to representative production accuracy.',
 'sha256':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['inputs_real_cells.jsonl','ground_truth_real_cells.jsonl']},'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (OUT/'real_cells_protocol.json').write_text(json.dumps(proto,indent=2,ensure_ascii=False)+'\n')
 print('frozen',len(inputs),'real corrected-cell cases')
if __name__=='__main__':main()
