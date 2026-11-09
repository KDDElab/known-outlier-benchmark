"""Export manuscript tables from complete condition cohorts, never missing rows."""
from pathlib import Path
import json,argparse
from decimal import Decimal,ROUND_HALF_UP
import pandas as pd
HERE=Path(__file__).resolve().parent;OUT=HERE/'runs/ccfa_acceptance_revision'
MODELS=['qwen35_9b','deepseek_v2_lite_chat','llama31_8b_instruct'];RANKERS=['gower_lof_subspace','gower_knn_subspace']
NAMES=dict(zip(MODELS+RANKERS,['Qwen','DeepSeek','Llama','Gower-LOF','Gower-kNN']))
NAMES['uniform_random_field']='Uniform (expected)'
def row(*x):return ' & '.join(x)+r' \\'+'\n'
def fmt(x):return str(Decimal(str(round(float(x),12))).quantize(Decimal('0.001'),rounding=ROUND_HALF_UP))
def table(name,caption,label,body,cols,wide=False):
 env='table*' if wide else 'table';width=r'\textwidth' if wide else r'\columnwidth'
 text='\\begin{'+env+'}[t]\n\\centering\\scriptsize\n\\caption{'+caption+'}\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.16}\n'
 text+='\\begin{tabular*}{'+width+r'}{@{\extracolsep{\fill}}|'+cols+'|}\n\\hline\n'+body+'\\hline\n\\end{tabular*}\n\\renewcommand{\\arraystretch}{1.0}\n\\label{'+label+'}\n\\end{'+env+'}\n'
 (OUT/name).write_text(text)
def export_prior(s):
 pri={(r['method'],r['condition']):r for r in s['prior']}
 body=row('Method',r'\multicolumn{3}{c|}{Original prompt}',r'\multicolumn{3}{c|}{Shared size prior}')+r'\cline{2-7}'+'\n'+row('','Exact','Hit','Ans.','Exact','Hit','Ans.')+r'\hline'+'\n'
 for m in MODELS:
  l=pri[(m,'original_same_cap')];r=pri[(m,'shared_prior3')]
  body+=row(NAMES[m],*[fmt(x[k]) for x in [l,r] for k in ['exact','hit4','answered']])
 body+=r'\hline'+'\n'
 for m in RANKERS:
  r=pri[(m,'gower_cap3')];body+=row(NAMES[m],'--','--','--',*[fmt(r[k]) for k in ['exact','hit4','answered']])
 table('prior_table.tex',r'Paired size prior: 96 targets, 24 templates. Both LLM conditions use the same one-to-three-field rule; no truncation. Gower searches that space. Hit is Hit@4; Ans. is answer coverage.','tab:prior',body,'l|rrr|rrr')

def export_structured(s):
 lookup={(r['method'],r['condition']):r for r in s['structured']}
 body=row('Model',r'\multicolumn{3}{c|}{Unconstrained}',r'\multicolumn{3}{c|}{JSON schema}')+r'\cline{2-7}'+'\n'+row('','Valid','Exact','Hit','Valid','Exact','Hit')+r'\hline'+'\n'
 for m in MODELS:
  a=lookup[(m,'minimal_unconstrained')];b=lookup[(m,'minimal_structured')];body+=row(NAMES[m],*[fmt(r[k]) for r in [a,b] for k in ['valid','exact','hit4']])
 table('structured_table.tex','Fixed JSON decoding with the same minimal prompt and 512-token budget; all 96 queries, no three-field prior. Valid is parser validity; Hit is Hit@4.','tab:structured',body,'l|rrr|rrr')

def export_new_rules(s):
 ns={(r['method'],r['condition']):r for r in s['new_rules']};rp={(r['method'],r['condition']):r for r in s['repeated_reference']};st={r['method']:r for r in s['reference_stability']}
 body=row('Method',r'\multicolumn{2}{c|}{New targets (48)}',r'\multicolumn{3}{c|}{Repeated draws (same 24)}',r'\multicolumn{2}{c|}{Complete population (24)}','Stable')+r'\cline{2-8}'+'\n'+row('','Exact','Hit@4','Draw 0 Exact','Draw 1 Exact','Draw 2 Exact','Exact','Hit@4','/ 24')+r'\hline'+'\n'
 for m in MODELS+RANKERS:
  n=ns[(m,'draw0')];c=rp[(m,'complete')]
  body+=row(NAMES[m],fmt(n['exact']),fmt(n['hit4']),*[fmt(rp[(m,d)]['exact']) for d in ['draw0','draw1','draw2']],fmt(c['exact']),fmt(c['hit4']),str(st[m]['all_three_available_same']))
 table('new_rules_table.tex','New-rule validation, averaged equally over 12 compositions. Queries have six fields; Gower searches all 63 subsets. Draws 0--2 independently sample 32 normal rows. Complete provides every normal tuple and declares completeness. Stable counts targets with an available, identical first set under all three draws.','tab:new-rules',body,'l|rr|rrr|rr|r',True)

def export_real_cells(s):
 ds=pd.read_csv(OUT/(s.get('_table_csv_prefix','')+'real_cells_by_dataset.csv')).set_index(['dataset','method']);rs={r['method']:r for r in s['real_cells']}
 body=row('Method',r'\multicolumn{3}{c|}{FieldHit@1 by source (24 each)}',r'\multicolumn{4}{c|}{Equal-source mean (72)}')+r'\cline{2-8}'+'\n'+row('','Flights','Beers','Rayyan','FieldHit@1','FieldHit@3','Recall@3','Inspections')+r'\hline'+'\n'
 for m in [x for base in MODELS for x in [base,base+':schema']]+RANKERS+['uniform_random_field']:
  r=rs[m];base=m.split(':')[0];label=NAMES[base]+(' (schema)' if ':schema' in m else '')
  body+=row(label,*[fmt(float(ds.loc[(d,m),'field_hit1'])) for d in ['flights','beers','rayyan']],*[fmt(r[k]) for k in ['field_hit1','field_hit3','recall_at3','first_correction_inspections']])
 table('real_cells_table.tex','Public corrected-field inspection, including canonicalization. Schema uses identical prompts with fixed singleton JSON decoding. Recall@3 covers recorded corrections. Inspections is the first corrected field\'s position among up to four suggestions; failure receives $d+1$ (6/9/10 fields by source). Invalid and abstained lists score zero on retrieval. Uniform is an analytic expectation.','tab:real-cells',body,'l|rrr|rrrr',True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--prior-only',action='store_true');p.add_argument('--section',choices=['all','prior','structured','new_rules','real_cells'],default='all');a=p.parse_args()
 section='prior' if a.prior_only else a.section
 s=json.loads((OUT/('summary.json' if section=='all' else 'partial_summary.json')).read_text())
 s['_table_csv_prefix']='' if section=='all' else 'partial_'
 required={'prior':['prior3'],'structured':['structured'],'new_rules':['new_rules'],'real_cells':['real_cells','real_cells_structured']}
 selected=list(required) if section=='all' else [section]
 if section=='all':assert s['complete']
 for key in selected:
  assert all(s['readiness'][f'{batch}/{m}']['complete'] for batch in required[key] for m in MODELS),key
  globals()['export_'+key](s)
 print('exported '+', '.join(selected))
if __name__=='__main__':main()
