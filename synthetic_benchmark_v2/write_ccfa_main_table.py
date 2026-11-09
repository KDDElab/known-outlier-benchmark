#!/usr/bin/env python3
"""Replace the manuscript main table from completed, repaired measurements."""
import argparse
import json
import pandas as pd
from prepare_ccfa_revision import OUT,HERE,sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-primary',action='store_true',help='Use the hash-locked, complete primary cohort while auxiliary runs continue')
    args=parser.parse_args()
    if args.completed_primary:
        snapshot=json.loads((OUT/'primary_complete_snapshot.json').read_text())
        path=OUT/'main_table_partial.csv'
        assert sha(path)==snapshot['files']['main_table.csv']
    else:
        assert json.loads((OUT/'analysis_manifest.json').read_text())['complete']
        path=OUT/'main_table.csv'
    data=pd.read_csv(path).set_index('method')
    groups=[('LMs',[('qwen35_9b','Qwen3.5-9B'),('deepseek_v2_lite_chat','DeepSeek-V2-Lite-Chat'),('llama31_8b_instruct','Llama-3.1-8B-Instruct')]),
            ('Rankers',[('gower_lof_subspace','Gower-LOF'),('gower_knn_subspace','Gower-kNN'),('rlax_reference_adaptation','RLAX adaptation')]),
            ('Adapt.',[('sod_subspace_adaptation','SOD adaptation'),('cop_error_vector_adaptation','COP adaptation')]),
            ('Controls',[('top_univariate','Univariate top four'),('random_size_prior','Random size prior'),('all_features','All fields'),('first_three_display','First three positions')])]
    metrics=['support_top1_sufficient','support_top1_minimal','support_any_minimal_hit','support_best_jaccard',
             'rule_top1_minimal','rule_any_minimal_hit','top1_fraction','answered']
    noncontrols=[method for group,rows in groups if group!='Controls' for method,_ in rows]
    best=data.loc[noncontrols,metrics[:6]].max()
    lines=[]
    for group,rows in groups:
        for i,(method,label) in enumerate(rows):
            values=[]
            for metric in metrics:
                v=float(data.loc[method,metric]);text=f'{v:.3f}'
                if group!='Controls' and metric in best and abs(v-best[metric])<1e-12:text=r'\textbf{'+text+'}'
                values.append(text)
            name=(r'\multirow{'+str(len(rows))+r'}{*}{\rotatebox{90}{\scriptsize '+group+'}}') if i==0 else ''
            lines.append(name+' & '+label+' & '+' & '.join(values)+r' \\')
        lines.append(r'\hline')
    table=r'''\begin{table*}[t]
\centering\footnotesize
\caption{Main comparison: 864 cases, identical inputs, one frozen random display per case. Point estimates use the paired tests' template estimator; failures remain misses. Figure~\ref{fig:pareto}(d) gives 95\% template-bootstrap intervals for the three models and two Gower rankers, conditional on these displays. Table~\ref{tab:random-orders} reports four-order sensitivity. Size is conditional on answering; bold marks the best non-control localization score. M\textsuperscript{2}OE is evaluated separately.}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.22}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}|c|l|rrrr|rr|rr|}
\hline
\multicolumn{2}{|c|}{Methods} & \multicolumn{4}{c|}{Population-support labels ($\uparrow$)} & \multicolumn{2}{c|}{Rule labels ($\uparrow$)} & \multicolumn{2}{c|}{Output behavior} \\
\cline{1-10}
Group & Method & Suff.@1 & Exact@1 & Hit@4 & Jaccard@4 & Exact@1 & Hit@4 & \shortstack{Top-one\\fraction} & Answered \\
\hline
'''+ '\n'.join(lines)+r'''
\end{tabular*}
\renewcommand{\arraystretch}{1.0}
\label{tab:main}
\end{table*}'''
    paper=HERE.parent/'phase_d_manuscript_draft.tex';source=paper.read_text()
    label=source.index(r'\label{tab:main}');start=source.rfind(r'\begin{table*}',0,label);end=source.index(r'\end{table*}',label)+len(r'\end{table*}')
    paper.write_text(source[:start]+table+source[end:])
    (OUT/'main_table_rendered.tex').write_text(table+'\n')
    print('Main table replaced from completed actual measurements.')

if __name__=='__main__':main()
