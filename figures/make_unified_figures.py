"""Data-bound revision figures; retain the established ICDE plot style."""
from pathlib import Path
import argparse,hashlib,itertools,json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,Patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;SYN=ROOT/'synthetic_benchmark_v2'
RUN=SYN/'runs/ccfa_unified_revision';ACC=SYN/'runs/ccfa_acceptance_revision'
sys.path.insert(0,str(SYN))
from present_acceptance_results import fmt
METHODS=['qwen35_9b','deepseek_v2_lite_chat','llama31_8b_instruct','sample_minimal','gower_knn','gower_lof']
LABELS=['Qwen','DeepSeek','Llama','SME','Gower-kNN','Gower-LOF']
COLORS=['#38A732','#80BEB5','#669DC8','#D39440','#08A9BF','#236EAC']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.labelsize':7.5,
 'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.5,
 'pdf.fonttype':42,'svg.fonttype':'none','hatch.linewidth':.45,'legend.fontsize':7,
 'legend.framealpha':1,'legend.fancybox':False})

def style(ax,axis='x'):
    ax.grid(axis=axis,color='#E3E3E3',linewidth=.5);ax.set_axisbelow(True)
    ax.tick_params(length=2.5,width=.5,pad=2)
    for s in ax.spines.values():s.set_color('#777');s.set_linewidth(.5)

def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(HERE/f'{name}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)

def introduction():
    # Enumerate the exact population from the recorded motivating invoice.
    from make_manuscript_figures import invoice_example
    gold,pop,_,_,ids,witness=invoice_example();target=gold['target_row'];clean=pop['clean_role_rows']
    fig=plt.figure(figsize=(3.49,2.85));ax=fig.add_axes([.13,.30,.84,.49])
    net=[r[0] for r in clean];gross=[r[2] for r in clean]
    ax.scatter(net,gross,s=19,color='#568DAF',edgecolor='white',linewidth=.3,zorder=3,label='Normal population')
    ax.scatter([target[ids[0]]],[target[ids[2]]],marker='x',s=50,color='#B96132',linewidth=1.4,zorder=5)
    ax.scatter([10],[14],s=48,facecolor='none',edgecolor='#5C8B50',linewidth=1.2,zorder=4)
    ax.annotate('Target (25, 14)',(25,14),xytext=(16,11),fontsize=7,color='#9F502B',
                arrowprops={'arrowstyle':'->','lw':.7,'color':'#9F502B'})
    ax.annotate('Gross 14 has a normal witness',(10,14),xytext=(10.5,27.8),fontsize=6.8,color='#4D7844',
                arrowprops={'arrowstyle':'->','lw':.65,'color':'#4D7844'})
    ax.set(xlim=(8.5,26.5),ylim=(9,30.5),xticks=[10,15,20,25],yticks=[10,14,20,25,30],xlabel='Net',ylabel='Gross')
    style(ax,'both')
    top=fig.add_axes([.04,.83,.93,.14]);top.axis('off')
    top.add_patch(Rectangle((0,0),1,1,facecolor='#FFF5E7',edgecolor='#CFB184',lw=.6))
    top.text(.02,.72,'One flagged invoice, different evaluation targets',fontsize=8,weight='bold',va='center')
    top.text(.5,.25,'Net = 25     Tax = 1     Gross = 14',fontsize=8,ha='center',va='center')
    bottom=fig.add_axes([.04,.025,.93,.14]);bottom.axis('off')
    bottom.add_patch(Rectangle((0,0),1,1,facecolor='#F2F6FA',edgecolor='#A8BCCF',lw=.6))
    bottom.text(.025,.70,'Recorded edit',fontsize=7,color='#555');bottom.text(.32,.70,'{gross}',fontsize=7.5)
    bottom.text(.025,.27,'Minimal support',fontsize=7,color='#555');bottom.text(.32,.27,'{net, gross} or {tax, gross}',fontsize=7.5)
    save(fig,'figure_intro_support_semantics')

def transfer(summary):
    fig,axes=plt.subplots(1,3,figsize=(7.16,2.55),sharey=True)
    fig.subplots_adjust(left=.12,right=.98,top=.80,bottom=.18,wspace=.22)
    for ax,track,condition,title in zip(axes,['new_rules','invoices','car_decision'],['draw0','sample','sample'],
        ['(a) New rules: 48 targets','(b) Invoices: 96 targets','(c) Car decisions: 48 targets']):
        data=summary[(summary.track==track)&(summary.condition==condition)].set_index('method').loc[METHODS]
        for i,(m,color) in enumerate(zip(METHODS,COLORS)):
            x=float(data.loc[m,'exact']);h=float(data.loc[m,'hit4']);y=5-i
            ax.plot([x,h],[y,y],color=color,lw=1.2)
            ax.scatter([x],[y],color=color,marker='o',s=23,zorder=4)
            ax.scatter([h],[y],facecolor='white',edgecolor=color,marker='D',s=21,zorder=4)
            ax.text(x,y+.17,fmt(x),fontsize=6.6,ha='center',color='#333')
        ax.set(xlim=(-.05,1.06),ylim=(-.45,5.55),xticks=[0,.25,.5,.75,1],xlabel='Recovery',yticks=range(6),yticklabels=LABELS[::-1])
        ax.set_title(title,pad=8);style(ax)
    fig.legend(handles=[Line2D([],[],marker='o',color='#555',lw=0,label='Exact@1'),Line2D([],[],marker='D',mfc='white',color='#555',lw=0,label='Hit@4')],
        loc='upper center',ncol=2,bbox_to_anchor=(.56,1.01),frameon=False)
    save(fig,'figure_unified_transfer')

def contract(summary):
    prior=pd.read_csv(ACC/'prior_per_case.csv');struct=pd.read_csv(ACC/'structured_per_case.csv')
    fig,axes=plt.subplots(2,1,figsize=(3.49,2.8),sharex=True)
    fig.subplots_adjust(left=.16,right=.97,bottom=.19,top=.84,hspace=.30)
    for ax,metric in zip(axes,['exact','hit4']):
        for m,color,label in zip(METHODS[:3],COLORS[:3],LABELS[:3]):
            values=[]
            for condition in ['original_same_cap','shared_prior3']:
                d=prior[(prior.method==m)&(prior.condition==condition)];values.append(d.groupby('template')[metric].mean().mean())
            d=struct[(struct.method==m)&(struct.condition=='minimal_structured')];values.append(d.groupby('template')[metric].mean().mean())
            values.append(summary[(summary.track=='paired96')&(summary.method==m)][metric].iloc[0])
            ax.plot(range(4),values,marker='o',ms=3,lw=.9,color=color,label=label)
        ax.set(ylim=(-.025,.8),yticks=[0,.2,.4,.6,.8],ylabel='Exact@1' if metric=='exact' else 'Hit@4');style(ax,'y')
    axes[-1].set(xticks=range(4),xticklabels=['Full /\ncap','Full +\nprior','Minimal /\nJSON','Joint'])
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
    save(fig,'figure_unified_contract')

def evidence():
    repeated=pd.read_csv(RUN/'repeated_summary.csv');errors=pd.read_csv(RUN/'complete_error_counts.csv').set_index('method')
    conditions=['draw0','draw1','draw2','complete']
    a=np.array([[repeated[(repeated.method==m)&(repeated.condition==c)].exact.iloc[0] for c in conditions] for m in METHODS])
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.65),gridspec_kw={'width_ratios':[1,1.1]})
    fig.subplots_adjust(left=.12,right=.98,bottom=.19,top=.79,wspace=.40)
    ax=axes[0];ax.imshow(a,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    for i in range(6):
        for j in range(4):ax.text(j,i,fmt(a[i,j]),ha='center',va='center',fontsize=7,color='white' if a[i,j]>.65 else '#222')
    ax.set(xticks=range(4),xticklabels=['Draw 1','Draw 2','Draw 3','Complete'],yticks=range(6),yticklabels=LABELS)
    ax.set_title('(a) Exact@1 on the same 24 targets',pad=8);ax.tick_params(length=0)
    categories=['exact','population_absent_nonminimal','supported_in_visible_references','unavailable','sample_only_absent']
    colors=['#5F9E6E','#D6AF6B','#A9BCCB','#CBCBCB','#B595BE']
    ax=axes[1];left=np.zeros(7);labels=LABELS+['Qwen (original)']
    for cat,color in zip(categories,colors):
        vals=np.array([int(errors.loc[m,cat]) if cat in errors.columns else 0 for m in METHODS]+[{'exact':3,'population_absent_nonminimal':11,'supported_in_visible_references':10}.get(cat,0)])
        ax.barh(range(7),vals,left=left,color=color,height=.64,edgecolor='white',lw=.4)
        for i,v in enumerate(vals):
            if v:ax.text(left[i]+v/2,i,str(v),ha='center',va='center',fontsize=6.8)
        left+=vals
    assert all(left==24)
    ax.set(yticks=range(7),yticklabels=labels,xticks=[0,6,12,18,24],xlim=(0,24),xlabel='Queries');ax.invert_yaxis();style(ax)
    ax.set_title('(b) Complete-population outcomes',pad=8)
    fig.legend([Patch(facecolor=c) for c in colors[:4]],['Exact','Nonminimal','Normal-supported','Unavailable'],loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.55,1.01),fontsize=6.8)
    save(fig,'figure_unified_evidence')

def lof():
    s=pd.read_csv(RUN/'lof_sensitivity_summary.csv').set_index('variant')
    fig,axes=plt.subplots(2,3,figsize=(7.16,3.55));fig.subplots_adjust(left=.08,right=.99,bottom=.13,top=.91,hspace=.70,wspace=.38)
    ax=axes[0,0]
    for mode,color,label in [('original','#236EAC','Fixed k'),('inclusive','#D39440','All boundary ties')]:
        ax.plot([1e-12,1e-9,1e-6],[s.loc[f'{mode}_eps{e:.0e}','score_max'] for e in [1e-12,1e-9,1e-6]],marker='o',ms=3,lw=1,color=color,label=label)
    ax.set(xscale='log',yscale='log',xlabel=r'Reachability floor $\epsilon$',ylabel='Largest winning score',xticks=[1e-12,1e-9,1e-6]);ax.set_title('(a) Score scale',pad=5);style(ax,'both');ax.legend(fontsize=6,frameon=False)
    keys=['original_eps1e-09','ascending_eps1e-09','descending_eps1e-09','inclusive_eps1e-09','distinct_radius_multiplicity','unique_inclusive'];labels=['Fixed k','Tie asc.','Tie desc.','All ties','Distinct','Unique']
    for ax,field,title in [(axes[0,1],'exact','(b) Exact@1'),(axes[0,2],'hit4','(c) Hit@4')]:
        vals=s.loc[keys,field].to_numpy();ax.bar(range(6),vals,color=['#90AFCC']*4+['#438B85','#D5AD70'],edgecolor='#333',lw=.4)
        for i,v in enumerate(vals):ax.text(i,v+.012,fmt(v),ha='center',fontsize=6)
        ax.set(ylim=(0,.52),xticks=range(6),xticklabels=labels,yticks=[0,.25,.5]);ax.tick_params(axis='x',labelrotation=38);ax.set_title(title,pad=5);style(ax,'y')
    ax=axes[1,0];sweep=pd.read_csv(ACC/'invoice_sweep.csv');print('sweep conditions',sweep.condition.unique())
    for method,color,label in [('lof','#236EAC','LOF'),('knn','#08A9BF','kNN')]:
        d=sweep[(sweep.method==method)&(sweep.cap==8)&(sweep.condition=='all_fields')]
        if len(d)==0:d=sweep[(sweep.method==method)&(sweep.cap==8)&(sweep.condition=='all')]
        if len(d)==0:d=sweep[(sweep.method==method)&(sweep.cap==8)&(sweep.condition=='full_fields')]
        assert len(d)==384,(method,len(d),sweep.condition.unique())
        vals=d.groupby('k').exact.mean().reindex([1,3,5,10]);ax.plot([1,3,5,10],vals,marker='o',ms=3,lw=1,color=color,label=label)
    ax.set(xlabel='Neighbors, k',ylabel='Exact@1',xticks=[1,3,5,10],ylim=(-.01,.5));ax.set_title('(d) Neighborhood size',pad=5);style(ax,'both');ax.legend(frameon=False,fontsize=6)
    ax=axes[1,1];npz=np.load(RUN/'lof_candidate_scores/constraint_line_net_000.npz')
    subs=[q for k in range(1,9) for q in itertools.combinations([f'F{i:03d}' for i in range(1,9)],k)]
    selected=[('F002','F003'),('F001','F004','F008')]
    for j,(variant,color) in enumerate([('original_eps1e-09','#90AFCC'),('distinct_radius_multiplicity','#438B85')]):
        vals=[npz[variant][subs.index(x)] for x in selected]
        ax.bar(np.arange(2)+(j-.5)*.34,vals,width=.32,color=color,edgecolor='#333',lw=.4,label=['Fixed k','Distinct'][j])
    ax.set(yscale='log',xticks=[0,1],xticklabels=['{F2,F3}\nSupported','{F1,F4,F8}\nMinimal'],ylabel='Candidate score');ax.set_title('(e) Line-net case 000',pad=5);style(ax,'y');ax.legend(frameon=False,fontsize=6)
    ax=axes[1,2];vals=s.loc[keys,'first_normal_supported'].to_numpy();ax.bar(range(6),vals,color=['#90AFCC']*4+['#438B85','#D5AD70'],edgecolor='#333',lw=.4)
    ax.set(ylim=(0,1),xticks=range(6),xticklabels=labels,yticks=[0,.5,1]);ax.tick_params(axis='x',labelrotation=38);ax.set_title('(f) Normal-supported winners',pad=5);style(ax,'y')
    save(fig,'figure_unified_lof')

def corrections():
    d=pd.read_csv(RUN/'correction_strata_per_case.csv');rest=pd.read_csv(RUN/'beers_non_ounces_per_case.csv')
    methods=METHODS[:3]+['gower_knn_subspace','gower_lof_subspace'];colors=COLORS[:3]+COLORS[-2:];labels=LABELS[:3]+LABELS[-2:]
    fig,axes=plt.subplots(2,1,figsize=(3.49,3.05));fig.subplots_adjust(left=.15,right=.98,bottom=.14,top=.78,hspace=.68)
    ax=axes[0]
    for m,color,label in zip(methods,colors,labels):
        vals=d[d.method==m].groupby('source').hit1.mean().reindex(['flights','beers','rayyan']);ax.plot(range(3),vals,marker='o',ms=3,color=color,lw=.8,label=label)
    ax.set(xticks=range(3),xticklabels=['Flights','Beers','Rayyan'],ylim=(-.03,1.06),ylabel='FieldHit@1');ax.set_title('(a) Published correction targets',pad=5);style(ax,'y')
    ax=axes[1];cohort=set(rest.case_id)
    for i,(m,color) in enumerate(zip(methods,colors)):
        full=d[(d.method==m)&d.case_id.isin(cohort)].hit1.mean();other=rest[rest.method==m].hit1.mean()
        ax.bar(i-.17,full,.31,color=color,edgecolor='#333',lw=.4);ax.bar(i+.17,other,.31,color=color,edgecolor='#333',lw=.4,hatch='xx')
    ax.set(xticks=range(5),xticklabels=['Qwen','DSeek','Llama','kNN','LOF'],ylim=(-.03,1.06),ylabel='FieldHit@1');ax.set_title('(b) Beers: the same 16 queries',pad=5);style(ax,'y')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.54,1.01),fontsize=6.4)
    fig.legend([Patch(facecolor='white',edgecolor='#555'),Patch(facecolor='white',edgecolor='#555',hatch='xx')],['All corrections','Non-ounces only'],loc='lower center',bbox_to_anchor=(.55,.005),ncol=2,frameon=False,fontsize=6.3)
    save(fig,'figure_unified_corrections')

def tables(summary):
    def cell(m,s):return '$'+fmt(m)+r'\!\pm\!'+fmt(s)+'$'
    def table(name,caption,header,rows,label):
        n=len(header);text='\\begin{table}[t]\n\\centering\\footnotesize\n\\caption{'+caption+'}\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.12}\n\\begin{tabular}{l'+'r'*(n-1)+'}\n\\toprule\n'
        text+=' & '.join(header)+r' \\'+'\n\\midrule\n'
        text+=''.join(' & '.join(r)+r' \\'+'\n' for r in rows)
        text+='\\bottomrule\n\\end{tabular}\n\\renewcommand{\\arraystretch}{1.0}\n\\label{'+label+'}\n\\end{table}\n';(RUN/name).write_text(text)
    primary=summary[summary.track=='paired96'].set_index('method')
    rows=[[label]+[cell(primary.loc[m,f],primary.loc[m,f+'_sd']) for f in ['exact','hit4','answered']] for m,label in zip(METHODS,LABELS)]
    table('primary_table.tex','Unified configuration on 96 targets: mean $\\pm$ SD over 24 templates. All methods share the three-field cap and four-candidate budget; LLMs use the minimal task and fixed JSON schema. Failures remain misses.', ['Method','Exact@1','Hit@4','Answered'],rows,'tab:unified')
    stats=pd.read_csv(ACC/'presentation_statistics.csv');rows=[]
    from present_acceptance_results import MAIN_METHODS,MAIN_NAMES
    for m,label in zip(MAIN_METHODS,MAIN_NAMES):
        values=[]
        for metric in ['support_top1_minimal','support_any_minimal_hit','support_top1_sufficient']:
            x=stats[(stats.track=='main')&(stats.method==m)&(stats.metric==metric)].iloc[0];values.append(cell(x['mean'],x['sd']))
        label={'Qwen3.5-9B':'Qwen (original)','DeepSeek-V2-Lite-Chat':'DeepSeek (original)','Llama-3.1-8B-Instruct':'Llama (original)','First three positions':'First three','Random size prior':'Random size'}.get(label,label)
        rows.append([label]+values)
    sm=json.loads((RUN/'original864_sample_minimal.json').read_text());rows.insert(5,['SME']+[cell(sm['mean'][f],sm['sd'][f]) for f in ['exact','hit4','exact']])
    table('original_compact_table.tex','Original 864-case diagnostic: mean $\\pm$ SD across 38 templates. Gower and SME use a three-field prior absent from the original LLM prompt. Rule, size and behavior metrics remain in the artifact; this is not the unified primary comparison.', ['Method','Exact@1','Hit@4','Suff.@1'],rows,'tab:original')

def main():
    p=argparse.ArgumentParser();p.add_argument('--intro-only',action='store_true');a=p.parse_args();introduction()
    if a.intro_only:return
    assert json.loads((RUN/'analysis_manifest.json').read_text())['complete']
    summary=pd.read_csv(RUN/'summary.csv');tables(summary);transfer(summary);contract(summary);evidence();lof();corrections()
    sources=[RUN/n for n in ['summary.csv','per_case.csv','repeated_summary.csv','complete_error_counts.csv','lof_sensitivity_summary.csv','correction_strata_per_case.csv','beers_non_ounces_per_case.csv']]
    (RUN/'presentation_manifest.json').write_text(json.dumps({'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'complete':True},indent=2)+'\n')
if __name__=='__main__':main()
