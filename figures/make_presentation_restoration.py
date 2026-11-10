"""Restore the approved figure/table grammar without changing experiments.

Figure 1 reuses Figure 2's native invoice objects, orange headings, aligned
field cells, arrows and support audit. Figures 4--7 use the earlier 2511-style
boxed axes, sans-serif type, method colors and condition hatches. Figure 2
and Figure 3 exports are protected: this script never writes them.
All numerical panels bind frozen CSV/NPZ records by method/case identity.
"""
from pathlib import Path
import hashlib,itertools,json,re,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch,Rectangle
from matplotlib.ticker import LogFormatterMathtext
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;SYN=ROOT/'synthetic_benchmark_v2'
RUN=SYN/'runs/ccfa_unified_revision';ACC=SYN/'runs/ccfa_acceptance_revision'
sys.path.insert(0,str(SYN))
from present_acceptance_results import fmt
METHODS=['qwen35_9b','deepseek_v2_lite_chat','llama31_8b_instruct','sample_minimal','gower_knn','gower_lof']
LABELS=['Qwen','DeepSeek','Llama','SME','Gower-kNN','Gower-LOF']
SHORT=['Qwen','DSeek','Llama','SME','kNN','LOF']
COLORS=['#38B52A','#A9D7D1','#66A8CD','#D4A35D','#08B8D0','#1477BC']
INK='#202020'

def plot_style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,
      'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'axes.titlesize':8,
      'axes.linewidth':.55,'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'stix',
      'hatch.linewidth':.45,'legend.fontsize':7,'legend.framealpha':1,'legend.fancybox':False})

def style(ax,grid='y'):
    for s in ax.spines.values():s.set_color('#727272');s.set_linewidth(.5)
    ax.tick_params(length=2.5,width=.5,pad=2);ax.set_axisbelow(True)
    ax.grid(axis=grid,color='#E6E6E6',linewidth=.45)

def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(HERE/f'{name}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)

def legend(ax,labels,hatches,anchor=(.5,1.12),ncol=None):
    handles=[Patch(facecolor='#E5E5E5',edgecolor=INK,lw=.5,hatch=h,label=l) for l,h in zip(labels,hatches)]
    ax.legend(handles=handles,loc='lower center',bbox_to_anchor=anchor,ncol=ncol or len(labels),
        fontsize=6.5,borderpad=.3,handlelength=1.25,columnspacing=.85,edgecolor='#B0B0B0')

def introduction():
    import make_manuscript_figures as native
    g,p,_,_,ids,witness=native.invoice_example()
    target=tuple(g['target_row'][f] for f in ids);population=p['clean_role_rows']
    assert target==(25,1,14) and len(population)==20
    clean=next(tuple(r) for r in population if r[:2]==list(target[:2]))
    assert clean==tuple(g['before_row_hidden'][f] for f in ids)
    assert tuple(witness)==(10,4,14)
    minima={frozenset(ids.index(f) for f in s) for s in g['reference_support_sufficient_minimal_subspaces']}
    fig,ax=native.canvas(3.49,3.50)
    txt,box,rule,arrow=native.txt,native.box,native.rule,native.arrow
    header='#FFDDAA';border='#E4BA7D';blue='#4879BF';teal='#4D9381';orange='#AE682F'
    def panel(y,h,title):
        box(ax,.035,y,3.42,h,'white',border,.8,.065)
        box(ax,.035,y+h-.245,3.42,.245,header,header,0,.055)
        txt(ax,.14,y+h-.125,title,9,weight='bold')
    panel(2.42,1.04,'(a) One flagged invoice')
    native.doc(ax,.14,2.98,.17,.20,native.GRAY)
    txt(ax,.41,3.075,'Record',8.4,color=native.GRAY)
    xs=[1.47,2.18,2.94];cs=[blue,teal,orange]
    for x,name,c in zip(xs,['Net','Tax','Gross'],cs):txt(ax,x,3.075,name,8.8,c,ha='center')
    rule(ax,.13,2.965,3.35,2.965,native.GRID,.6)
    txt(ax,.15,2.855,'Before edit',8.4)
    for x,v,c in zip(xs,clean,cs):txt(ax,x,2.855,str(v),9.6,c,ha='center')
    box(ax,.13,2.49,3.22,.245,'#FBE8C9','none',0,.025)
    txt(ax,.15,2.615,'Flagged row',8.6,weight='bold')
    for x,v,c in zip(xs,target,cs):txt(ax,x,2.615,str(v),10,c,ha='center',weight='bold')
    arrow(ax,3.23,2.86,3.23,2.63,orange,.8)
    arrow(ax,1.745,2.405,1.745,2.295,native.GRAY,.85)
    panel(.98,1.30,'(b) Check normal support and minimality')
    txt(ax,1.745,1.935,'P: 20 clean triples satisfying net + tax = gross',7.8,native.GRAY,ha='center')
    for x,s,ha in [(.15,'Candidate','left'),(1.94,'Evidence in P','center'),(3.08,'Minimal?','center')]:txt(ax,x,1.755,s,7.8,weight='bold',ha=ha)
    rule(ax,.13,1.675,3.35,1.675,native.GRID,.55)
    rows=[('{gross}',[2],'Witness (10, 4, 14)'),('{net, gross}',[0,2],'No matching row'),
          ('{tax, gross}',[1,2],'No matching row'),('{net, tax, gross}',[0,1,2],'Absent, redundant')]
    for i,(label,subset,evidence) in enumerate(rows):
        y=1.575-i*.165;is_minimal=frozenset(subset) in minima
        supported=any(all(r[j]==target[j] for j in subset) for r in population)
        assert supported==(i==0)
        if is_minimal:box(ax,.13,y-.073,3.22,.15,'#F0F6EE','none',0,.015)
        txt(ax,.15,y,label,7.8,color=teal if is_minimal else native.INK)
        txt(ax,1.94,y,evidence,7.1,native.GRAY,ha='center')
        native.status(ax,3.08,y,is_minimal)
    arrow(ax,1.745,.962,1.745,.855,native.GRAY,.85)
    panel(.045,.79,'(c) Two evaluation targets')
    box(ax,.13,.125,1.10,.39,'#FFF5E9',border,.5,.035)
    box(ax,1.37,.125,1.98,.39,'#F3EFF8','#B5A3CB',.5,.035)
    txt(ax,.68,.425,'Recorded edit',7.8,native.GRAY,ha='center')
    txt(ax,.68,.245,'{gross}',9,orange,ha='center',weight='bold')
    txt(ax,2.36,.43,'Minimal support explanations',7.7,native.PURPLE,ha='center')
    txt(ax,2.36,.30,'{net, gross}',8.5,blue,ha='center')
    txt(ax,2.36,.19,'or  {tax, gross}',8.5,teal,ha='center')
    save(fig,'figure_intro_support_semantics')

def evidence():
    repeated=pd.read_csv(RUN/'repeated_summary.csv');errors=pd.read_csv(RUN/'complete_error_counts.csv').set_index('method')
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.58));fig.subplots_adjust(left=.066,right=.985,bottom=.24,top=.72,wspace=.24)
    ax=axes[0];x=np.arange(6)
    for j,(cond,hatch) in enumerate(zip(['draw0','draw1','draw2','complete'],['','//','xx','..'])):
        d=repeated[repeated.condition==cond].set_index('method').loc[METHODS]
        ax.bar(x+(j-1.5)*.18,d.exact,width=.165,color=COLORS,edgecolor=INK,lw=.4,hatch=hatch)
    ax.set(xticks=x,xticklabels=SHORT,ylim=(0,1.08),yticks=[0,.5,1],ylabel='Exact@1')
    style(ax);legend(ax,['Draw 1','Draw 2','Draw 3','Complete'],['','//','xx','..'],(.5,1.04),2)
    ax.text(.5,-.37,'(a) Same targets, different normal evidence',transform=ax.transAxes,ha='center',fontsize=8,weight='bold')
    ax=axes[1];bottom=np.zeros(7)
    cats=['exact','population_absent_nonminimal','supported_in_visible_references','unavailable']
    colors=['#38B52A','#E3BC79','#96BCD3','#D7D7D7'];hatches=['','//','xx','..']
    old={'exact':3,'population_absent_nonminimal':11,'supported_in_visible_references':10,'unavailable':0}
    for cat,c,hatch,label in zip(cats,colors,hatches,['Exact','Nonminimal','Supported','Unavailable']):
        vals=np.array([int(errors.loc[m,cat]) if cat in errors else 0 for m in METHODS]+[old[cat]])
        ax.bar(range(7),vals,bottom=bottom,color=c,edgecolor=INK,lw=.4,width=.66,hatch=hatch,label=label)
        for i,v in enumerate(vals):
            if v:ax.text(i,bottom[i]+v/2,str(v),ha='center',va='center',fontsize=6.7,bbox={'facecolor':c,'edgecolor':'none','pad':.15})
        bottom+=vals
    assert all(bottom==24)
    ax.set(xticks=range(7),xticklabels=SHORT+['Qwen\nold'],ylim=(0,25.9),yticks=[0,8,16,24],ylabel='Queries')
    style(ax);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.04),ncol=2,fontsize=6.5,borderpad=.3,columnspacing=.9,edgecolor='#B0B0B0')
    ax.text(.5,-.37,'(b) Complete-population first choices',transform=ax.transAxes,ha='center',fontsize=8,weight='bold')
    save(fig,'figure_unified_evidence')

def contract():
    prior=pd.read_csv(ACC/'prior_per_case.csv');struct=pd.read_csv(ACC/'structured_per_case.csv');joint=pd.read_csv(RUN/'summary.csv')
    fig,axes=plt.subplots(3,1,figsize=(3.49,2.85),sharex=True)
    fig.subplots_adjust(left=.16,right=.98,bottom=.14,top=.81,hspace=.24)
    conditions=['Full / cap','Full + prior','Minimal / JSON','Joint'];hatches=['','//','xx','..']
    for ax,metric,label in zip(axes,['exact','hit4','answered'],['Exact@1','Hit@4','Answered']):
        for i,m in enumerate(METHODS[:3]):
            vals=[]
            for c in ['original_same_cap','shared_prior3']:
                d=prior[(prior.method==m)&(prior.condition==c)];vals.append(d.groupby('template')[metric].mean().mean())
            d=struct[(struct.method==m)&(struct.condition=='minimal_structured')];vals.append(d.groupby('template')[metric].mean().mean())
            vals.append(joint[(joint.track=='paired96')&(joint.method==m)][metric].iloc[0])
            for j,(v,h) in enumerate(zip(vals,hatches)):
                ax.bar(i+(j-1.5)*.18,v,.165,color=COLORS[i],edgecolor=INK,lw=.45,hatch=h)
        ax.set(ylabel=label,ylim=(0,1.08 if metric=='answered' else .64),yticks=[0,.5,1] if metric=='answered' else [0,.3,.6]);style(ax)
    axes[-1].set(xticks=range(3),xticklabels=LABELS[:3]);legend(axes[0],conditions,hatches,(.5,1.11),2)
    save(fig,'figure_unified_contract')

def lof():
    s=pd.read_csv(RUN/'lof_sensitivity_summary.csv').set_index('variant')
    keys=['original_eps1e-09','ascending_eps1e-09','descending_eps1e-09','inclusive_eps1e-09','distinct_radius_multiplicity','unique_inclusive']
    labels=['Fixed\nk','Tie\nasc.','Tie\ndesc.','All\nties','Dist.\nradius','Unique\nrows'];hatches=['','//','\\\\','xx','..','++']
    fig,axes=plt.subplots(2,3,figsize=(7.16,3.05));fig.subplots_adjust(left=.065,right=.985,bottom=.18,top=.92,wspace=.45,hspace=.66)
    def bars(ax,field,title,ylim,ylabel):
        vals=s.loc[keys,field].to_numpy();bs=ax.bar(range(6),vals,.65,color='#1477BC',edgecolor=INK,lw=.45)
        for b,h,v in zip(bs,hatches,vals):
            b.set_hatch(h);ax.text(b.get_x()+b.get_width()/2,v+ylim*.028,fmt(v),ha='center',fontsize=6.2)
        ax.set(xticks=range(6),xticklabels=labels,ylim=(0,ylim),ylabel=ylabel);ax.tick_params(axis='x',labelsize=6.3)
        ax.set_title(title,loc='left',weight='bold',pad=5);style(ax)
    bars(axes[0,0],'exact','(a) Exact recovery',.31,'Exact@1')
    bars(axes[0,1],'hit4','(b) Candidate hit',.57,'Hit@4')
    bars(axes[0,2],'first_normal_supported','(c) Supported winners',1.,'Fraction')
    ax=axes[1,0]
    for mode,offset,h,c,label in [('original',-.18,'','#1477BC','Fixed k'),('inclusive',.18,'xx','#08B8D0','All ties')]:
        vals=[s.loc[f'{mode}_eps{e:.0e}','score_max'] for e in [1e-6,1e-9,1e-12]]
        ax.bar(np.arange(3)+offset,vals,.32,color=c,edgecolor=INK,lw=.45,hatch=h,label=label)
    ax.set(yscale='log',ylim=(1e4,5e13),yticks=[1e4,1e8,1e12],xticks=range(3),xticklabels=[r'$10^{-6}$',r'$10^{-9}$',r'$10^{-12}$'],xlabel=r'Reachability floor $\epsilon$',ylabel='Largest score')
    ax.legend(loc='upper left',fontsize=6,borderpad=.2,handlelength=1.1,labelspacing=.15);style(ax)
    ax.set_title('(d) Numerical scale',loc='left',weight='bold',pad=5)
    ax=axes[1,1];sweep=pd.read_csv(ACC/'invoice_sweep_summary.csv')
    for method,label,c,offset,h in [('knn','Gower-kNN','#08B8D0',-.18,''),('lof','Gower-LOF','#1477BC',.18,'xx')]:
        d=sweep[(sweep.condition=='all')&(sweep.cap==8)&(sweep.method==method)].sort_values('k')
        ax.bar(np.arange(4)+offset,d.exact,.32,color=c,edgecolor=INK,lw=.45,hatch=h,label=label)
    ax.set(xticks=range(4),xticklabels=['1','3','5','10'],ylim=(0,.56),ylabel='Exact@1',xlabel='Neighbors, k');style(ax)
    ax.legend(loc='upper right',fontsize=5.8,borderpad=.2,handlelength=1.1,labelspacing=.1)
    ax.set_title('(e) Neighborhood size',loc='left',weight='bold',pad=5)
    ax=axes[1,2];npz=np.load(RUN/'lof_candidate_scores/constraint_line_net_000.npz')
    subs=[s for k in range(1,9) for s in itertools.combinations([f'F{i:03d}' for i in range(1,9)],k)]
    candidates=[('F002','F003'),('F001','F004','F008')]
    for j,(key,c,h,label) in enumerate([('original_eps1e-09','#C8C8C8','//','Fixed k'),('distinct_radius_multiplicity','#38B52A','','Distinct')]):
        vals=[npz[key][subs.index(s)] for s in candidates]
        ax.barh(np.arange(2)+(j-.5)*.33,np.array(vals)-.1,left=.1,height=.29,color=c,edgecolor=INK,lw=.45,hatch=h,label=label)
        for i,v in enumerate(vals):ax.text(v*1.5,i+(j-.5)*.33,fmt(v) if v<10 else f'{v:.2e}',va='center',fontsize=5.7)
    ax.set(xscale='log',xlim=(.1,1e12),xticks=[1,1e4,1e8],yticks=[0,1],yticklabels=['{F2,F3}\nSupported','{F1,F4,F8}\nMinimal'],xlabel='Candidate LOF score');ax.invert_yaxis();ax.tick_params(axis='y',labelsize=6.1);style(ax,'x')
    ax.legend(loc='lower right',fontsize=6,borderpad=.2,handlelength=1.1);ax.set_title('(f) Line-net case 000',loc='left',weight='bold',pad=5)
    save(fig,'figure_unified_lof')

def corrections():
    d=pd.read_csv(RUN/'correction_strata_per_case.csv');rest=pd.read_csv(RUN/'beers_non_ounces_per_case.csv')
    methods=METHODS[:3]+['gower_knn_subspace','gower_lof_subspace'];colors=COLORS[:3]+COLORS[-2:]
    fig,axes=plt.subplots(2,1,figsize=(3.49,2.8));fig.subplots_adjust(left=.16,right=.98,bottom=.17,top=.83,hspace=.76)
    ax=axes[0]
    for j,(source,hatch) in enumerate(zip(['flights','beers','rayyan'],['','//','xx'])):
        vals=d[d.source==source].groupby('method').hit1.mean().reindex(methods)
        ax.bar(np.arange(5)+(j-1)*.23,vals,.21,color=colors,edgecolor=INK,lw=.45,hatch=hatch)
    ax.set(xticks=range(5),xticklabels=['Qwen','DSeek','Llama','kNN','LOF'],ylim=(0,1.14),yticks=[0,.5,1],ylabel='FieldHit@1');style(ax)
    legend(ax,['Flights','Beers','Rayyan'],['','//','xx'],(.5,1.08),3)
    ax.text(.5,-.38,'(a) Published correction targets',transform=ax.transAxes,ha='center',weight='bold',fontsize=7.8)
    ax=axes[1];cohort=set(rest.case_id)
    for i,(m,c) in enumerate(zip(methods,colors)):
        full=d[(d.method==m)&d.case_id.isin(cohort)].hit1.mean();other=rest[rest.method==m].hit1.mean()
        ax.bar(i-.17,full,.31,color=c,edgecolor=INK,lw=.45);ax.bar(i+.17,other,.31,color=c,edgecolor=INK,lw=.45,hatch='xx')
    ax.set(xticks=range(5),xticklabels=['Qwen','DSeek','Llama','kNN','LOF'],ylim=(0,1.14),yticks=[0,.5,1],ylabel='FieldHit@1');style(ax)
    legend(ax,['All corrections','Non-ounces only'],['','xx'],(.5,1.03),2)
    ax.text(.5,-.40,'(b) Same 16 Beers queries',transform=ax.transAxes,ha='center',weight='bold',fontsize=7.8)
    save(fig,'figure_unified_corrections')

def cell(row,metric):return '$'+fmt(row[metric])+r'\!\pm\!'+fmt(row[metric+'_sd'])+'$'

def table_start(caption,columns):
    return '\\begin{table*}[t]\n\\centering\\scriptsize\n\\caption{'+caption+'}\n\\setlength{\\tabcolsep}{2.5pt}\n\\renewcommand{\\arraystretch}{1.14}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}'+columns+'}\n\\hline\n'
def table_end(label):return '\\hline\n\\end{tabular*}\n\\renewcommand{\\arraystretch}{1.0}\n\\label{'+label+'}\n\\end{table*}\n'
def row(vals):return ' & '.join(vals)+r' \\'+'\n'

def tables():
    summary=pd.read_csv(RUN/'summary.csv');rep=pd.read_csv(RUN/'repeated_summary.csv');pc=pd.read_csv(RUN/'per_case.csv')
    cap='Unified support comparison, mean $\\pm$ SD. Paired targets use 24 template means, new rules 12 rule means, invoices four relation means, and Car 48 cases from one source. Shared size caps and four-candidate budgets apply within each track; all LLMs use fixed JSON decoding. Failures remain misses.'
    tex=table_start(cap,'|l|rr|rr|rr|rr|')
    tex+=row(['Method',r'\multicolumn{2}{c|}{Paired targets (96; cap 3)}',r'\multicolumn{2}{c|}{New rules (48; cap 6)}',r'\multicolumn{2}{c|}{Invoices (96; cap 8)}',r'\multicolumn{2}{c|}{Car decisions (48; cap 6)}'])
    tex+='\\cline{2-9}\n'+row(['']+['Exact@1','Hit@4']*4)+'\\hline\n'
    for i,(m,label) in enumerate(zip(METHODS,LABELS)):
        vals=[label]
        for track,cond in [('paired96','sample'),('new_rules','draw0'),('invoices','sample'),('car_decision','sample')]:
            z=summary[(summary.track==track)&(summary.condition==cond)&(summary.method==m)].iloc[0]
            vals.extend(cell(z,k) for k in ['exact','hit4'])
        tex+=row(vals)
        if i==2:tex+='\\hline\n'
    (RUN/'presentation_primary_wide.tex').write_text(tex+table_end('tab:unified'))
    cap='Unified new-rule evidence diagnostics, mean $\\pm$ SD across 12 rules. The three draws and Complete use the same 24 targets. Complete also declares exhaustive normal evidence. Stable counts targets with an answered, nonempty and identical first set across all three draws.'
    tex=table_start(cap,'|l|rr|rrr|rr|r|')
    tex+=row(['Method',r'\multicolumn{2}{c|}{New targets (48)}',r'\multicolumn{3}{c|}{Repeated references (same 24)}',r'\multicolumn{2}{c|}{Complete population (24)}','Stable'])
    tex+='\\cline{2-8}\n'+row(['','Exact@1','Hit@4','Draw 1 Exact','Draw 2 Exact','Draw 3 Exact','Exact@1','Hit@4','/ 24'])+'\\hline\n'
    target_ids=set(pc[(pc.track=='new_rules')&(pc.evidence_condition=='complete')].target_id)
    for i,(m,label) in enumerate(zip(METHODS,LABELS)):
        a=summary[(summary.track=='new_rules')&(summary.condition=='draw0')&(summary.method==m)].iloc[0]
        vals=[label,cell(a,'exact'),cell(a,'hit4')]
        for cond in ['draw0','draw1','draw2']:
            z=rep[(rep.condition==cond)&(rep.method==m)].iloc[0];vals.append(cell(z,'exact'))
        z=rep[(rep.condition=='complete')&(rep.method==m)].iloc[0];vals.extend(cell(z,k) for k in ['exact','hit4'])
        d=pc[(pc.method==m)&(pc.track=='new_rules')&(pc.evidence_condition!='complete')&pc.target_id.isin(target_ids)]
        stable=sum(len(z)==3 and bool(z.answered.all()) and z['first'].fillna('').nunique()==1 and z['first'].notna().all() for _,z in d.groupby('target_id'))
        vals.append(str(stable));tex+=row(vals)
        if i==2:tex+='\\hline\n'
    (RUN/'presentation_evidence_wide.tex').write_text(tex+table_end('tab:unified-evidence'))
    # Keep every value and method row of the requested original wide table.
    old=(ACC/'presentation_main_table.tex').read_text()
    caption='Original 864-case diagnostic: mean $\\pm$ SD across 38 templates; answered-only size uses coverage-weighted template SD. Gower uses a three-field prior absent from the original LLM prompt. This table preserves the original configuration; Table~\\ref{tab:unified} gives the joint comparison. Bold marks the best non-control mean; M\\textsuperscript{2}OE is assessed separately.'
    old=old[:old.index('\\caption{')]+'\\caption{'+caption+'}\n'+old[old.index('\\setlength{'):]
    old=old.replace('\\label{tab:main}','\\label{tab:original}')
    (RUN/'presentation_original_wide.tex').write_text(old)

def main():
    introduction();plot_style();evidence();contract();lof();corrections();tables()
    sources=[RUN/name for name in ['summary.csv','repeated_summary.csv','complete_error_counts.csv','per_case.csv','lof_sensitivity_summary.csv','correction_strata_per_case.csv','beers_non_ounces_per_case.csv']]
    (RUN/'presentation_restoration_manifest.json').write_text(json.dumps({'scope':'Presentation only; unchanged frozen experiments',
        'protected_figures':['figure2_overall_framework','figure_unified_transfer'],
        'style_reference':'Previous manuscript and 2511.16455v1.pdf: boxed axes, method colors, hatches, grouped wide tables',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print('Rendered figures 1 and 4--7 and three wide tables; figures 2/3 untouched.')
if __name__=='__main__':main()
