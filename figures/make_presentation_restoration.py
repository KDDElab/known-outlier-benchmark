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
from matplotlib.patches import Patch,Rectangle,Polygon,Ellipse,Circle
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
    """Single-column, icon-led invoice story; no additional quantitative charts.

    All values come from constraint_tax_total_000 and its finite population.
    Receipt/edit/alert, database/search, and linked-field/check icons explain
    the three operations. Native vector geometry follows the Figure 2 family;
    icons are authoring primitives, without external assets or fictional data.
    Population and edit-history views are explanatory, evaluator-only evidence.
    """
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
    def receipt(x,values,title,flag=False):
        y=2.48;w=1.12;h=.68;fold=.105
        ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+h-fold),(x+w-fold,y+h),(x,y+h)],
            fc='#FFFCF6' if flag else 'white',ec=border if flag else native.GRID,lw=.8,zorder=3))
        rule(ax,x+w-fold,y+h,x+w-fold,y+h-fold,border if flag else native.GRID,.6)
        rule(ax,x+w-fold,y+h-fold,x+w,y+h-fold,border if flag else native.GRID,.6)
        txt(ax,x+.10,3.06,title,8.2,weight='bold')
        rule(ax,x+.08,2.975,x+w-.08,2.975,native.GRID,.5)
        for i,(label,value,color) in enumerate(zip(['Net','Tax','Gross'],values,[blue,teal,orange])):
            yy=2.885-i*.145
            if flag and i==2:box(ax,x+.065,yy-.064,w-.13,.13,'#FBE8C9','none',0,.015,z=4)
            txt(ax,x+.10,yy,label,8.2,color)
            txt(ax,x+w-.14,yy,str(value),9.3,color,ha='right',weight='bold' if flag else 'normal')
        if flag:
            xx=x+w-.025;yy=3.12
            ax.add_patch(Polygon([(xx-.075,yy-.055),(xx+.075,yy-.055),(xx,yy+.075)],fc='#FFF2CC',ec=orange,lw=.7,zorder=7))
            txt(ax,xx,yy-.005,'!',7.2,orange,ha='center',weight='bold')
    def pencil(x,y):
        ax.add_patch(Polygon([(x-.09,y-.065),(x-.055,y-.10),(x+.10,y+.055),(x+.065,y+.09)],
            fc='#FBE8C9',ec=orange,lw=.8,zorder=7))
        ax.add_patch(Polygon([(x-.09,y-.065),(x-.055,y-.10),(x-.105,y-.115)],fc=orange,ec=orange,lw=.5,zorder=7))
        rule(ax,x+.04,y+.065,x+.075,y+.03,orange,.6)
    def calculator(x,y):
        box(ax,x-.075,y-.08,.15,.16,'white',native.GRAY,.7,.018,z=7)
        box(ax,x-.05,y+.020,.10,.033,native.BLUE_BG,native.GRAY,.5,.004,z=8)
        for dx in [-.033,.033]:
            for dy in [-.016,-.050]:ax.add_patch(Rectangle((x+dx-.012,y+dy-.009),.024,.018,fc=native.GRAY,ec='none',zorder=8))
    def colored_text(cx,y,parts,size,weight='normal'):
        from matplotlib.font_manager import FontProperties
        font=FontProperties(family='Liberation Serif',size=size,weight=weight)
        widths=[fig.canvas.get_renderer().get_text_width_height_descent(text,font,False)[0]/fig.dpi for text,_ in parts]
        x=cx-sum(widths)/2
        for (text,color),width in zip(parts,widths):
            txt(ax,x,y,text,size,color,weight=weight);x+=width
    def chain(x,y,color):
        for dx,dy in [(-.035,-.025),(.035,.025)]:
            ax.add_patch(Ellipse((x+dx,y+dy),.145,.077,angle=38,fill=False,ec=color,lw=.95,zorder=8))
    def shield(x,y):
        ax.add_patch(Polygon([(x-.071,y+.067),(x+.071,y+.067),(x+.065,y-.018),(x,y-.083),(x-.065,y-.018)],
            fc=native.TEAL_BG,ec=teal,lw=.7,zorder=7))
        rule(ax,x-.033,y,x-.008,y-.025,teal,.9);rule(ax,x-.008,y-.025,x+.036,y+.027,teal,.9)
    panel(2.26,1.20,'1. Follow one flagged invoice')
    receipt(.14,clean,'Before edit');receipt(2.23,target,'Flagged row',True)
    pencil(1.745,3.04)
    txt(ax,1.745,2.845,'Edit gross',8.1,orange,ha='center')
    arrow(ax,1.39,2.73,2.09,2.73,orange,.95)
    txt(ax,1.745,2.58,'26 → 14',8.6,orange,ha='center')
    calculator(.31,2.37)
    txt(ax,.48,2.37,'25 + 1 ≠ 14: the total is inconsistent',8.0)
    arrow(ax,1.745,2.245,1.745,2.175,native.GRAY,.85)
    panel(1.17,.99,'2. Search the clean population')
    native.database(ax,.26,1.80,.20,native.PURPLE)
    txt(ax,.47,1.80,'20 clean triples: net + tax = gross',8.1,native.PURPLE)
    for x,title,values in [(.14,'Net / tax match',clean),(1.81,'Gross match',tuple(witness))]:
        box(ax,x,1.335,1.54,.365,native.BLUE_BG,native.GRID,.6,.04)
        native.search(ax,x+.16,1.51,.22,native.BLUE)
        txt(ax,x+.88,1.60,title,7.9,native.GRAY,ha='center')
        parts=[('(',native.GRAY)]
        for j,(v,c) in enumerate(zip(values,[blue,teal,orange])):
            if j:parts.append((', ',native.GRAY))
            parts.append((str(v),c if v==target[j] else native.GRAY))
        parts.append((')',native.GRAY));colored_text(x+.88,1.43,parts,9.0)
    txt(ax,1.745,1.25,'Every single target value has a normal match.',7.7,ha='center')
    arrow(ax,1.745,1.155,1.745,1.085,native.GRAY,.85)
    panel(.035,1.035,'3. Keep each minimal explanation')
    for x,label,subset,color in [(.14,'{net, gross}',[0,2],blue),(1.81,'{tax, gross}',[1,2],teal)]:
        assert frozenset(subset) in minima
        assert not any(all(r[j]==target[j] for j in subset) for r in population)
        assert all(any(r[j]==target[j] for r in population) for j in subset)
        box(ax,x,.38,1.54,.395,'#F0F6EE','#ACBFA7',.6,.04)
        chain(x+.16,.58,color);shield(x+1.39,.58)
        colored_text(x+.79,.65,[('{',native.GRAY),('net' if subset[0]==0 else 'tax',color),
            (', ',native.GRAY),('gross',orange),('}',native.GRAY)],8.4,'bold')
        colored_text(x+.79,.475,[('(',native.GRAY),(str(target[subset[0]]),color),
            (', ',native.GRAY),(str(target[2]),orange),(')',native.GRAY)],9.0)
    txt(ax,1.745,.285,'No joint match; either field alone has support.',7.6,ha='center')
    native.doc(ax,.18,.075,.12,.15,native.GRAY)
    native.doc(ax,.24,.095,.12,.15,native.GRAY)
    txt(ax,.48,.15,'All three fields: redundant',8.1,native.GRAY)
    native.status(ax,3.20,.15,False)
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
    introduction()
    if '--intro-only' not in sys.argv:
        plot_style();evidence();contract();lof();corrections();tables()
    sources=[RUN/name for name in ['summary.csv','repeated_summary.csv','complete_error_counts.csv','per_case.csv','lof_sensitivity_summary.csv','correction_strata_per_case.csv','beers_non_ounces_per_case.csv']]
    (RUN/'presentation_restoration_manifest.json').write_text(json.dumps({'scope':'Presentation only; unchanged frozen experiments',
        'protected_figures':['figure2_overall_framework','figure_unified_transfer'],
        'style_reference':'Previous manuscript and 2511.16455v1.pdf: boxed axes, method colors, hatches, grouped wide tables',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print('Rendered Figure 1 only; other exports untouched.' if '--intro-only' in sys.argv else
          'Rendered figures 1 and 4--7 and three wide tables; figures 2/3 untouched.')
if __name__=='__main__':main()
