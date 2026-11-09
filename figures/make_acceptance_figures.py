"""Frozen-result figures styled after 2511.16455v1.pdf, Figures 7--10.

Bordered axes, fine grids, method colors, condition hatches, aggregate bars
above distribution/diagnostic panels. No reference-paper values are used.
"""
from pathlib import Path
import gzip, hashlib, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterMathtext

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SYN = ROOT / 'synthetic_benchmark_v2'
DATA = SYN / 'runs/ccfa_acceptance_revision'
sys.path.insert(0, str(SYN))
import present_acceptance_results as presentation
METHODS = presentation.METHODS
SHORT = ['Qwen', 'DeepSeek', 'Llama', 'Gower\nLOF', 'Gower\nkNN']
COLORS = ['#38B52A', '#A9D7D1', '#66A8CD', '#1477BC', '#08B8D0']
INK = '#202020'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,
 'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,
 'axes.titlesize':8,'axes.linewidth':.55,'pdf.fonttype':42,
 'svg.fonttype':'none','mathtext.fontset':'stix','hatch.linewidth':.45,
 'legend.fontsize':7,'legend.framealpha':1,'legend.fancybox':False})

def style(ax, grid='y'):
    for spine in ax.spines.values():
        spine.set_color('#727272'); spine.set_linewidth(.5)
    ax.tick_params(length=2.5,width=.5,pad=2,colors=INK)
    ax.set_axisbelow(True); ax.grid(axis=grid,color='#E6E6E6',linewidth=.45)

def save(fig, name):
    for ext in ['pdf','svg','png']:
        fig.savefig(HERE/f'{name}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)

def legend_conditions(fig, labels, hatches):
    handles=[Patch(facecolor='#E5E5E5',edgecolor=INK,linewidth=.5,hatch=h,label=l)
             for l,h in zip(labels,hatches)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.99),
        ncol=len(labels),borderpad=.3,handlelength=1.4,handletextpad=.5,
        columnspacing=1.,edgecolor='#B0B0B0')

def primary_intervals(values):
    summary=pd.read_csv(SYN/'runs/ccfa_revision/main_summary.csv')
    fig,axes=plt.subplots(2,1,figsize=(3.49,2.65),sharex=True)
    fig.subplots_adjust(left=.14,right=.98,bottom=.20,top=.87,hspace=.58)
    x=np.arange(5)
    for metric,offset,hatch in [('support_top1_minimal',-.18,''),
                                ('support_any_minimal_hit',.18,'xx')]:
        v=summary[summary.metric==metric].set_index('method').loc[METHODS]
        axes[0].bar(x+offset,v.estimate,width=.32,color=COLORS,hatch=hatch,
            edgecolor=INK,linewidth=.45,yerr=[v.estimate-v.lo,v.hi-v.estimate],
            capsize=1.8,error_kw={'elinewidth':.65,'capthick':.65})
        samples=[values[(values.track=='main')&(values.method==m)&
            (values.metric==metric)].value.to_numpy() for m in METHODS]
        assert all(len(v)==38 for v in samples)
        boxes=axes[1].boxplot(samples,positions=x+offset,widths=.28,
            patch_artist=True,manage_ticks=False,showfliers=True,
            medianprops={'color':INK,'linewidth':.75},
            whiskerprops={'linewidth':.55},capprops={'linewidth':.55},
            flierprops={'marker':'o','markersize':2,'markerfacecolor':'white',
                        'markeredgecolor':INK,'markeredgewidth':.4})
        for box,color in zip(boxes['boxes'],COLORS):
            box.set(facecolor=color,edgecolor=INK,linewidth=.45,hatch=hatch)
    for ax in axes: style(ax,'both'); ax.set_xlim(-.6,4.6)
    axes[0].set(ylim=(0,.85),yticks=[0,.2,.4,.6,.8],ylabel='Recovery')
    axes[1].set(ylim=(-.025,1.04),yticks=[0,.5,1],ylabel='Template score',
                xticks=x,xticklabels=SHORT)
    axes[0].text(.5,-.16,'(a) Macro means and 95% intervals',
        transform=axes[0].transAxes,ha='center',va='top',fontsize=7.5)
    axes[1].text(.5,-.51,'(b) Distribution across 38 templates',
        transform=axes[1].transAxes,ha='center',va='top',fontsize=7.5)
    legend_conditions(fig,['Exact@1','Hit@4'],['','xx'])
    save(fig,'figure_acceptance_primary_intervals')

def paired_conditions(stats,track,conditions,names,metrics,labels,filename,height):
    methods=METHODS if track=='prior' else METHODS[:3]
    fig,axes=plt.subplots(3,1,figsize=(3.49,height),sharex=True)
    fig.subplots_adjust(left=.16,right=.98,bottom=.14,top=.87,hspace=.26)
    for ax,metric,label in zip(axes,metrics,labels):
        style(ax); maxima=[]
        for j,method in enumerate(methods):
            conds=conditions if j<3 else ['gower_cap3']
            for index,condition in enumerate(conds):
                entry=stats[(stats.track==track)&(stats.method==method)&
                    (stats.condition==condition)&(stats.metric==metric)]
                assert len(entry)==1
                value=float(entry['mean'].iloc[0]); maxima.append(value)
                offset=(-.18 if index==0 else .18) if j<3 else 0
                ax.bar(j+offset,value,width=.32 if j<3 else .40,
                    color=COLORS[j],edgecolor=INK,linewidth=.45,
                    hatch='' if index==0 and j<3 else 'xx')
                if track=='structured':
                    ax.text(j+offset,value+.025,presentation.fmt(value),
                        ha='center',va='bottom',fontsize=6.6)
        ceiling=1.18 if max(maxima)>.8 else (.82 if track=='prior' else .69)
        ax.set(ylim=(0,ceiling),ylabel=label,xlim=(-.6,len(methods)-.4))
        ax.set_yticks([0,.5,1] if ceiling>1 else [0,.3,.6])
    axes[-1].set_xticks(np.arange(len(methods)),SHORT[:len(methods)])
    legend_conditions(fig,names,['','xx'])
    save(fig,filename)

def invoice_diagnostics(stats):
    summary=pd.read_csv(DATA/'invoice_sweep_summary.csv')
    raw=[json.loads(line) for line in gzip.open(DATA/'invoice_candidate_scores.jsonl.gz','rt')]
    calibration=json.loads((DATA/'invoice_lof_mechanism_summary.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(7.16,3.00))
    fig.subplots_adjust(left=.065,right=.985,bottom=.18,top=.93,wspace=.49,hspace=.57)
    for ax in axes.flat:style(ax)
    for ax,metric,label,limit,ylabel in zip(axes[0],
        ['support_top1_minimal','support_any_minimal_hit','support_best_jaccard'],
        ['(a) Exact recovery','(b) Candidate hit','(c) Set overlap'],
        [.57,.69,1.02],['Exact@1','Hit@4','Jaccard@4']):
        d=stats[(stats.track=='invoice')&(stats.metric==metric)].set_index('method').loc[METHODS]
        ax.bar(np.arange(5),d['mean'],color=COLORS,width=.64,edgecolor=INK,linewidth=.45)
        for j,value in enumerate(d['mean']):
            ax.text(j,value+.015,presentation.fmt(value),ha='center',va='bottom',fontsize=6.6)
        ax.set(xticks=np.arange(5),xticklabels=['Qwen','DSeek','Llama','LOF','kNN'],
            ylim=(0,limit),ylabel=ylabel)
        ax.set_title(label,loc='left',pad=5,fontweight='bold')
    ax=axes[1,0]
    for method,label,color,offset,hatch in [('knn','Gower-kNN',COLORS[4],-.18,''),
                                           ('lof','Gower-LOF',COLORS[3],.18,'xx')]:
        d=summary[(summary.condition=='all')&(summary.cap==8)&
                  (summary.method==method)].sort_values('k')
        ax.bar(np.arange(4)+offset,d.exact,width=.32,color=color,hatch=hatch,
               edgecolor=INK,linewidth=.45,label=label)
    ax.set(xticks=np.arange(4),xticklabels=['1','3','5','10'],ylim=(0,.56),
           ylabel='Exact@1',xlabel='Neighbors, k')
    ax.legend(loc='upper right',fontsize=6.3,borderpad=.2,handlelength=1.1,labelspacing=.2)
    ax.set_title('(d) Neighborhood size',loc='left',pad=5,fontweight='bold')
    ax=axes[1,1]
    variants=[('all',8),('all',3),('roles_only_oracle',8),('unique_projection_diagnostic',8)]
    vals=[float(summary[(summary.condition==c)&(summary.cap==cap)&
        (summary.k==3)&(summary.method=='lof')].exact.iloc[0]) for c,cap in variants]
    vals.append(calibration['calibrated_exact'])
    bars=ax.bar(np.arange(5),vals,width=.62,color=COLORS[3],edgecolor=INK,linewidth=.45)
    for bar,hatch in zip(bars,['','xx','///','..','++']):bar.set_hatch(hatch)
    for j,value in enumerate(vals):
        ax.text(j,value+.012,presentation.fmt(value),ha='center',va='bottom',fontsize=6.6)
    ax.set(ylim=(0,.34),yticks=[0,.1,.2,.3],xticks=np.arange(5),
           xticklabels=['All','Cap 3','Roles','Unique','Rank'],ylabel='LOF Exact@1')
    ax.tick_params(axis='x',labelrotation=35)
    ax.set_title('(e) Matched interventions',loc='left',pad=5,fontweight='bold')
    ax=axes[1,2]
    rows=[r for r in raw if r['case_id']=='constraint_line_net_000' and r['condition']=='all']
    rows.sort(key=lambda r:(-r['scores']['lof_k3'],len(r['subset']),r['subset']))
    selected=[rows[0],rows[1],next(r for r in rows if r['gold'])]
    scores=[r['scores']['lof_k3'] for r in selected]
    bars=ax.barh(np.arange(3),np.asarray(scores)-1,left=1,height=.52,
        color=['#D9D9D9','#D9D9D9',COLORS[0]],edgecolor=INK,linewidth=.45)
    bars[0].set_hatch('//');bars[1].set_hatch('//')
    labels=['{'+','.join(f.replace('F00','F') for f in r['subset'])+'}' for r in selected]
    ax.set_yticks(np.arange(3),labels,fontsize=6.6);ax.invert_yaxis()
    ax.set_xscale('log');ax.set_xlim(1,2e11)
    ax.set_xticks([1,1e4,1e8]);ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.grid(axis='y',visible=False);ax.grid(axis='x',color='#E6E6E6',lw=.45)
    for j,value in enumerate(scores):
        text=f'{value:.3f}' if value<10 else f'{value:.2e}'
        ax.text(value*1.5,j,text,va='center',fontsize=6.4)
    ax.set_xlabel('Raw LOF score (log scale)')
    ax.set_title('(f) Line-net case 000',loc='left',pad=5,fontweight='bold')
    save(fig,'figure_acceptance_invoice_mechanism')

def main():
    presentation.main()
    stats=pd.read_csv(DATA/'presentation_statistics.csv')
    values=pd.read_csv(DATA/'presentation_group_values.csv')
    primary_intervals(values)
    paired_conditions(stats,'prior',['original_same_cap','shared_prior3'],
        ['Original prompt','Shared size prior'],['exact','hit4','answered'],
        ['Exact@1','Hit@4','Answered'],'figure_acceptance_shared_prior',2.35)
    paired_conditions(stats,'structured',['minimal_unconstrained','minimal_structured'],
        ['Unconstrained','JSON schema'],['valid','exact','hit4'],
        ['Valid JSON','Exact@1','Hit@4'],'figure_acceptance_structured_output',2.35)
    invoice_diagnostics(stats)
    paths=[DATA/'presentation_statistics.csv',DATA/'presentation_group_values.csv',
        DATA/'invoice_sweep_summary.csv',DATA/'invoice_candidate_scores.jsonl.gz',
        DATA/'invoice_lof_mechanism_summary.json',SYN/'runs/ccfa_revision/main_summary.csv']
    provenance={'style_reference':'2511.16455v1.pdf Figures 7--10: bars, hatches, boxed axes, grids, aggregate/distribution panels',
        'reference_values_used':False,'new_model_inference':False,
        'primary_uncertainty':'Original 2000-replicate, 95% template-bootstrap intervals',
        'primary_distributions':'38 actual template means; median/IQR, 1.5-IQR whiskers, visible outliers',
        'intervention_figures':'Original macro means; no new significance tests',
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (DATA/'presentation_figures_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    # Refresh the existing per-figure receipts as well as the shared receipt.
    for name in ['primary_interval_figure_provenance.json','invoice_figure_provenance.json']:
        (DATA/name).write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print('Rendered four experimental figures from frozen records.')

if __name__=='__main__':main()
