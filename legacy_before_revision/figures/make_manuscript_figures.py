"""Publication-size vector overviews; rerun from any working directory.

Figure 1 is sourced from Pima case C0035, its frozen reference rows, and
recorded Qwen / leave-one-model-out outputs. Figure 2 is a protocol diagram.
The supplied figures 117, 1171, and 1172 inform grouping, arrows and panels.
All glyphs are drawn here as editable vector elements.
"""
from __future__ import annotations
import ast
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INK = '#26333F'
GRAY = '#61707D'
ORANGE = '#DB8A3C'
ORANGE_LIGHT = '#FFDFAC'
WARM = '#FFFBF4'
BLUE = '#3973AD'
BLUE_LIGHT = '#EAF2F9'
GREEN = '#47866D'
GREEN_LIGHT = '#EDF5EF'
PURPLE = '#7961A8'
PURPLE_LIGHT = '#F2EDF8'
RED = '#BF5448'
GRID = '#CBD4DC'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':7.0,
                     'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
                     'axes.linewidth':.6, 'mathtext.fontset':'dejavusans'})


def txt(ax,x,y,s,size=7.0,color=INK,weight='normal',ha='left',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,
                   ha=ha,va=va,linespacing=1.2,zorder=8,**kw)


def box(ax,x,y,w,h,face='white',edge=GRID,lw=.7,r=.055,dash=False):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                    facecolor=face,edgecolor=edge,linewidth=lw,
                    linestyle=(0,(3,2)) if dash else '-',zorder=2)
    ax.add_patch(p)
    return p


def arrow(ax,x1,y1,x2,y2,color=GRAY,lw=.85,dash=False,rad=0):
    p=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=7,
                      connectionstyle=f'arc3,rad={rad}',color=color,linewidth=lw,
                      linestyle=(0,(3,2)) if dash else '-',shrinkA=0,shrinkB=0,zorder=5)
    ax.add_patch(p)


def rule(ax,x1,y1,x2,y2,color=GRID,lw=.6,**kw):
    ax.plot([x1,x2],[y1,y2],color=color,lw=lw,zorder=4,**kw)


def table_icon(ax,x,y,w=.36,h=.31,color=BLUE,target=False):
    ax.add_patch(Rectangle((x,y),w,h,fc='white',ec=color,lw=.7,zorder=3))
    ax.add_patch(Rectangle((x,y+h*.75),w,h*.25,fc=color,ec='none',zorder=5))
    if target: ax.add_patch(Rectangle((x,y),w,h*.25,fc='#F8D2BC',ec='none',zorder=3.5))
    for k in [1,2]: rule(ax,x+w*k/3,y,x+w*k/3,y+h*.75,color,.45)
    for k in [1,2]: rule(ax,x,y+h*k/4,x+w,y+h*k/4,color,.45)


def chip_icon(ax,x,y,color=PURPLE):
    box(ax,x-.10,y-.10,.20,.20,'white',color,.8,.025)
    for d in [-.065,0,.065]:
        rule(ax,x-.14,y+d,x-.10,y+d,color,.7);rule(ax,x+.10,y+d,x+.14,y+d,color,.7)
        rule(ax,x+d,y-.14,x+d,y-.10,color,.7);rule(ax,x+d,y+.10,x+d,y+.14,color,.7)
    for px,py in [(x-.04,y+.025),(x+.04,y+.025),(x,y-.04)]:
        ax.add_patch(Circle((px,py),.014,fc=color,ec='none',zorder=7))
    rule(ax,x-.04,y+.025,x,y-.04,color,.5);rule(ax,x+.04,y+.025,x,y-.04,color,.5)


def search_icon(ax,x,y,color=BLUE):
    ax.add_patch(Circle((x,y),.075,fc='white',ec=color,lw=.9,zorder=6))
    rule(ax,x+.055,y-.055,x+.12,y-.12,color,1.4)
    for k,h in enumerate([.04,.08,.06]):
        ax.add_patch(Rectangle((x-.045+k*.033,y-.035),.019,h,fc=color,ec='none',zorder=7))


def gate_icon(ax,x,y,color=BLUE):
    for dy in [-.06,.06]: rule(ax,x-.16,y+dy,x-.04,y+dy,color,.8)
    box(ax,x-.04,y-.11,.17,.22,'white',color,.8,.035)
    txt(ax,x+.045,y,'&',7,color,'bold',ha='center')
    rule(ax,x+.13,y,x+.23,y,color,.8)
    ax.add_patch(Circle((x+.16,y),.025,fc='white',ec=color,lw=.7,zorder=7))


def panel(ax,x,y,w,h,title):
    box(ax,x,y,w,h,WARM,ORANGE,.85,.075)
    box(ax,x+.035,y+h-.255,w-.07,.22,ORANGE_LIGHT,'none',0,.06)
    txt(ax,x+w/2,y+h-.144,title,8.0,'#633E19','bold',ha='center')


def canvas(w,h):
    f=plt.figure(figsize=(w,h),facecolor='white')
    a=f.add_axes([0,0,1,1]);a.set_xlim(0,w);a.set_ylim(0,h);a.axis('off')
    return f,a


def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(HERE/f'{name}.{ext}',dpi=240,facecolor='white',
                    metadata={'Creator':'make_manuscript_figures.py'} if ext=='pdf' else None)
    plt.close(fig)


def pima_case():
    d=np.load(ROOT/'data/adbench/29_Pima.npz')
    cases=list(csv.DictReader((ROOT/'manifests/cases.csv').open()))
    c=next(c for c in cases if c['case_id']=='C0035')
    refs=json.loads(c['normal_reference_source_row_indices'])
    q=next(json.loads(line) for line in (ROOT/'runs/annotations.jsonl').open()
           if json.loads(line).get('case_id')=='C0035')
    score=next(row for row in csv.DictReader((ROOT/'reports/leave_one_model_out_by_case.csv').open())
               if row['case_id']=='C0035' and row['held_out_annotator_id']=='qwen35_9b')
    return d['X'],d['y'],int(c['source_row_index']),refs,q['parsed_annotation']['feature_ids_ranked'],score


def make_intro_figure():
    fig,ax=canvas(7.16,3.22)
    X,y,target,refs,pred,score=pima_case()
    txt(ax,.08,3.08,'(a) A flagged row in the Pima table',8.2,weight='bold')
    txt(ax,3.30,3.08,'(b) The actual query and model output',8.2,weight='bold')
    # A real two-field projection, matching the axes of the supplied example.
    sc=fig.add_axes([.075,.275,.355,.60])
    valid=(X[:,1]>0)&(X[:,5]>0)
    for label,color,marker in [(0,BLUE,'o'),(1,'#D88B64','^')]:
        m=valid&(y==label)
        sc.scatter(X[m,1],X[m,5],s=7 if label==0 else 8,c=color,marker=marker,
                   alpha=.40,linewidths=0,zorder=2,label=f'ADBench y={label}')
    rr=np.array(refs);rr=rr[valid[rr]]
    sc.scatter(X[rr,1],X[rr,5],s=21,fc='none',ec=INK,lw=.7,zorder=4,label='12 references')
    sc.scatter(X[target,1],X[target,5],marker='*',s=72,c=RED,ec='white',lw=.5,zorder=6)
    sc.annotate('Target 259\n(155, 33.3)',(X[target,1],X[target,5]),xytext=(136,52),
                 fontsize=6.8,color=RED,fontweight='bold',ha='center',
                 arrowprops={'arrowstyle':'-','color':RED,'lw':.8},
                 bbox={'boxstyle':'round,pad=.18','fc':'white','ec':'none','alpha':.9},zorder=7)
    sc.set(xlim=(35,205),ylim=(14,70),xlabel='Glucose (mg/dL)',ylabel='BMI')
    sc.set_xticks([40,80,120,160,200]);sc.set_yticks([20,30,40,50,60,70])
    sc.tick_params(labelsize=6.7,length=2,color=GRAY,pad=2)
    sc.xaxis.label.set_size(7.1);sc.yaxis.label.set_size(7.1)
    sc.grid(color='#E1E6EB',lw=.45);sc.set_axisbelow(True)
    for side in ['top','right']:sc.spines[side].set_visible(False)
    for side in ['left','bottom']:sc.spines[side].set_color('#8796A1')
    sc.legend(loc='upper left',fontsize=5.9,frameon=False,handletextpad=.25,
              labelspacing=.3,borderpad=.05,markerscale=1.15)
    # Concrete model-visible rows; this slice keeps feature IDs opaque.
    box(ax,3.31,1.76,3.75,1.13,BLUE_LIGHT,'#B8CDE0',.7,.065)
    table_icon(ax,3.43,2.64,.24,.16,BLUE,True)
    txt(ax,3.77,2.72,'Case C0035: eight fields + 12 reference rows',7.2,BLUE,'bold')
    cols=[3.44,4.10,4.68,5.26,5.86,6.49]
    heads=['Row','F001','F002','F005','F006','F007']
    for xx,s in zip(cols,heads):txt(ax,xx,2.49,s,6.8,weight='bold')
    rule(ax,3.42,2.40,6.94,2.40,'#A2B9CD',.6)
    shown=[('Target',target),('$R_1$',refs[0]),('$R_2$',refs[1])]
    for j,(name,idx) in enumerate(shown):
        yy=2.29-j*.17
        if j==0:box(ax,3.41,yy-.078,3.54,.156,'#FFE5CC','none',0,.012)
        vals=[name,*[f'{X[idx,k]:g}' for k in [0,1,4,5,6]]]
        for xx,s in zip(cols,vals):txt(ax,xx,yy,s,6.8,RED if j==0 else INK,'bold' if j==0 else 'normal')
    txt(ax,3.44,1.82,'… 10 more reference rows; three other fields omitted here',6.4,GRAY)
    # The displayed output is a frozen response, not an invented explanation.
    box(ax,3.31,.78,3.75,.88,'#F7F5FB','#C9BEDB',.7,.065)
    chip_icon(ax,3.49,1.43,PURPLE)
    txt(ax,3.75,1.45,'Observed feature ranking',7.2,PURPLE,'bold')
    txt(ax,3.45,1.23,'Qwen',7.0,weight='bold')
    txt(ax,4.17,1.23,' > '.join(pred),7.1,PURPLE,weight='bold')
    txt(ax,3.45,1.04,'Peers',7.0,weight='bold')
    peer=ast.literal_eval(score['peer_consensus_features'])
    txt(ax,4.17,1.04,'{'+', '.join(peer)+'}',7.1,GREEN,weight='bold')
    txt(ax,3.45,.86,f"Held-out Jaccard = {float(score['jaccard']):.3f}   ·   feature truth unavailable",6.6,GRAY)
    arrow(ax,5.18,1.75,5.18,1.68,BLUE)
    # Short left-to-right task pipeline as in the Text-to-SQL reference.
    labels=[('Flagged row',BLUE_LIGHT,BLUE),('Schema + references', '#F1EEF6',PURPLE),
            ('Rank feature IDs','#FFF1DC',ORANGE),('Measure peer agreement',GREEN_LIGHT,GREEN)]
    xs=[.11,1.82,3.54,5.27];ws=[1.47,1.49,1.49,1.78]
    for i,((s,fc,c),xx,w) in enumerate(zip(labels,xs,ws)):
        box(ax,xx,.13,w,.31,fc,'none',0,.07);txt(ax,xx+w/2,.285,s,7,c,'bold',ha='center')
        if i<3:arrow(ax,xx+w+.025,.285,xs[i+1]-.035,.285,GRAY,.75)
    save(fig,'figure_intro_real_dataset')


def make_framework_figure():
    fig,ax=canvas(7.16,3.93)
    # Upper construction/comparison panels and lower parallel evaluation paths.
    panel(ax,.04,1.96,2.73,1.93,'1. Construct labeled cases')
    panel(ax,2.91,1.96,4.21,1.93,'2. Compare methods under one query')
    txt(ax,.20,3.49,'40 rules · 3 seeds · disjoint template splits',6.9,GRAY)
    table_icon(ax,.22,2.98,.40,.31,GREEN,True)
    txt(ax,.75,3.22,'Finite clean population',7.1,GREEN,'bold')
    txt(ax,.75,3.05,'Enumerate + inject anomalies',6.7,GRAY)
    # Branch to what the model sees versus what the evaluator knows.
    rule(ax,1.40,2.92,1.40,2.82,GREEN,.8)
    rule(ax,.73,2.82,2.12,2.82,GREEN,.8)
    arrow(ax,.73,2.82,.73,2.71,GREEN)
    arrow(ax,2.12,2.82,2.12,2.71,ORANGE,dash=True)
    box(ax,.18,2.13,1.12,.57,GREEN_LIGHT,'#A3C4B0',.7,.045)
    txt(ax,.74,2.56,'Visible query',7.1,GREEN,'bold',ha='center')
    txt(ax,.74,2.37,r'$T,\;x^*,\;R$',8.5,GREEN,ha='center')
    txt(ax,.74,2.22,'8–32 normal rows',6.5,GRAY,ha='center')
    box(ax,1.45,2.13,1.17,.57,'#FFF2DF','#E0B47D',.7,.045,dash=True)
    txt(ax,2.035,2.56,'Hidden labels',7.1,'#9C642C','bold',ha='center')
    txt(ax,2.035,2.37,'Support · rule',6.8,ha='center')
    txt(ax,2.035,2.22,'Mechanism · edits',6.5,ha='center')
    arrow(ax,2.79,3.05,2.89,3.05,ORANGE,1.0)
    # Fixed query, two method lanes, and a common output contract.
    box(ax,3.06,2.42,.86,1.00,'white','#9EBBD1',.75,.055)
    txt(ax,3.49,3.29,'Same input',7.0,BLUE,'bold',ha='center')
    table_icon(ax,3.28,2.80,.42,.29,BLUE,True)
    txt(ax,3.49,2.65,r'$T,\;x^*,\;R$',8.4,BLUE,ha='center')
    txt(ax,3.49,2.49,'No gold labels',6.4,GRAY,ha='center')
    box(ax,4.16,2.99,1.61,.49,PURPLE_LIGHT,'#C3B4D8',.7,.05)
    chip_icon(ax,4.36,3.24,PURPLE)
    txt(ax,4.58,3.33,'Prompted LMs',7.3,PURPLE,'bold')
    txt(ax,4.58,3.13,'Qwen · DeepSeek · Llama',6.4)
    box(ax,4.16,2.36,1.61,.49,BLUE_LIGHT,'#ABC3D9',.7,.05)
    search_icon(ax,4.34,2.61,BLUE)
    txt(ax,4.58,2.70,'Subspace rankers',7.1,BLUE,'bold')
    txt(ax,4.58,2.50,'Gower · RLAX · SOD/COP',6.4)
    rule(ax,3.93,2.91,4.04,2.91,BLUE,.75)
    rule(ax,4.04,2.61,4.04,3.24,BLUE,.75)
    arrow(ax,4.04,3.24,4.15,3.24,PURPLE)
    arrow(ax,4.04,2.61,4.15,2.61,BLUE)
    rule(ax,5.79,3.24,5.90,3.24,PURPLE,.75)
    rule(ax,5.79,2.61,5.90,2.61,BLUE,.75)
    rule(ax,5.90,2.61,5.90,3.24,BLUE,.75)
    arrow(ax,5.90,2.91,6.04,2.91,BLUE)
    box(ax,6.05,2.42,.91,1.00,'white','#A7BDC9',.75,.05)
    txt(ax,6.505,3.28,'Ranked sets',7.1,BLUE,'bold',ha='center')
    for j in range(3):
        yy=3.08-j*.17
        txt(ax,6.18,yy,f'{j+1}.',6.7,GRAY)
        box(ax,6.39,yy-.058,.40,.116,[PURPLE_LIGHT,BLUE_LIGHT,GREEN_LIGHT][j],'none',0,.025)
        txt(ax,6.59,yy,f'$S_{j+1}$',7.2,ha='center')
    txt(ax,6.505,2.49,'up to four',6.4,GRAY,ha='center')
    txt(ax,3.08,2.18,'Controls: rarity · random · all fields',6.8,GRAY)
    txt(ax,3.08,2.04,'M$^2$OE: separate comparison with three unordered choices',6.45,GRAY)
    # Three traces use distinct evidence rather than a pooled score.
    panel(ax,.04,.045,7.08,1.75,'3. Evaluate each track with its own evidence')
    arrow(ax,2.04,1.95,2.04,1.80,ORANGE,dash=True)
    arrow(ax,6.50,1.95,6.50,1.80,BLUE)
    txt(ax,.20,1.42,'DATA / QUERY',6.6,GRAY,'bold')
    txt(ax,2.53,1.42,'EVALUATION REFERENCE',6.6,GRAY,'bold')
    txt(ax,4.83,1.42,'REPORTED OUTCOME',6.6,GRAY,'bold')
    rows=[
        (1.14,'Synthetic benchmark','864 test cases · three seeds',GREEN,GREEN_LIGHT,
         'Exhaustive support / rule sets','Suff@1 · Exact@1 · Hit@4', 'Jaccard · size · validity'),
        (.73,'Digital circuits','86 queries · four circuits',BLUE,BLUE_LIGHT,
         'Gate-negation input fields','Jaccard + field recall','Fault-specific localization'),
        (.32,'Real tables','78 queries · ten datasets',PURPLE,PURPLE_LIGHT,
         'Leave-one-model-out majority','Jaccard + peer coverage','Agreement; no feature truth')]
    for j,(yy,head,count,color,fc,evidence,metric,note) in enumerate(rows):
        box(ax,.14,yy-.17,6.87,.35,fc,'none',0,.04)
        if j==1:gate_icon(ax,.37,yy,color)
        else:table_icon(ax,.22,yy-.12,.30,.23,color,j==2)
        txt(ax,.66,yy+.067,head,7.3,color,'bold')
        txt(ax,.66,yy-.078,count,6.6,GRAY)
        arrow(ax,2.23,yy,2.43,yy,color)
        txt(ax,2.55,yy+.055,evidence,7.0,color,'bold')
        txt(ax,2.55,yy-.09,['Full-population labels','Known injected fault','Five model families'][j],6.4,GRAY)
        arrow(ax,4.50,yy,4.70,yy,color)
        txt(ax,4.83,yy+.06,metric,7.0,color,'bold')
        txt(ax,4.83,yy-.09,note,6.5,GRAY)
    save(fig,'figure2_overall_framework')


if __name__=='__main__':
    make_intro_figure()
    make_framework_figure()
