"""Redraw the published result panels from the frozen manuscript artifact.

No experiment is rerun. Pareto and runtime values are the frozen values used by
synthetic_benchmark_v2/make_review_revision_figures.py; the other panels read
case-level audits, bootstrap effects, and output-decomposition tables directly.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ART=ROOT/'artifact_release_staging/synthetic_benchmark_v2'
MET=ROOT/'artifact_release_staging/manuscript_figures'
C=['#3973AD','#D1783F','#47866D']
MODELS=['Qwen3.5-9B','DeepSeek-V2-Lite-Chat','Llama-3.1-8B-Instruct']
SHORT=['Qwen','DeepSeek','Llama']
INK='#293744'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,
 'axes.titlesize':7.5,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.3,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK})


def save(f,name):
 for ext in ['pdf','svg','png']: f.savefig(HERE/f'{name}.{ext}',dpi=240,facecolor='white')
 plt.close(f)


def style(ax,axis='y'):
 ax.grid(axis=axis,color='#E1E6EB',lw=.45);ax.set_axisbelow(True)
 ax.tick_params(length=2.5,width=.5,pad=2)
 for side in ['left','bottom']:ax.spines[side].set_color('#86939F')


def localization_and_outputs():
 f,axs=plt.subplots(2,2,figsize=(7.16,3.94),gridspec_kw={'height_ratios':[1.17,1]})
 f.subplots_adjust(left=.09,right=.98,top=.92,bottom=.105,wspace=.39,hspace=.62)
 a,b,c,d=axs.flat
 # Tuples: name, size mean/low/high, sufficiency mean/low/high, label anchor.
 data=[('Qwen',.207,.186,.228,.714,.617,.798,(.03,.89),C[0]),
       ('DeepSeek',.371,.344,.397,.429,.384,.470,(.53,.32),C[1]),
       ('Llama',.199,.195,.204,.394,.327,.457,(.04,.26),C[2]),
       ('Gower-LOF',.181,.175,.187,.545,.421,.660,(.015,.60),'#7961A8'),
       ('Gower-kNN',.229,.220,.237,.608,.500,.707,(.38,.74),'#B16B9C'),
       ('RLAX',.234,.226,.242,.507,.376,.628,(.47,.51),'#B4912E'),
       ('All fields',1,1,1,1,1,1,(.66,.96),'#4A5660')]
 a.set_title('(a) Sufficiency versus subset size',loc='left',fontweight='bold',pad=5)
 for name,x,lo,hi,y,yl,yh,pos,col in data:
  a.errorbar(x,y,xerr=[[x-lo],[hi-x]],yerr=[[y-yl],[yh-y]],fmt='none',
             ecolor=col,alpha=.45,elinewidth=.75,capsize=1.5,zorder=2)
  a.scatter(x,y,s=27,marker='^' if name in SHORT else ('s' if name=='All fields' else 'o'),
            color=col,ec='white',lw=.5,zorder=4)
  a.annotate(name,(x,y),xytext=pos,fontsize=6.4,color=col,
              arrowprops={'arrowstyle':'-','lw':.55,'color':col},zorder=5)
 a.set(xlim=(0,1.07),ylim=(.15,1.08),xlabel='Selected-field fraction',ylabel='Support sufficiency@1')
 a.set_xticks([0,.25,.5,.75,1]);style(a,'both')
 # Deployed and valid-only sufficiency are a focused pair, avoiding tiny multiples.
 frame=pd.read_csv(MET/'parseable_decomposition.csv')
 b.set_title('(b) Effect of invalid responses',loc='left',fontweight='bold',pad=5)
 for i,model in enumerate(MODELS):
  row=frame.loc[frame.model==model].iloc[0]
  y0=float(row['sufficiency@1_deployed']);y1=float(row['sufficiency@1_parseable'])
  b.bar(i-.18,y0,.31,fc=C[i],ec=C[i],lw=.6,zorder=3)
  b.bar(i+.18,y1,.31,fc='white',ec=C[i],lw=1.0,hatch='///',zorder=3)
  b.text(i-.18,y0+.025,f'{y0:.3f}',fontsize=6.2,ha='center')
  b.text(i+.18,y1+.025,f'{y1:.3f}',fontsize=6.2,ha='center')
 b.set(ylim=(0,.95),ylabel='Support sufficiency@1');b.set_xticks(range(3),SHORT)
 b.legend(handles=[Patch(fc='#667887',label='All queries'),Patch(fc='white',ec='#667887',hatch='///',label='Valid only')],
           frameon=False,loc='upper right',ncol=2,handlelength=1,columnspacing=.65,fontsize=6.0,borderpad=0)
 style(b)
 c.set_title('(c) Output validity',loc='left',fontweight='bold',pad=5)
 valid=np.array([856,696,706])/864;strict=np.array([854,334,571])/864
 for i,(v,s) in enumerate(zip(valid,strict)):
  c.barh(i-.15,v,.25,color=C[i],zorder=3)
  c.barh(i+.15,s,.25,color=C[i],alpha=.30,zorder=3)
  c.text(v+.015,i-.15,f'{v:.3f}',fontsize=6.0,va='center')
  c.text(s+.015,i+.15,f'{s:.3f}',fontsize=6.0,va='center')
 c.set_yticks(range(3),SHORT);c.invert_yaxis();c.set(xlim=(0,1.14),xlabel='Fraction of 864 queries')
 c.legend(handles=[Patch(fc='#667887',label='Schema valid'),Patch(fc='#667887',alpha=.3,label='Strict JSON')],
           loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,ncol=2,handlelength=1,columnspacing=.8,fontsize=6.0,borderpad=0)
 c.set_title('(c) Output validity',loc='left',fontweight='bold',pad=19)
 style(c,'x')
 d.set_title('(d) Latency per query',loc='left',fontweight='bold',pad=19)
 means=[11.98,17.38,7.20];p95=[17.83,36.90,15.09]
 for i,(v,h) in enumerate(zip(means,p95)):
  d.plot([v,h],[i,i],c=C[i],lw=2,alpha=.55,zorder=2)
  d.scatter([v],[i],s=25,c=C[i],ec='white',lw=.4,zorder=3)
  d.scatter([h],[i],s=45,c=C[i],marker='|',linewidths=1.2,zorder=3)
  d.text(v,i-.19,f'{v:.2f}',ha='center',va='bottom',fontsize=6.3,color=C[i])
  d.text(h,i+.19,f'{h:.2f}',ha='center',va='top',fontsize=6.3,color=C[i])
 d.set_yticks(range(3),['Qwen (1 GPU)','DeepSeek (2 GPUs)','Llama (1 GPU)'])
 d.set(xlim=(0,42),ylim=(2.55,-.55),xlabel='Seconds');d.tick_params(axis='y',labelsize=6.0)
 d.legend(handles=[Line2D([],[],c='#667887',marker='o',ls='',label='Mean'),Line2D([],[],c='#667887',marker='|',ls='',label='95th percentile')],
           loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,ncol=2,handlelength=1,columnspacing=.7,fontsize=6.0,borderpad=0)
 style(d,'x')
 save(f,'figure2_sufficiency_compactness_validity')


def reference_and_order():
 raw=pd.read_csv(ART/'runs/reference_sample_audit_multiseed/per_case_test.csv')
 cells=[]
 for (width,refs),g in raw.groupby(['feature_count','reference_count']):
  t=g.groupby('template_group_id').agg(extra=('spurious_sample_minimal_count','mean'),retained=('gold_candidate_recall','mean'))
  cells.append((width,refs,t.extra.mean(),t.retained.mean()))
 summary=pd.DataFrame(cells,columns=['width','refs','extra','retained'])
 f=plt.figure(figsize=(7.16,2.58))
 # Extra room for the order metric labels makes the final-size type legible.
 a=f.add_axes([.065,.21,.19,.64]);b=f.add_axes([.335,.21,.19,.64]);c=f.add_axes([.73,.21,.25,.64])
 a.set_title('(a) Extra minimal sets',loc='left',fontweight='bold',pad=7)
 m=np.array([[summary.query('width==@w and refs==@r').extra.iloc[0] for r in [8,16,32]] for w in [6,12,24]])
 im=a.imshow(m,cmap='YlGnBu',norm=LogNorm(vmin=4,vmax=250),aspect='auto')
 for i in range(3):
  for j in range(3):a.text(j,i,f'{m[i,j]:.1f}',ha='center',va='center',fontsize=7,color='white' if m[i,j]>25 else INK)
 a.set_xticks(range(3),['8','16','32']);a.set_yticks(range(3),['6','12','24'])
 a.set(xlabel='Reference rows',ylabel='Feature count');a.tick_params(length=0)
 ca=f.add_axes([.266,.21,.012,.64]);cb=f.colorbar(im,cax=ca);cb.ax.tick_params(labelsize=5.8,length=2,pad=1)
 cb.ax.set_title('Count',fontsize=5.8,pad=5)
 b.set_title('(b) Gold retained',loc='left',fontweight='bold',pad=7)
 for w,color,marker in zip([6,12,24],C,['o','s','^']):
  q=summary.query('width==@w').sort_values('refs')
  b.plot(q.refs,q.retained,c=color,marker=marker,ms=3.7,lw=1,label=f'{w} fields')
 b.set(xlim=(7,33),ylim=(.55,1.02),xlabel='Reference rows',ylabel='')
 b.set_xticks([8,16,32]);b.set_yticks([.6,.8,1]);b.legend(frameon=False,loc='lower right',fontsize=6.0,handlelength=1.2,borderpad=.1)
 style(b)
 c.set_title('(c) Field-order effect',loc='left',fontweight='bold',pad=7)
 tests=pd.read_csv(ART/'runs/expanded_random_field_order_288/random_order_pairwise_tests.csv')
 keys=['support_top1_sufficient','support_top1_minimal','support_any_minimal_hit','support_best_jaccard']
 names=['Suff.','Exact','Hit@4','Jacc.'];markers=['o','s','D','^'];ys=[];labels=[]
 for i,model in enumerate(MODELS):
  for j,key in enumerate(keys):
   row=tests.query('model==@model and metric==@key').iloc[0];yy=i*4.7+j
   v=row.random_minus_original;lo=row.ci95_low;hi=row.ci95_high
   c.errorbar(v,yy,xerr=[[v-lo],[hi-v]],fmt=markers[j],color=C[i],ms=3.1,elinewidth=.8,capsize=1.4)
   ys.append(yy);labels.append(f'{SHORT[i]}  {names[j]}')
 c.set_yticks(ys,labels);c.tick_params(axis='y',labelsize=6.0,pad=2,length=0)
 c.set(xlim=(-.275,.275),ylim=(ys[-1]+.6,-.6),xlabel='Random order − original')
 c.set_xticks([-.2,0,.2]);c.axvline(0,c='#818C95',ls='--',lw=.7)
 c.spines['left'].set_visible(False);style(c,'x');c.tick_params(axis='y',length=0)
 save(f,'figure3_reference_and_order_robustness')


if __name__=='__main__':
 localization_and_outputs()
 reference_and_order()
