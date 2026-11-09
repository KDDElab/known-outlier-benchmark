"""Rebuild quantitative figures from completed, repaired experimental evidence."""
import json
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
RUN=ROOT/'synthetic_benchmark_v2/runs/ccfa_revision'
MODELS=['qwen35_9b','deepseek_v2_lite_chat','llama31_8b_instruct']
LABELS=dict(zip(MODELS,['Qwen','DeepSeek','Llama']))
LABELS.update(gower_lof_subspace='Gower-LOF',gower_knn_subspace='Gower-kNN',rlax_reference_adaptation='RLAX',all_features='All fields')
COLORS=dict(zip(MODELS,['#3973AD','#D1783F','#47866D']))
COLORS.update(gower_lof_subspace='#7961A8',gower_knn_subspace='#B16B9C',rlax_reference_adaptation='#B4912E',all_features='#4A5660')
INK='#293744'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.labelsize':7.3,'axes.titlesize':8,'xtick.labelsize':6.8,'ytick.labelsize':6.8,'legend.fontsize':6.7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK})
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(HERE/f'{name}.{ext}',dpi=240,facecolor='white')
 plt.close(fig)
def style(ax,axis='y'):
 ax.grid(axis=axis,color='#E1E6EB',lw=.45);ax.set_axisbelow(True);ax.tick_params(length=2.5,width=.5,pad=2)
 for side in ['left','bottom']:ax.spines[side].set_color('#86939F')
def pick(frame,method,metric,condition='main'):
 q=frame[(frame.method==method)&(frame.metric==metric)]
 if 'condition' in frame:q=q[q.condition==condition]
 assert len(q)==1,(method,metric,condition,len(q))
 return q.iloc[0]
def lines(ax,frame,conditions,metric,methods,ticklabels):
 for method in methods:
  points=[pick(frame,method,metric,c) for c in conditions]
  y=np.array([p.estimate for p in points]);lo=np.array([p.lo for p in points]);hi=np.array([p.hi for p in points])
  ax.errorbar(range(len(conditions)),y,yerr=np.vstack([y-lo,hi-y]),color=COLORS[method],marker='o' if method in MODELS else 's',ms=3.3,lw=1.,elinewidth=.65,capsize=1.8,ls='-' if method in MODELS else '--',label=LABELS[method])
 ax.set_xticks(range(len(conditions)),ticklabels);ax.set_xlim(-.22,len(conditions)-.78);style(ax)
def reference_and_order():
 raw=pd.read_csv(ROOT/'synthetic_benchmark_v2/runs/reference_sample_audit_multiseed/per_case_test.csv');cells=[]
 for (width,refs),d in raw.groupby(['feature_count','reference_count']):
  t=d.groupby(['generator_seed','template_group_id'])[['spurious_sample_minimal_count','gold_candidate_recall']].mean().groupby('template_group_id').mean()
  cells.append((width,refs,t.spurious_sample_minimal_count.mean(),t.gold_candidate_recall.mean()))
 summary=pd.DataFrame(cells,columns=['width','refs','extra','retained']);panel=pd.read_csv(RUN/'panel_summary.csv')
 fig=plt.figure(figsize=(7.16,2.60));a=fig.add_axes([.063,.23,.185,.61]);b=fig.add_axes([.36,.23,.22,.61]);c=fig.add_axes([.72,.23,.26,.61])
 a.set_title('(a) Extra minimal sets',loc='left',fontweight='bold',pad=8)
 matrix=np.array([[summary.query('width==@w and refs==@r').extra.iloc[0] for r in [8,16,32]] for w in [6,12,24]])
 im=a.imshow(matrix,cmap='YlGnBu',norm=LogNorm(vmin=min(matrix.flat),vmax=max(matrix.flat)),aspect='auto')
 for i in range(3):
  for j in range(3):a.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=7.1,color='white' if matrix[i,j]>25 else INK)
 a.set_xticks(range(3),['8','16','32']);a.set_yticks(range(3),['6','12','24']);a.set(xlabel='Reference rows',ylabel='Feature count');a.tick_params(length=0)
 ca=fig.add_axes([.26,.23,.011,.61]);cb=fig.colorbar(im,cax=ca);cb.ax.tick_params(labelsize=6,length=2,pad=1)
 b.set_title('(b) Population gold retained',loc='left',fontweight='bold',pad=8)
 for w,col,mark in zip([6,12,24],list(COLORS.values())[:3],['o','s','^']):
  q=summary.query('width==@w').sort_values('refs');b.plot(q.refs,q.retained,c=col,marker=mark,ms=3.6,lw=1,label=f'{w} fields')
 b.set(ylim=(0,1.04),xlabel='Reference rows',ylabel='Gold-set retention');b.set_xticks([8,16,32]);b.legend(frameon=False,loc='lower right',handlelength=1.1);style(b)
 c.set_title('(c) Position diagnostic',loc='left',fontweight='bold',pad=8)
 lines(c,panel,['front','main','back'],'support_any_minimal_hit',MODELS+['gower_lof_subspace'],['Front','Random','Back']);c.set(ylim=(0,1.03),ylabel='Hit@4')
 handles=[Line2D([],[],color=COLORS[m],marker='o' if m in MODELS else 's',ls='-' if m in MODELS else '--',lw=1,label=LABELS[m]) for m in MODELS+['gower_lof_subspace']]
 fig.legend(handles=handles,frameon=False,ncol=4,loc='lower right',bbox_to_anchor=(.98,.005),columnspacing=.8,handlelength=1.2)
 save(fig,'figure3_reference_and_order_robustness')
def localization_and_outputs():
 main=pd.read_csv(RUN/'main_summary.csv');scores=pd.read_csv(RUN/'all_scores.csv',low_memory=False);main_scores=scores[scores.condition=='main']
 fig,axes=plt.subplots(2,2,figsize=(7.16,3.90));fig.subplots_adjust(left=.085,right=.985,top=.92,bottom=.18,wspace=.38,hspace=.65);a,b,c,d=axes.flat
 methods=MODELS+['gower_lof_subspace','gower_knn_subspace','rlax_reference_adaptation','all_features']
 a.set_title('(a) Size and quality of answered queries',loc='left',fontweight='bold',pad=6)
 for method in methods:
  x=pick(main,method,'top1_fraction');y=pick(main,method,'answered_support_sufficiency')
  a.errorbar(x.estimate,y.estimate,xerr=[[x.estimate-x.lo],[x.hi-x.estimate]],yerr=[[y.estimate-y.lo],[y.hi-y.estimate]],fmt='^' if method in MODELS else 'o',ms=4.2,color=COLORS[method],elinewidth=.7,capsize=1.7,label=LABELS[method])
 a.set(xlim=(0,1.07),ylim=(0,1.07),xlabel='First-candidate field fraction',ylabel='Suff@1 | answered');style(a,'both')
 b.set_title('(b) Validity and answer coverage',loc='left',fontweight='bold',pad=6)
 for i,method in enumerate(MODELS):
  valid=pick(main,method,'structurally_valid').estimate;answered=pick(main,method,'answered').estimate
  b.bar(i-.17,valid,.30,fc='white',ec=COLORS[method],lw=1,hatch='///');b.bar(i+.17,answered,.30,color=COLORS[method]);b.text(i+.17,answered+.025,f'{answered:.3f}',ha='center',fontsize=6.3)
 b.set_xticks(range(3),[LABELS[m] for m in MODELS]);b.set(ylim=(0,1.16),ylabel='Template-macro rate')
 b.legend(handles=[Patch(fc='white',ec='#667887',hatch='///',label='Valid'),Patch(fc='#667887',label='Answered')],frameon=False,loc='upper right',ncol=2,handlelength=1);style(b)
 c.set_title('(c) All-query outcome decomposition',loc='left',fontweight='bold',pad=6)
 statecols=['#CFD6DD','#D9B988','#77A0C6','#3B7965'];states=['Invalid / empty','Abstained','Answered, inexact','Exact@1']
 for i,method in enumerate(MODELS):
  q=main_scores[main_scores.method==method].copy();abstain=q.abstain.map(lambda v:str(v).lower()=='true');q['state0']=~q.answered.astype(bool)&~abstain;q['state1']=abstain;q['state2']=q.answered.astype(bool)&~q.support_top1_minimal.astype(bool);q['state3']=q.support_top1_minimal.astype(bool)
  vals=q.groupby(['generator_seed','template_id'])[[f'state{k}' for k in range(4)]].mean().groupby('template_id').mean().mean().to_numpy();assert np.isclose(sum(vals),1.);offset=0.
  for v,col in zip(vals,statecols):c.barh(i,v,left=offset,height=.50,color=col,ec='white',lw=.5);offset+=v
 c.set_yticks(range(3),[LABELS[m] for m in MODELS]);c.invert_yaxis();c.set(xlim=(0,1),xlabel='Template-macro fraction')
 fig.legend(handles=[Patch(fc=col,label=name) for col,name in zip(statecols,states)],loc='lower center',bbox_to_anchor=(.275,.012),ncol=2,frameon=False,handlelength=1,columnspacing=.8,fontsize=6.5);style(c,'x')
 d.set_title('(d) Exact recovery and candidate retrieval',loc='left',fontweight='bold',pad=6)
 show=MODELS+['gower_lof_subspace','gower_knn_subspace']
 for i,method in enumerate(show):
  exact=pick(main,method,'support_top1_minimal');hit=pick(main,method,'support_any_minimal_hit');d.plot([exact.estimate,hit.estimate],[i,i],color=COLORS[method],lw=1.2)
  for result,marker in [(exact,'o'),(hit,'s')]:d.errorbar(result.estimate,i,xerr=[[result.estimate-result.lo],[result.hi-result.estimate]],fmt=marker,ms=3.4,color=COLORS[method],elinewidth=.6,capsize=1.4)
 d.set_yticks(range(len(show)),[LABELS[m] for m in show]);d.invert_yaxis();d.set(xlim=(0,1),xlabel='All-query score');style(d,'x')
 fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#667887',label='Exact@1'),Line2D([],[],marker='s',ls='',color='#667887',label='Hit@4')],frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.78,.025),fontsize=6.5)
 a.legend(frameon=False,fontsize=6.3,ncol=2,loc='lower right',handlelength=1.0,columnspacing=.7)
 save(fig,'figure2_sufficiency_compactness_validity')
def matched_supplements():
 panel=pd.read_csv(RUN/'panel_summary.csv');fig,axes=plt.subplots(2,2,figsize=(7.16,3.75));fig.subplots_adjust(left=.08,right=.97,top=.88,bottom=.11,wspace=.28,hspace=.62)
 specifications=[(axes[0,0],'(a) Nested references: retrieval',['refs8','refs16','main'],'support_any_minimal_hit',MODELS+['gower_lof_subspace','gower_knn_subspace'],['8','16','32']),(axes[0,1],'(b) Nested references: coverage',['refs8','refs16','main'],'answered',MODELS,['8','16','32']),(axes[1,0],'(c) Output contract: retrieval',['main','full1024','minimal512'],'support_any_minimal_hit',MODELS,['Full / 512','Full / 1024','Minimal / 512']),(axes[1,1],'(d) Output contract: coverage',['main','full1024','minimal512'],'answered',MODELS,['Full / 512','Full / 1024','Minimal / 512'])]
 for ax,title,conditions,metric,methods,ticks in specifications:
  ax.set_title(title,loc='left',fontweight='bold',pad=6);lines(ax,panel,conditions,metric,methods,ticks);ax.set(ylim=(0,1.07),ylabel='Hit@4' if metric.startswith('support') else 'Answered rate')
 axes[0,0].set_xlabel('Normal reference rows');axes[0,1].set_xlabel('Normal reference rows')
 handles=[Line2D([],[],color=COLORS[m],marker='o' if m in MODELS else 's',ls='-' if m in MODELS else '--',lw=1,label=LABELS[m]) for m in MODELS+['gower_lof_subspace','gower_knn_subspace']]
 fig.legend(handles=handles,frameon=False,ncol=5,loc='upper center',bbox_to_anchor=(.52,1.0),handlelength=1.4,columnspacing=1.1);save(fig,'figure5_matched_reference_and_output')
if __name__=='__main__':
 assert json.loads((RUN/'analysis_manifest.json').read_text())['complete'],'Complete results required'
 reference_and_order();localization_and_outputs();matched_supplements()
