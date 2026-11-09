"""Reference-led paper figures with reproducible data and editable vector objects.

User references: 图117.png (three semantic regions and colored headings),
图118.png (typed left-to-right flow and pale modules), 图1171.png (overview
above an aligned, worked-example comparison), 图1172.png (highlighted real
records and structured result tables). Figure 2 uses 图1171 as primary layout.
Figure 1 adapts the record-to-output composition to one column.

Final canvases: 3.49 x 3.50 in and 7.16 x 3.82 in. Coordinates are inches.
Paper labels use Times-compatible Liberation Serif; source data and scoring
are frozen. Generic icons are native vector primitives. Qwen uses the user's
supplied bitmap logo, preserved byte-for-byte in qwen_logo_user.png; the native
figure viewport shows its rounded blue tile without the screenshot margins.
Qwen official identity: https://github.com/QwenLM (checked 2026-11-08).
Figure 2 also places the official Meta Llama avatar and the original DeepSeek
whale SVG path. Their source URLs and exact placement are documented below.
No training stages, reward mechanisms or invented examples are transferred
from the reference images. Source functions bind every displayed case result.
"""
from __future__ import annotations
import ast
import csv
import json
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle, Ellipse, Polygon
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
INK='#253347'; GRAY='#647084'; GRID='#CFD7E1'
GREEN='#64995D'; GREEN_BG='#EEF5EB'
BLUE='#4879BF'; BLUE_BG='#EEF3FB'
PURPLE='#8068AD'; PURPLE_BG='#F3EFF8'
WARM='#BD8351'; WARM_BG='#FFF7EC'
RED='#BE635A'; RED_BG='#FAECE8'
TEAL='#4D9381'; TEAL_BG='#ECF5EF'
plt.rcParams.update({'font.family':'Liberation Serif','font.size':8.0,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'axes.linewidth':.6,'mathtext.fontset':'stix'})


def txt(ax,x,y,s,size=8,color=INK,weight='normal',ha='left',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,
                   va=va,linespacing=1.13,zorder=10,**kw)


def box(ax,x,y,w,h,face='white',edge=GRID,lw=.65,r=.055,dash=False,z=2):
    q=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
        fc=face,ec=edge,lw=lw,linestyle=(0,(4,2.5)) if dash else '-',zorder=z)
    ax.add_patch(q);return q


def rule(ax,x1,y1,x2,y2,color=GRID,lw=.7,**kw):
    ax.plot([x1,x2],[y1,y2],color=color,lw=lw,zorder=9,**kw)


def arrow(ax,x1,y1,x2,y2,color=BLUE,lw=1,dash=False,rad=0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',
        mutation_scale=7.5,connectionstyle=f'arc3,rad={rad}',color=color,
        lw=lw,linestyle=(0,(3,2)) if dash else '-',shrinkA=0,shrinkB=0,zorder=6))


def doc(ax,x,y,w=.20,h=.25,color=BLUE,kind='table'):
    # Document outline with a folded corner and aligned content rules.
    f=w*.27
    ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+h-f),(x+w-f,y+h),(x,y+h)],
                         closed=True,fc='white',ec=color,lw=.8,zorder=7))
    rule(ax,x+w-f,y+h,x+w-f,y+h-f,color,.65)
    rule(ax,x+w-f,y+h-f,x+w,y+h-f,color,.65)
    for k in range(3):rule(ax,x+w*.17,y+h*(.22+.20*k),x+w*.80,y+h*(.22+.20*k),color,.65)
    if kind=='table':
        rule(ax,x+w*.46,y+h*.14,x+w*.46,y+h*.69,color,.5)
    elif kind=='medical':
        box(ax,x+w*.60,y-.02,w*.50,h*.31,'white',color,.65,.015,z=8)
        rule(ax,x+w*.85,y+.005,x+w*.85,y+h*.20,color,.85)
        rule(ax,x+w*.68,y+h*.10,x+w*1.02,y+h*.10,color,.85)


def model(ax,x,y,s=.26,color=BLUE):
    # Neural processor icon: fixed common optical size across repeated rows.
    box(ax,x-s/2,y-s/2,s,s,'white',color,.85,.035,z=7)
    for t in [-.28,0,.28]:
        rule(ax,x-s*.69,y+s*t,x-s*.50,y+s*t,color,.7)
        rule(ax,x+s*.50,y+s*t,x+s*.69,y+s*t,color,.7)
    pts=[(x-s*.26,y+s*.22),(x-s*.26,y-s*.22),(x+s*.24,y)]
    for xx,yy in pts[:2]:rule(ax,xx,yy,*pts[2],color,.6)
    for xx,yy in pts:ax.add_patch(Circle((xx,yy),s*.075,fc=color,ec='none',zorder=8))


def qwen_logo(ax,x,y,s=.20):
    """Place the supplied logo as an image asset in the editable figure.

    Source: /root/nas/SPELT-master/千问logo.png (60 x 72 pixels).
    The logo tile occupies [6,54] x [14,62] in that screenshot. A native
    rounded viewport excludes surrounding UI; the source PNG is unmodified.
    """
    logo=plt.imread(HERE/'qwen_logo_user.png')
    assert logo.shape[:2]==(72,60)
    scale=s/48
    artist=ax.imshow(logo,origin='upper',interpolation='lanczos',aspect='auto',
        extent=(x-30*scale,x+30*scale,y-34*scale,y+38*scale),zorder=8)
    viewport=FancyBboxPatch((x-s/2,y-s/2),s,s,
        boxstyle=f'round,pad=0,rounding_size={s*.20}',transform=ax.transData)
    artist.set_clip_path(viewport)


def llama_logo(ax,x,y,s=.26):
    """Unmodified logo from the official Meta Llama Hugging Face profile.

    https://huggingface.co/meta-llama, retrieved 2026-11-08.
    Original avatar: https://cdn-avatars.huggingface.co/v1/production/uploads/
    646cf8084eefb026fb8fd8bc/oCTqufkdTkjyGodsx1vo1.png
    Meta trademark; used to identify the evaluated Llama model family.
    The official avatar is the blue Meta infinity, not the llama.cpp mascot.
    """
    # The official URL ends in .png but serves WebP; detect the actual bytes.
    with Image.open(HERE/'llama_logo_official.webp') as source:
        logo=np.asarray(source.convert('RGBA'))
    # Preserve the source canvas and aspect ratio, including its clear space.
    ax.imshow(logo,origin='upper',interpolation='lanczos',aspect='auto',
              extent=(x-s*.70,x+s*.70,y-s*.70,y+s*.70),zorder=8)


@lru_cache(maxsize=1)
def deepseek_mark():
    """Read the original whale path; retain exact cubic Bezier geometry.

    https://github.com/deepseek-ai/DeepSeek-LLM/blob/main/images/logo.svg
    Retrieved 2026-11-08. DeepSeek trademark, used for model identification.
    Only the separate whale element is placed; the adjacent wordmark is
    replaced by the same live model-name typography used in the other nodes.
    """
    root=ET.parse(HERE/'deepseek_logo_official.svg').getroot()
    element=root.find('{http://www.w3.org/2000/svg}path')
    tokens=re.findall(r'[MCZ]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',element.attrib['d'])
    assert set(re.findall(r'[A-Za-z]',element.attrib['d']))=={'M','C','Z'}
    vertices=[];codes=[];i=0;command=None;start=None
    while i<len(tokens):
        if tokens[i] in {'M','C','Z'}:
            command=tokens[i];i+=1
        if command=='Z':
            vertices.append(start);codes.append(MplPath.CLOSEPOLY);command=None
        elif command=='M':
            point=tuple(map(float,tokens[i:i+2]));i+=2
            vertices.append(point);codes.append(MplPath.MOVETO);start=point
        elif command=='C':
            values=list(map(float,tokens[i:i+6]));i+=6
            vertices.extend(zip(values[::2],values[1::2]));codes.extend([MplPath.CURVE4]*3)
        else:raise ValueError('Unsupported command in official logo')
    return MplPath(vertices,codes),element.attrib['fill']


def deepseek_logo(ax,x,y,s=.28):
    path,color=deepseek_mark();b=path.get_extents();scale=s/b.width
    transform=Affine2D().translate(-b.x0-b.width/2,-b.y0-b.height/2).scale(scale,-scale).translate(x,y)
    ax.add_patch(PathPatch(path,transform=transform+ax.transData,fc=color,ec='none',zorder=8))


def search(ax,x,y,s=.27,color=BLUE):
    doc(ax,x-s*.45,y-s*.43,s*.58,s*.75,color)
    ax.add_patch(Circle((x+s*.13,y+s*.05),s*.29,fc='white',ec=color,lw=.85,zorder=8))
    rule(ax,x+s*.34,y-s*.16,x+s*.55,y-s*.38,color,1.2)
    for dx,dy in [(-.03,.08),(.18,.13),(.12,-.08)]:
        ax.add_patch(Circle((x+s*dx,y+s*dy),s*.035,fc=color,ec='none',zorder=9))


def circuit(ax,x,y,s=.24,color=GREEN):
    box(ax,x-s*.20,y-s*.28,s*.50,s*.56,'white',color,.8,.045,z=7)
    for yy in [-.18,.18]:rule(ax,x-s*.48,y+s*yy,x-s*.20,y+s*yy,color,.8)
    rule(ax,x+s*.30,y,x+s*.56,y,color,.8)
    txt(ax,x+s*.05,y,'&',6.7,color,ha='center')


def database(ax,x,y,s=.24,color=GREEN):
    ax.add_patch(Rectangle((x-s*.43,y-s*.28),s*.86,s*.56,fc='white',ec=color,lw=.7,zorder=7))
    for yy in [-.28,0,.28]:
        ax.add_patch(Ellipse((x,y+s*yy),s*.86,s*.26,fc='white',ec=color,lw=.7,zorder=8))


def status(ax,x,y,yes):
    c=TEAL if yes else RED
    ax.add_patch(Circle((x,y),.065,fc=TEAL_BG if yes else RED_BG,ec=c,lw=.7,zorder=7))
    if yes:
        ax.plot([x-.034,x-.008,x+.039],[y,y-.026,y+.030],color=c,lw=1,zorder=9)
    else:
        for sign in [-1,1]:rule(ax,x-.026,y+sign*.026,x+.026,y-sign*.026,c,.9)


def canvas(w,h):
    f=plt.figure(figsize=(w,h),facecolor='white')
    a=f.add_axes([0,0,1,1]);a.set_xlim(0,w);a.set_ylim(0,h);a.axis('off')
    return f,a


def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(HERE/f'{name}.{ext}',dpi=300,facecolor='white',
          metadata={'Creator':'make_manuscript_figures.py'} if ext=='pdf' else None)
    plt.close(fig)


def pima_case():
    """Bind the real query, archived Qwen ranking and leave-one-model-out peers."""
    d=np.load(ROOT/'data/adbench/29_Pima.npz')
    cases=list(csv.DictReader((ROOT/'manifests/cases.csv').open()))
    c=next(c for c in cases if c['case_id']=='C0035')
    refs=json.loads(c['normal_reference_source_row_indices'])
    annotations=[json.loads(line) for line in (ROOT/'runs/annotations.jsonl').open()]
    q=next(row for row in annotations
           if row.get('case_id')=='C0035' and row.get('annotator_id')=='qwen35_9b')
    reports=[row for row in csv.DictReader((ROOT/'reports/leave_one_model_out_by_case.csv').open())
             if row['case_id']=='C0035']
    score=next(row for row in reports if row['held_out_annotator_id']=='qwen35_9b')
    peer_ids=ast.literal_eval(score['peer_annotators'])
    peer_rows={row['held_out_annotator_id']:ast.literal_eval(row['held_out_features'])
               for row in reports if row['held_out_annotator_id'] in peer_ids}
    fields=[f'F{i:03d}' for i in range(1,9)]
    votes=[sum(fid in values for values in peer_rows.values()) for fid in fields]
    pred=q['parsed_annotation']['feature_ids_ranked']
    peer=[fid for fid,n in zip(fields,votes) if n>len(peer_ids)/2]
    assert len(peer_rows)==len(peer_ids)==4
    assert peer==ast.literal_eval(score['peer_consensus_features'])
    assert pred==ast.literal_eval(score['held_out_features'])
    assert len(set(pred)&set(peer))/len(set(pred)|set(peer))==float(score['jaccard'])
    return d['X'],d['y'],int(c['source_row_index']),refs,pred,peer,votes,score


def make_intro_figure():
    """One-column real-data companion to the accepted Figure 2.

    Same thin orange frame, title bands, neutral panels and field-index grids.
    All values, ranks and peer votes come from case C0035. The scatter omits
    rows with a zero glucose or BMI only for visualization (752/768 remain).
    Models receive eight opaque fields, the target and twelve reference rows;
    semantic names in the axes are reader aids. Peer agreement is not gold.
    """
    X,y,target,refs,pred,peer,votes,score=pima_case()
    fig,ax=canvas(3.49,3.50)
    edge='#DEB57C';band='#FFDBA6';neutral='#F8F8F7'
    warm='#A46532'
    fields=[f'F{i:03d}' for i in range(1,9)]
    shared=set(pred)&set(peer)
    def panel(y0,h,title):
        box(ax,.045,y0,3.40,h,'white',edge,.75,.065)
        box(ax,.045,y0+h-.235,3.40,.235,band,band,0,.06)
        txt(ax,.15,y0+h-.12,title,8.8,weight='bold')
    panel(1.52,1.93,f'(a) A real Pima query: record {target}')
    database(ax,.23,3.105,.19,GRAY)
    txt(ax,.39,3.105,f'{len(X)} rows  ·  {len(refs)} references',7.8)
    txt(ax,2.84,3.105,'Target x*',8.0,weight='bold',ha='center')

    # The two-dimensional view supplies context, not a field-level gold label.
    valid=(X[:,1]>0)&(X[:,5]>0)
    rr=np.asarray(refs);rr=rr[valid[rr]]
    assert len(rr)==len(refs)
    for xx,label,kind in [(.20,'Rows','row'),(.82,'References','ref'),(1.76,'x*','target')]:
        if kind=='row':ax.scatter([xx],[2.948],s=10,c='#C1CBD3',edgecolors='none',zorder=8)
        elif kind=='ref':ax.scatter([xx],[2.948],s=17,fc='white',ec=BLUE,lw=.7,zorder=8)
        else:ax.scatter([xx],[2.948],s=30,marker='*',c=warm,edgecolors='white',lw=.3,zorder=8)
        txt(ax,xx+.10,2.948,label,7.0,GRAY)
    sc=fig.add_axes([.405/3.49,1.975/3.50,1.68/3.49,.885/3.50])
    sc.scatter(X[valid,1],X[valid,5],s=3.5,c='#C1CBD3',alpha=.65,ec='none',zorder=2)
    sc.scatter(X[rr,1],X[rr,5],s=15,fc='white',ec=BLUE,lw=.7,zorder=4)
    sc.scatter(X[target,1],X[target,5],marker='*',s=64,c=warm,ec='white',lw=.5,zorder=6)
    sc.annotate(f'x* ({X[target,1]:g}, {X[target,5]:g})',(X[target,1],X[target,5]),xytext=(150,58),
        fontsize=7.0,color=warm,ha='center',
        arrowprops={'arrowstyle':'-','color':warm,'lw':.65},zorder=7)
    sc.set(xlim=(35,205),ylim=(14,70),xlabel='F002: glucose (mg/dL)',ylabel='F006: BMI')
    sc.set_xticks([40,80,120,160,200]);sc.set_yticks([20,40,60])
    sc.tick_params(labelsize=6.8,length=2,width=.5,pad=1.5)
    sc.xaxis.label.set_size(7.2);sc.yaxis.label.set_size(7.2)
    sc.xaxis.labelpad=2;sc.yaxis.labelpad=2
    for side in ['top','right']:sc.spines[side].set_visible(False)
    for side in ['left','bottom']:sc.spines[side].set_color('#8D98A3')

    # Full target values remain visible even though the scatter uses two fields.
    box(ax,2.28,1.745,1.055,1.255,neutral,'#ADB6C0',.6,.035,dash=True)
    txt(ax,2.36,2.91,'Field',7.4,GRAY)
    txt(ax,3.25,2.91,'Value',7.4,GRAY,ha='right')
    rule(ax,2.34,2.84,3.28,2.84,'#BFC6CF',.5)
    for j,fid in enumerate(fields):
        yy=2.765-j*.132
        if j in [1,5]:box(ax,2.33,yy-.058,.95,.116,'#E7EFF6','none',0,.014)
        txt(ax,2.37,yy,fid,7.3,BLUE if j in [1,5] else INK)
        txt(ax,3.25,yy,f'{X[target,j]:g}',7.4,BLUE if j in [1,5] else INK,ha='right')
    txt(ax,1.745,1.615,'Model input: x* + 12 references + eight field IDs',7.7,ha='center')
    arrow(ax,1.745,1.52,1.745,1.44,'#727579',.8)

    panel(.045,1.395,'(b) Recorded selections and peer agreement')
    # Column alignment exposes shared fields, differing fields and rank order.
    cx=[1.38+j*.256 for j in range(8)]
    txt(ax,.15,1.115,'Field',7.4,GRAY)
    for xx,fid in zip(cx,fields):txt(ax,xx,1.115,fid,6.7,GRAY,ha='center')
    for yy in [.94,.445]:box(ax,.12,yy-.115,3.24,.23,neutral,'none',0,.03)
    qwen_logo(ax,.245,.948,.18)
    txt(ax,.39,.965,'Qwen',8.0,weight='bold')
    txt(ax,.39,.857,'rank 1–3',7.0,GRAY)
    txt(ax,.15,.72,'Peer votes',7.8,weight='bold')
    txt(ax,.15,.605,'four models',7.0,GRAY)
    txt(ax,.15,.46,'Majority',7.8,weight='bold')
    txt(ax,.15,.345,'≥3 votes',7.0,GRAY)
    for xx,fid,n in zip(cx,fields,votes):
        for yy,selection in [(.942,pred),(.445,peer)]:
            chosen=fid in selection
            col=BLUE if fid in shared else warm
            box(ax,xx-.099,yy-.084,.198,.168,
                col if chosen else 'white',col if chosen else GRID,.55,.014)
            if chosen:
                label=str(pred.index(fid)+1) if selection is pred else '•'
                txt(ax,xx,yy,label,7.7,'white',ha='center')
        # A four-unit bar with its exact count, including visible zero baselines.
        box(ax,xx-.061,.60,.122,.14,'#EEF0F2','none',0,.008)
        if n:
            box(ax,xx-.061,.60,.122,.14*n/4,
                BLUE if fid in shared else warm,'none',0,.008)
        txt(ax,xx,.785,str(n),7.1,GRAY if n==0 else INK,ha='center')
    rule(ax,.15,.278,3.34,.278,'#D3D6DB',.55)
    n_shared=len(shared);n_union=len(set(pred)|set(peer))
    txt(ax,1.745,.207,f"Shared {n_shared} / union {n_union} = {float(score['jaccard']):.3f} Jaccard",7.8,ha='center')
    txt(ax,1.745,.095,'Agreement only · no field-level truth',7.5,PURPLE,ha='center')
    save(fig,'figure_intro_real_dataset')


def invoice_example():
    """Read the first frozen tax-total case, selected by relation and index.

    Labels and population stay in the evaluator lane. Semantic column names
    are reader aids; model prompts contain eight opaque IDs and 32 references.
    The displayed method rows are a subset of the five executed invoice methods.
    """
    root=ROOT/'synthetic_benchmark_v2/runs/ccfa_constraint_transfer'
    case_id='constraint_tax_total_000'
    def case(path):
        with path.open() as f:
            return next(row for line in f if (row:=json.loads(line))['case_id']==case_id)
    gold=case(root/'ground_truth.jsonl')
    with (root/'populations.jsonl').open() as f:
        population=next(row for line in f if (row:=json.loads(line))['family']=='tax_total')
    qwen=case(root/'annotations_qwen35_9b.jsonl')['parsed_prediction']
    knn=case(root/'baselines/predictions_gower_knn_subspace.jsonl')
    lof=case(root/'baselines/predictions_gower_lof_subspace.jsonl')
    with (root/'scores_per_case.csv').open() as f:
        scores={row['method']:row for row in csv.DictReader(f) if row['case_id']==case_id}
    predictions=[('Qwen','qwen35_9b',qwen),('Gower-kNN','gower_knn_subspace',knn),
                 ('Gower-LOF','gower_lof_subspace',lof)]
    roles=gold['rule_role_to_feature']
    ids=[roles[r] for r in ['tax_exclusive','tax_amount','tax_inclusive']]
    target=gold['target_row']
    witness=next(row for row in population['clean_role_rows'] if row[2]==target[ids[2]])
    assert target[ids[2]] not in {r[ids[2]] for r in gold['normal_reference_rows']}
    assert len(gold['reference_support_sufficient_minimal_subspaces'])==2
    # Displayed binary endpoint values are read from the frozen score file.
    for _,key,pred in predictions:
        assert json.loads(scores[key]['explanations_predicted'])[0]==sorted(pred['explanations'][0]['features'])
    return gold,population,predictions,scores,ids,witness


def make_framework_figure():
    """Two upper modules and three lower traces, adapted from 图1171.

    Reference geometry: upper modules share a baseline and warm title bands,
    with approximately 35/65 widths; a full-width example occupies the lower
    half. Inputs/gold are at left; repeated method -> candidates -> population
    check -> binary evaluation lanes share columns. Native paper-sized text
    and thin neutral connectors replace the reference's decorative artwork.
    Evaluation is not training: no RL, reward, or model-update edge is copied.
    Qwen remains the supplied asset; Llama/DeepSeek use official-source marks.
    Population labels are isolated in dashed purple evaluator boxes.
    Refinement preserves the approved two-phase/three-lane composition while
    allocating more height to the case. Insets bind eight-field masks, R/P
    gross-value presence, the 4x20 net/gross support grid, and LOF rank hits
    directly to archived rows and predictions. The lattice shows all minimal
    pairs. Field colors encode identity, never explanation correctness.
    """
    gold,population,predictions,scores,ids,witness=invoice_example()
    target=gold['target_row'];refs=gold['normal_reference_rows']
    fig,ax=canvas(7.16,3.82)
    field_colors={ids[0]:'#3C6A9B',ids[1]:'#367E70',ids[2]:'#A86635'}
    def field_set(x,y,fields,size=8.0,rank=None):
        # Live text objects preserve exact field IDs and consistent role colors.
        parts=([(str(rank)+'  ',GRAY)] if rank is not None else [])+[('{',INK)]
        for i,fid in enumerate(fields):
            if i:parts.append((', ',INK))
            parts.append((fid,field_colors.get(fid,INK)))
        parts.append(('}',INK))
        renderer=fig.canvas.get_renderer()
        font=FontProperties(family=plt.rcParams['font.family'],size=size)
        widths=[renderer.get_text_width_height_descent(s,font,False)[0]/fig.dpi for s,_ in parts]
        left=x-sum(widths)/2
        for (s,c),width in zip(parts,widths):
            txt(ax,left,y,s,size,c);left+=width
    gross=target[ids[2]]
    sample_absent=all(row[ids[2]]!=gross for row in refs)
    pair=(target[ids[0]],gross)
    pair_absent=all((row[0],row[2])!=pair for row in population['clean_role_rows'])
    singletons_present=all(any(row[k]==target[ids[k]] for row in population['clean_role_rows']) for k in [0,2])
    assert sample_absent and pair_absent and singletons_present
    # More space goes to evidence graphics; keep the approved two-phase layout,
    # the three method lanes, the original canvas and all logo assets.
    edge='#DEB57C';band='#FFDBA6';neutral='#F8F8F7';line='#727579'
    all_fields=sorted(target)
    clean=population['clean_role_rows']
    gross_domain=sorted({r[2] for r in clean})
    net_domain=sorted({r[0] for r in clean})
    sample_gross={r[ids[2]] for r in refs}
    support_pairs={(r[0],r[2]) for r in clean}
    minimal={frozenset(v) for v in gold['reference_support_sufficient_minimal_subspaces']}
    assert len(all_fields)==8 and len(gross_domain)==20 and len(net_domain)==4

    # Upper left: six tracks as a compact source grid, plus the shared input.
    for x,w,title in [(.04,2.47,'1.  Build benchmark queries'),
                       (2.58,4.54,'2.  Generate candidates and evaluate')]:
        box(ax,x,2.20,w,1.57,'#FFFEFC',edge,.75,.08)
        box(ax,x,3.55,w,.22,band,'none',0,.075)
        txt(ax,x+w/2,3.66,title,9.0,INK,'bold',ha='center')
    sources=[('Original rules','864 targets','table'),
             ('New compositions','48 targets','table'),
             ('Invoices','96 targets','invoice'),
             ('Public corrections','72 targets','database'),
             ('Circuits','86 queries','circuit'),
             ('ADBench / UCI','70 / 8 queries','database')]
    for j,(name,n,kind) in enumerate(sources):
        xx=.15+(j%2)*1.14;yy=3.37-(j//2)*.28
        box(ax,xx,yy-.19,1.09,.25,neutral,'none',0,.025)
        if kind in ['table','invoice']:doc(ax,xx+.04,yy-.07,.105,.12,GRAY)
        elif kind=='circuit':circuit(ax,xx+.095,yy-.01,.13,GRAY)
        else:database(ax,xx+.095,yy-.01,.13,GRAY)
        txt(ax,xx+.195,yy,name,7.5)
        txt(ax,xx+.195,yy-.125,n,7.1,GRAY)
    arrow(ax,1.28,2.62,1.28,2.57,line,.8)
    box(ax,.15,2.27,2.25,.29,'#EDF2F7','#B1BECA',.7,.035)
    for xx in [.90,1.65]:rule(ax,xx,2.30,xx,2.53,'#B1BECA',.55)
    for xx,title,detail in [(.525,'Target x*','Known outlier'),
                             (1.275,'Normal R','Reference rows'),
                             (2.025,'Schema T','Field IDs + types')]:
        txt(ax,xx,2.49,title,8.0,INK,'bold',ha='center')
        txt(ax,xx,2.345,detail,7.1,GRAY,ha='center')

    # Query fans out to LLM prompting and track-specific algorithmic methods.
    rule(ax,2.41,2.415,2.72,2.415,line,.85)
    rule(ax,2.72,2.39,2.72,3.15,line,.85)
    arrow(ax,2.72,3.15,2.85,3.15,line,.85)
    arrow(ax,2.72,2.39,2.85,2.39,line,.85)
    box(ax,2.87,2.79,1.79,.65,neutral,'#AEB5BE',.65,.045,dash=True)
    txt(ax,3.765,3.34,'LLM prompting',8.6,INK,'bold',ha='center')
    for xx,name,version,icon in [(3.18,'Qwen3.5','9B',qwen_logo),
                                (3.76,'Llama3.1','8B-Instruct',llama_logo),
                                (4.36,'DeepSeek','V2-Lite-Chat',deepseek_logo)]:
        icon(ax,xx,3.15,.21 if name.startswith('Qwen') else .25)
        txt(ax,xx,2.99,name,8.0,ha='center')
        txt(ax,xx,2.875,version,7.1,GRAY,ha='center')
    txt(ax,3.765,2.72,'Direct / fixed-schema decoding',7.4,GRAY,ha='center')
    txt(ax,3.765,2.60,'Shared size prior in paired tests',7.3,GRAY,ha='center')
    box(ax,2.87,2.28,1.79,.23,neutral,'#AEB5BE',.65,.045,dash=True)
    txt(ax,3.765,2.395,'Gower-kNN / LOF + other baselines',7.7,INK,'bold',ha='center')
    rule(ax,4.68,3.15,4.83,3.15,line,.8)
    rule(ax,4.68,2.39,4.83,2.39,line,.8)
    rule(ax,4.83,2.39,4.83,3.32,line,.8)
    arrow(ax,4.83,3.32,5.04,3.32,line,.8)
    box(ax,5.06,3.16,1.87,.31,'white','#AEB5BE',.65,.035)
    doc(ax,5.19,3.205,.15,.21,BLUE)
    txt(ax,5.49,3.38,'Ranked field subsets',8.4,INK,'bold')
    txt(ax,5.49,3.235,'or feature rankings',7.6,GRAY)
    arrow(ax,6.00,3.14,6.00,3.09,line,.8)
    box(ax,5.06,2.63,1.87,.44,'#EFF4EE','#A7BDA2',.7,.035)
    txt(ax,5.995,2.975,'Validate and score',8.5,INK,'bold',ha='center')
    txt(ax,5.995,2.83,'Parse · field IDs · abstention',7.5,GRAY,ha='center')
    txt(ax,5.995,2.705,'Exact / hits / agreement + coverage',7.3,GRAY,ha='center')
    box(ax,5.06,2.28,1.87,.28,'#FAF8FC','#AE9BC6',.65,.035,dash=True)
    txt(ax,5.995,2.48,'Evaluator-only labels',7.8,PURPLE,'bold',ha='center')
    txt(ax,5.995,2.345,'Support/rule/edit · cells/faults/peers',7.2,GRAY,ha='center')
    ax.add_patch(FancyArrowPatch((6.71,2.575),(6.71,2.62),arrowstyle='-|>',
        mutation_scale=3.5,color=PURPLE,lw=.75,shrinkA=0,shrinkB=0,zorder=6))

    # Lower panel: expanded case with data-bound masks and support graphics.
    box(ax,.04,.035,7.08,2.095,'#FFFEFC',edge,.75,.08)
    txt(ax,.16,2.06,'Recorded invoice',8.6,INK,'bold')
    txt(ax,2.00,2.06,'8 fields · 32 references R · 20 clean triples P · recorded predictions',7.9,GRAY)
    box(ax,.15,1.08,1.56,.82,'white','#AEB5BE',.6,.035,dash=True)
    for xx,name,fid in zip([.75,1.11,1.48],['Net','Tax','Gross'],ids):
        txt(ax,xx,1.79,name,7.5,field_colors[fid],ha='center')
        txt(ax,xx,1.65,fid,7.2,field_colors[fid],ha='center')
    rule(ax,.20,1.55,1.66,1.55,'#B7BEC5',.5)
    for j,(label,row) in enumerate([('Ref. 1',refs[0]),('Ref. 2',refs[1]),('Target',target)]):
        yy=1.46-j*.15
        if j==2:box(ax,.18,yy-.065,1.50,.13,'#F9E4C8','none',0,.012)
        txt(ax,.23,yy,label,7.4,weight='bold' if j==2 else 'normal')
        for xx,fid in zip([.75,1.11,1.48],ids):
            txt(ax,xx,yy,f'{row[fid]:g}',8.0,field_colors[fid] if j==2 else INK,ha='center')
    txt(ax,.93,.99,'P: net + tax = gross',7.4,PURPLE,ha='center')

    # Subset lattice: supported singletons -> the two minimal excluded pairs.
    box(ax,.15,.25,1.56,.66,'#FAF8FC','#AE9BC6',.6,.035,dash=True)
    txt(ax,.93,.835,'Minimal pairs (absent)',7.8,PURPLE,'bold',ha='center')
    pair_centers=[.54,1.32]
    singletons=[(ids[0],.39),(ids[2],.93),(ids[1],1.47)]
    labelsets=gold['reference_support_sufficient_minimal_subspaces']
    for fields,xx in zip(labelsets,pair_centers):
        for fid,sx in singletons:
            if fid in fields:arrow(ax,sx,.55,xx,.605,GRAY,.65)
        box(ax,xx-.345,.605,.69,.15,'#EDE7F4','#B6A3CD',.55,.025)
        field_set(xx,.68,fields,7.1)
    for fid,xx in singletons:
        box(ax,xx-.175,.40,.35,.15,'white',field_colors[fid],.55,.025)
        txt(ax,xx,.475,fid,7.2,field_colors[fid],ha='center')
    txt(ax,.93,.325,'Singletons supported in P',7.1,GRAY,ha='center')

    for xx,label in [(2.445,'Method'),(3.65,'Subset / mask'),(5.20,'Support / rank audit')]:
        txt(ax,xx,1.93,label,7.8,INK,'bold',ha='center')
    txt(ax,6.38,1.93,'Exact@1',7.6,INK,'bold',ha='center')
    txt(ax,6.84,1.93,'Hit@4',7.6,INK,'bold',ha='center')
    rule(ax,1.73,1.49,1.84,1.49,line,.8)
    lane_y=[1.61,1.035,.46]
    rule(ax,1.84,lane_y[-1],1.84,lane_y[0],line,.8)
    # Shared gross-value axis for the R/P strip and the net-by-gross grid.
    gx0=4.77;dx=.059
    def gx(value):return gx0+gross_domain.index(value)*dx
    support_fill='#7EAAAE'
    for j,(name,key,pred) in enumerate(predictions):
        yy=lane_y[j]
        box(ax,1.95,yy-.25,5.07,.50,'#F7F8F8','none',0,.045)
        arrow(ax,1.84,yy,1.97,yy,line,.8)
        box(ax,1.98,yy-.18,.93,.36,'white','#AEB5BE',.6,.035)
        if key=='qwen35_9b':qwen_logo(ax,2.13,yy+.04,.18)
        else:search(ax,2.13,yy+.04,.16,BLUE)
        txt(ax,2.285,yy+.055,name,7.7,INK)
        txt(ax,2.445,yy-.095,['Table prompting','kNN contrast','LOF density'][j],7.1,GRAY,ha='center')
        arrow(ax,2.93,yy,3.05,yy,line,.8)
        box(ax,3.07,yy-.24,1.16,.48,'white','#AEB5BE',.6,.035,dash=True)
        selected=pred['explanations'][0]['features']
        field_set(3.65,yy+.15,selected,7.9,rank=1)
        # A labelled binary mask is computed from the archived first subset.
        for k,fid in enumerate(all_fields):
            xx=3.175+k*.135;chosen=fid in selected
            face=field_colors.get(fid,GRAY) if chosen else '#F2F4F6'
            box(ax,xx-.058,yy-.075,.116,.12,face,'#CDD4DB',.35,.012)
            txt(ax,xx,yy-.015,str(k+1),7.0,'white' if chosen else GRAY,ha='center')
        if key=='gower_lof_subspace':
            field_set(3.65,yy-.16,pred['explanations'][1]['features'],7.2,rank=2)
        else:
            label=(f'{gross:g} absent from R' if j==0 else 'Both singletons occur in P')
            txt(ax,3.65,yy-.16,label,7.1,GRAY,ha='center')
        arrow(ax,4.25,yy,4.38,yy,line,.8)
        box(ax,4.40,yy-.24,1.60,.48,'white','#BAC8C6',.6,.035)
        if j==0:
            txt(ax,5.20,yy+.175,'(a) Gross-value support',7.5,INK,'bold',ha='center')
            for row_y,source,label in [(yy+.04,sample_gross,'R'),(yy-.075,set(gross_domain),'P')]:
                txt(ax,4.62,row_y,label,7.3,GRAY,ha='right')
                for value in gross_domain:
                    present=value in source
                    face=(field_colors[ids[2]] if value==gross else support_fill) if present else 'white'
                    border=field_colors[ids[2]] if value==gross else '#CCD7DB'
                    box(ax,gx(value)-.026,row_y-.035,.052,.07,face,border,.75 if value==gross else .35,.005)
            for value in [gross_domain[0],gross,gross_domain[-1]]:
                txt(ax,gx(value),yy-.18,str(value),7.1,field_colors[ids[2]] if value==gross else GRAY,ha='center')
            txt(ax,5.42,yy-.18,'Gross',7.0,GRAY,ha='center')
        elif j==1:
            txt(ax,5.20,yy+.175,f'(b) Excluded pair ({pair[0]}, {pair[1]})',7.3,INK,'bold',ha='center')
            # Four net rows by twenty gross columns: every cell is enumerated
            # from P, not a continuous support approximation or a fitted curve.
            for ni,net in enumerate(net_domain):
                ry=yy-.085+ni*.047
                for value in gross_domain:
                    hit=(net,value) in support_pairs
                    is_target=(net,value)==pair
                    box(ax,gx(value)-.026,ry-.021,.052,.042,
                        support_fill if hit else 'white',field_colors[ids[2]] if is_target else '#D2DBDF',
                        .85 if is_target else .3,.001)
                    if is_target:
                        for sign in [-1,1]:rule(ax,gx(value)-.017,ry+sign*.013,gx(value)+.017,ry-sign*.013,field_colors[ids[2]],.65)
                if net in [net_domain[0],net_domain[-1]]:
                    txt(ax,4.68,ry,str(net),7.0,GRAY,ha='right')
            txt(ax,4.48,yy-.015,'Net',7.0,GRAY,ha='center',rotation=90)
            for value in [gross_domain[0],gross,gross_domain[-1]]:
                txt(ax,gx(value),yy-.18,str(value),7.1,field_colors[ids[2]] if value==gross else GRAY,ha='center')
            txt(ax,5.42,yy-.18,'Gross',7.0,GRAY,ha='center')
            ty=yy-.085+net_domain.index(pair[0])*.047
            arrow(ax,5.18,yy+.12,gx(gross)+.015,ty+.024,field_colors[ids[2]],.6)
        else:
            txt(ax,5.20,yy+.175,'(c) Minimality at ranks 1–4',7.5,INK,'bold',ha='center')
            rank_hits=[frozenset(e['features']) in minimal for e in pred['explanations'][:4]]
            assert len(rank_hits)==4
            for rank,(xx,hit) in enumerate(zip([4.64,5.00,5.36,5.72],rank_hits),1):
                txt(ax,xx,yy+.065,str(rank),7.2,GRAY,ha='center')
                status(ax,xx,yy-.055,hit)
            txt(ax,5.20,yy-.18,'Rank 2 recovers a gold pair',7.1,TEAL,ha='center')
        arrow(ax,6.02,yy,6.15,yy,line,.8)
        box(ax,6.17,yy-.16,.83,.32,'white','#AEB5BE',.6,.035)
        rule(ax,6.585,yy-.13,6.585,yy+.13,GRID,.5)
        for xx,col in [(6.31,'support_top1_minimal'),(6.77,'support_any_minimal_hit')]:
            value=int(scores[key][col]=='True');status(ax,xx,yy,value)
            txt(ax,xx+.14,yy,str(value),8.1,TEAL if value else RED,ha='center')
    # An evaluation-only bus is routed below all method and support lanes.
    rule(ax,1.72,.30,1.80,.30,PURPLE,.7,linestyle=(0,(3,2)))
    rule(ax,1.80,.30,1.80,.095,PURPLE,.7,linestyle=(0,(3,2)))
    rule(ax,1.80,.095,7.065,.095,PURPLE,.65,linestyle=(0,(3,2)))
    rule(ax,7.065,.095,7.065,1.61,PURPLE,.65,linestyle=(0,(3,2)))
    for yy in lane_y:
        ax.add_patch(FancyArrowPatch((7.065,yy),(7.005,yy),arrowstyle='-|>',
            mutation_scale=4,color=PURPLE,lw=.65,shrinkA=0,shrinkB=0,zorder=6))
    txt(ax,4.43,.145,'Mask digits 1–8: F001–F008; filled = selected · Support grids: filled = present',7.0,PURPLE,ha='center')
    save(fig,'figure2_overall_framework')


if __name__=='__main__':
    make_intro_figure()
    make_framework_figure()
