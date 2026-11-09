"""Create an unfilled, auditable two-rater packet; never synthesize judgments."""
from pathlib import Path
import json,html
from run_gower_subspace_baseline import extract_case_data
HERE=Path(__file__).resolve().parent;OUT=HERE/'runs/ccfa_acceptance_revision'
def main():
 cases=[json.loads(x) for x in (OUT/'ground_truth_real_cells.jsonl').read_text().splitlines()]
 inputs={r['case_id']:r for r in [json.loads(x) for x in (OUT/'inputs_real_cells.jsonl').read_text().splitlines()]}
 sections=[]
 for c in cases:
  rows=''.join('<tr><td>'+html.escape(f)+'</td><td>'+html.escape(name)+'</td><td>'+html.escape(str(c['original_target'][f]))+'</td></tr>' for f,name in c['feature_names'].items())
  changes=''.join('<li>'+html.escape(r['feature_id']+' / '+r['field_name']+': '+repr(r['original'])+' → '+repr(r['corrected']))+'</li>' for r in c['corrections'])
  fields=list(c['feature_names']);refs=extract_case_data(inputs[c['case_id']]['user_prompt'])['normal_reference_rows']
  reference_header='<tr><th>源行号</th>'+''.join('<th>'+html.escape(f+' / '+c['feature_names'][f])+'</th>' for f in fields)+'</tr>'
  reference_rows=''.join('<tr><td>'+str(index)+'</td>'+''.join('<td>'+html.escape(str(r[f]))+'</td>' for f in fields)+'</tr>' for index,r in zip(c['reference_row_indices'],refs))
  sections.append('<section><h2>'+html.escape(c['case_id'])+'</h2><p>来源：'+html.escape(c['dataset'])+'；原始 CSV 的零起点行号：'+str(c['source_row_index_zero_based'])+'</p><table><tr><th>ID</th><th>字段</th><th>目标值</th></tr>'+rows+'</table><details><summary>查看模型可见的 32 条正常参考（目标的干净副本不在其中）</summary><div style="overflow-x:auto"><table style="min-width:1200px">'+reference_header+reference_rows+'</table></div></details><p>独立填写：可接受的解释/检查字段集合（允许多个）、依据及不确定性。若证据不足，请如实注明。两份独立记录完成后再登记分歧原因。</p><details><summary>独立判断后再查看公开纠正记录</summary><ul>'+changes+'</ul><p>这些记录是公开纠正版本的差异，不是完备最小解释真值。</p></details></section>')
 page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>真实字段案例：待人工审核包</title><style>body{font:16px/1.6 system-ui;max-width:1000px;margin:36px auto;color:#253347}table{border-collapse:collapse;width:100%;table-layout:fixed}td,th{border:1px solid #ccd4df;padding:6px;overflow-wrap:anywhere}section{margin:32px 0;border-top:2px solid #4879bf;padding-top:12px}details{background:#eef3fb;padding:12px}h2{font-size:20px}</style><h1>真实字段案例：待人工审核包</h1><p>本文件没有完成的人工标签。请两位标注者分别复制 human_annotation_blank.csv 独立填写，再由第三步对账保留原始分歧。case_id 与 ground_truth_real_cells.jsonl 一一对应；其参考行索引可在 public_data 中核查。</p><ol><li>先仅查看目标与公开参考行，列出可接受的一个或多个解释集合，说明依据及不确定性。</li><li>独立判断完成后再查看本页折叠区的公开纠正记录；记录任何与公开版本不同的合理解释，不能自动把分歧判为错误。</li><li>保留两份原始标签，另记录协商结果。没有真实填写记录时，不计算或宣称人工一致性。</li><li>若测量检查收益，应另做顺序随机、方法名称隐藏的辅助/无辅助配对任务，并保存实际检查字段及时间。离线 FieldHit、Recall 与检查位置不等于实际节时效果。</li></ol><p>来源：<a href="https://github.com/BigDaMa/raha">Raha/Baran 作者仓库</a>，固定提交 7be1334b8c7bbdac3f47ef514fb3e1e8c5fc181c。原文件、SHA256、许可证和字段对齐说明均在 public_data/provenance.json 及 real_cells_protocol.json。</p>'''+''.join(sections)+'</html>'
 (OUT/'human_annotation_packet.html').write_text(page)
 print('72 unfilled cases and source corrections exported; no human judgments added.')
if __name__=='__main__':main()
