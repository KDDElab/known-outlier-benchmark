"""Inspect actual chat rendering, examples, token budgets and stop settings."""
from pathlib import Path
import json,hashlib
import annotate_v2 as base
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent;OUT=HERE/'runs/ccfa_acceptance_revision'
def read(p):return [json.loads(x) for x in p.open() if x.strip()]
def main():
 panel=read(HERE/'runs/ccfa_revision/inputs_panel.jsonl')
 batches={'original_full':panel,**{b:read(OUT/f'inputs_{b}.jsonl') for b in ['prior3','structured','new_rules','real_cells']}}
 meta=[];records=[]
 for model in ['qwen35_9b','deepseek_v2_lite_chat','llama31_8b_instruct']:
  path=Path(base.ANNOTATORS[model]['model_path'])
  hint=json.loads((path/'config.json').read_text())
  tok=AutoTokenizer.from_pretrained(path,trust_remote_code=hint.get('model_type')!='deepseek_v2',local_files_only=True)
  if tok.pad_token_id is None:tok.pad_token=tok.eos_token
  cfg=json.loads((path/'generation_config.json').read_text()) if (path/'generation_config.json').exists() else {}
  for batch,cases in batches.items():
   for i,c in enumerate(cases):
    messages=[{'role':'system','content':c['system_prompt']},{'role':'user','content':c['user_prompt']}]
    try:
     direct=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False);branch='enable_thinking_false'
    except Exception:
     direct=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);branch='default_chat_template'
    rendered=base.render_prompt(tok,c['system_prompt'],c['user_prompt']);assert direct==rendered
    n=len(tok(rendered,truncation=False)['input_ids'])
    r={'model':model,'batch':batch,'case_id':c['case_id'],'paired_case_id':c.get('paired_case_id',c['case_id']),
     'prompt_tokens':n,'input_feasible':n<=16384,'render_branch':branch,'render_sha256':hashlib.sha256(rendered.encode()).hexdigest()};records.append(r)
    if i==0 and batch=='structured':
     example=c['user_prompt'].split('Example shape: ',1)[1].split('.\n',1)[0]
     decoded,_=json.JSONDecoder().raw_decode(example);assert example in rendered
     (OUT/f'rendered_minimal_example_{model}.txt').write_text(rendered)
     meta.append({'model':model,'tokenizer_class':type(tok).__name__,'eos_token_id_used':tok.eos_token_id,'eos_token':tok.eos_token,
      'pad_token_id':tok.pad_token_id,'generation_config_stop_strings':cfg.get('stop_strings'),'generation_config_max_time':cfg.get('max_time'),
      'example_valid_json':True,'example_survives_chat_render':True,'chat_template_sha256':hashlib.sha256(str(tok.chat_template).encode()).hexdigest(),
      'generation':'Original runner passes tokenizer EOS, no custom stopping criteria, greedy512; archived failed minimal outputs end before512. Actual EOS token IDs were not saved in those old records.'})
  print(model,'token budget audit complete',flush=True)
 (OUT/'prompt_token_audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
 (OUT/'chat_template_audit.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
