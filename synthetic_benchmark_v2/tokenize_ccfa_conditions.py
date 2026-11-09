#!/usr/bin/env python3
"""Determine input feasibility from tokenizers before using outcome information."""
import json
import pandas as pd
from transformers import AutoTokenizer
from annotate_v2 import ANNOTATORS, render_prompt
from prepare_ccfa_revision import OUT, MODELS, read, sha

rows=[]
for model in MODELS:
    spec=ANNOTATORS[model]
    tokenizer=AutoTokenizer.from_pretrained(spec['model_path'],local_files_only=True,
                                            trust_remote_code=model!='deepseek_v2_lite_chat')
    for c in read(OUT/'inputs_all_conditions.jsonl'):
        rendered=render_prompt(tokenizer,c['system_prompt'],c['user_prompt'])
        length=len(tokenizer(rendered,truncation=False)['input_ids'])
        rows.append({'method':model,'case_id':c['case_id'],'source_case_id':c['source_case_id'],
                     'generator_seed':c['generator_seed'],'condition':c['condition'],
                     'prompt_sha256':c['prompt_sha256'],'prompt_tokens':length,'input_feasible':length<=16384})
    print(model,'tokenization completed',flush=True)
pd.DataFrame(rows).to_csv(OUT/'token_lengths_all_conditions.csv',index=False)
