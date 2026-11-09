"""Fixed JSON Schema constrained decoder; unchanged original minimal prompt.

LM Format Enforcer 0.11.3 imports a tokenizer base from its old Transformers
location. The in-process alias below restores that import only; model weights,
tokenizer behavior, loader, generation settings and old outputs are unchanged.
"""
import json,hashlib
from pathlib import Path
import transformers
import transformers.tokenization_utils as tokenization_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
tokenization_utils.PreTrainedTokenizerBase=PreTrainedTokenizerBase
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data,build_transformers_prefix_allowed_tokens_fn
import annotate_v2 as base

CACHE={}
def schema_for(ids):
 return {'type':'object','properties':{
  'explanations':{'type':'array','minItems':0,'maxItems':4,'items':{'type':'object','properties':{
   'features':{'type':'array','minItems':1,'maxItems':len(ids),'items':{'type':'string','enum':sorted(ids)}}},
   'required':['features'],'additionalProperties':False}},
  'abstain':{'type':'boolean'}},'required':['explanations','abstain'],'additionalProperties':False}

def generate_one(model,tokenizer,case):
 # The original generation routine owns input/token/stopping behavior. Wrap
 # model.generate only for this call to add the fixed grammar processor.
 key=id(tokenizer)
 if key not in CACHE:CACHE[key]=build_token_enforcer_tokenizer_data(tokenizer)
 parser=JsonSchemaParser(schema_for(case['feature_ids']))
 prefix=build_transformers_prefix_allowed_tokens_fn(CACHE[key],parser)
 original=model.generate
 def constrained(**kwargs):
  return original(**kwargs,prefix_allowed_tokens_fn=prefix)
 model.generate=constrained
 try:return ORIGINAL_GENERATE(model,tokenizer,case)
 finally:model.generate=original

ORIGINAL_GENERATE=base.generate_one
base.generate_one=generate_one
if __name__=='__main__':base.main()
