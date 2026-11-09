"""Joint shared candidate information and fixed-schema support localization."""
import hashlib, json, sys
from pathlib import Path
import annotate_acceptance_structured as decoder
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn, build_token_enforcer_tokenizer_data

def generate_one(model,tokenizer,case):
    key=id(tokenizer)
    if key not in decoder.CACHE:
        decoder.CACHE[key]=build_token_enforcer_tokenizer_data(tokenizer)
    schema=decoder.schema_for(case['feature_ids'])
    schema['properties']['explanations']['items']['properties']['features']['maxItems']=case['candidate_size_cap']
    prefix=build_transformers_prefix_allowed_tokens_fn(decoder.CACHE[key],JsonSchemaParser(schema))
    original=model.generate
    def constrained(**kwargs):return original(**kwargs,prefix_allowed_tokens_fn=prefix)
    model.generate=constrained
    try:return decoder.ORIGINAL_GENERATE(model,tokenizer,case)
    finally:model.generate=original

decoder.base.generate_one=generate_one
if __name__=='__main__':
    print(json.dumps({'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'decoder':'LMFE0.11.3','argv':sys.argv}),flush=True)
    decoder.base.main()
