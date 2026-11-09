"""Check the completed inference chronology and record portable evidence hashes."""
from datetime import datetime
from pathlib import Path
import collections,hashlib,json
from prepare_unified_revision import OUT,HERE,read

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    frozen=datetime.fromisoformat(protocol['frozen_at_utc'])
    inputs={c['case_id']:c for c in read(OUT/'inputs.jsonl')}
    assert json.loads((OUT/'analysis_manifest.json').read_text())['complete']
    cohorts=[]
    for model in protocol['models']:
        path=OUT/f'annotations_{model}.jsonl';rows=read(path)
        starts=[r for r in read(OUT/f'model_runs_{model}.jsonl') if r.get('event')=='started']
        assert starts and len(rows)==360 and len({r['case_id'] for r in rows})==360
        assert all(datetime.fromisoformat(r['started_at_utc'])>=frozen and r['inputs_sha256']==sha(OUT/'inputs.jsonl') for r in starts)
        for r in rows:
            assert r['prompt_sha256']==inputs[r['case_id']]['prompt_sha256']
            assert datetime.fromisoformat(r['timestamp_utc'])>=frozen
            assert r['do_sample'] is False and r['max_input_tokens']==16384 and r['max_new_tokens']==512
            assert r['protocol_id']==protocol['protocol_id']
        cohorts.append({'model':model,'query_records':len(rows),'actual_generations':sum(bool(r['raw_output']) for r in rows),
            'inference_or_parse_errors':dict(collections.Counter(r.get('parse_or_inference_error') or 'none' for r in rows)),
            'first_record_utc':min(r['timestamp_utc'] for r in rows),'last_record_utc':max(r['timestamp_utc'] for r in rows),
            'responses_sha256':sha(path)})
    result={'complete':True,'frozen_at_utc':protocol['frozen_at_utc'],
        'query_records':sum(x['query_records'] for x in cohorts),
        'actual_generations':sum(x['actual_generations'] for x in cohorts),'cohorts':cohorts,
        'current_code_sha256':{p.name:sha(p) for p in sorted(HERE.glob('*unified*.py'))},
        'inputs_sha256':sha(OUT/'inputs.jsonl'),'effective_population_labels_sha256':sha(OUT/'ground_truth.jsonl'),
        'label_mapping_correction':'corrections/population_label_mapping/correction.json',
        'independent_machine_replication':False}
    (OUT/'execution_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'complete':True,'query_records':result['query_records'],'actual_generations':result['actual_generations']}))

if __name__=='__main__':main()
