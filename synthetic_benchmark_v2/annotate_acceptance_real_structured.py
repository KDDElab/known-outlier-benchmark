"""Fixed singleton schema for the paired public-correction condition."""
import annotate_acceptance_structured as structured

ORIGINAL_SCHEMA=structured.schema_for
def singleton_schema(ids):
    schema=ORIGINAL_SCHEMA(ids)
    schema['properties']['explanations']['items']['properties']['features']['maxItems']=1
    return schema

structured.schema_for=singleton_schema
if __name__=='__main__':
    structured.base.main()
