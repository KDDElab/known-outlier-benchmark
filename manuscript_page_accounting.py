"""ICDE 2027 accounting for this manuscript's explicit disclosure-page boundary."""
from pathlib import Path
import pymupdf


def page_accounting(path):
    with pymupdf.open(Path(path)) as doc:
        texts = [page.get_text() for page in doc]
        disclosure = [i for i, text in enumerate(texts)
                      if 'ACKNOWLEDGMENTS AND AI USE DISCLOSURE' in text]
        if len(disclosure) != 1:
            raise ValueError('Expected one AI disclosure section after the main-text page break.')
        index = disclosure[0]
        if not texts[index].lstrip().startswith('ACKNOWLEDGMENTS AND AI USE DISCLOSURE'):
            raise ValueError('The AI disclosure must begin a separate page for this accounting.')
        if any('\nAPPENDIX' in text for text in texts):
            raise ValueError('ICDE 2027 does not allow an appendix.')
        return {'total_pages': len(doc), 'main_text_pages': index,
                'disclosure_starts_page': index + 1,
                'excluded_reference_and_disclosure_pages': len(doc) - index,
                'main_text_within_limit': index <= 12,
                'appendix': False}
