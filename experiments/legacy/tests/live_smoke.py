"""PX1 live checks. --reader makes up to three billed vision-reader calls.

Run matching services on index-px1-regions first. All output goes to eval/results.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))


def run(reader=False, resume_reader=False):
    import corpus
    import encoder_contract
    import index_manifest
    import queryembed
    import rag
    expected = encoder_contract.identity(corpus.LAYOUT)
    health = requests.get(rag.SEARCH_API+'/health', timeout=3).json()
    assert health['index_id'] == expected['index_id']
    encoder_contract.validate(requests.get(queryembed.ENCODER_URL+'/health',timeout=3).json(),expected)
    def text_vector():
        response=requests.post(queryembed.ENCODER_URL+'/embed',json={'text':'mounting holes in a garage door frame'},timeout=60)
        response.raise_for_status()
        return np.asarray(response.json()['embedding'])
    before=text_vector()
    with Image.open(corpus.page_path(0,21)) as image:
        w,h=image.size
        crop=image.crop((round(.76*w),round(.42*h),round(.95*w),round(.53*h)))
        buffer=io.BytesIO()
        crop.save(buffer,format='PNG')
        reference=buffer.getvalue()
    pages, _, stats=rag.retrieve_for('Znajdź pokazany detal montażowy w dokumentacji.','visual',n_pages=4,query_image=reference)
    after=text_vector()
    cosine=float(before@after/(np.linalg.norm(before)*np.linalg.norm(after)))
    assert np.isfinite(after).all() and cosine>.9999
    assert 22 in [p.page for p in pages]
    assert stats['jev'] is None
    result={'index_id':expected['index_id'],'index_fingerprint':index_manifest.fingerprint(corpus.LAYOUT.index_dir),
            'screenshot':{'gold_page':22,'returned_pages':[p.page for p in pages],'stats':stats},
            'text_cosine_after_image':cosine,'reader_executed':reader}
    output=Path('eval/results/live-smoke.json')
    output.parent.mkdir(parents=True,exist_ok=True)

    def save():
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

    if reader:
        import citeparse
        import providers
        source=corpus.source_id(0)
        if resume_reader:
            prior=json.loads(output.read_text(encoding='utf-8'))
            assert prior['index_fingerprint']==result['index_fingerprint']
            assert prior['reader_provider']=='gemini' and prior['reader_model']=='gemini-3.6-flash'
            assert prior['reader_calls'] <= 3 and 'price_with_zoom' in prior
            result.update(prior)
            result.pop('reader_error',None)
            price=result['price_with_zoom']
            # Re-run only local citation resolution after a parser fix. Retain
            # the original result and consume no additional reader call.
            events=price['trace']
            reset=max((i for i,e in enumerate(events) if e['type']=='answer_reset'),default=-1)
            raw=''.join(e['text'] for e in events[reset+1:] if e['type']=='answer_delta')
            body,cites=citeparse.split_citations(raw)
            assert body==price['answer'] and cites
            pages_seen=[e for e in events if e['type']=='tile']
            price.setdefault('citations_before_local_replay',price['citations'])
            price['citations']=citeparse.resolve_citations(body,cites,pages_seen)
            price['citation_resolution_replayed_locally']=True
        else:
            result['reader_calls']=0
        result['reader_provider']='gemini'
        result['reader_model']='gemini-3.6-flash'

        class BudgetedReader(providers.GeminiReader):
            def read_pages(self, *args, **kwargs):
                if result['reader_calls'] >= 3:
                    raise RuntimeError('Approved maximum of three reader calls reached')
                assert all(p['source_id']==source and p['page']==36 for p in kwargs['pages'])
                assert not any(p.get('query_image') for p in kwargs['pages'])
                result['reader_calls'] += 1
                save()
                print(f"Reader call {result['reader_calls']}/3: {self.name}/{self.model}",flush=True)
                return super().read_pages(*args,**kwargs)

        actual_reader=BudgetedReader(model=result['reader_model'])
        task='For this integration check, your first response must request exactly one zoom, with no answer prose: ---ZOOM---\n'+json.dumps({'source_id':source,'page':36,'rect':{'x':6,'y':15,'w':88,'h':14}})
        try:
            with patch.object(providers,'reader_for',return_value=actual_reader):
                price=result.get('price_with_zoom')
                if price is None:
                    price=rag.run_agent('Ile wynosi cena netto w PLN bramy UniPro ręczna, woodgrain RAL 9016, So=3000 mm i Ho=2000 mm?',
                                        provider='gemini',retrieval='visual',mode='oneshot',pages_override=[(0,36)],n_pages=1,cache=False,task_instruction=task)
                result['price_with_zoom']=price
                save()
                assert price['steps']==2 and sum(e['type']=='zoom' for e in price['trace'])==1
                assert '4349' in price['answer'].replace(' ','').replace('\u00a0','')
                assert price['citations'] and all(c.get('source_id')==source and c['status']!='unsupported' for c in price['citations'])
                blank=result.get('blank_cell')
                if blank is None:
                    blank=rag.run_agent('Czy tabela UniPro ręczna, woodgrain RAL 9016, podaje cenę dla So=6000 mm i Ho=2500 mm? Nie przenoś ceny z sąsiedniej kolumny.',
                                        provider='gemini',retrieval='visual',mode='oneshot',pages_override=[(0,36)],n_pages=1,cache=False,
                                        task_instruction='Answer from the attached page without requesting a zoom; abstain if it is insufficient.')
                result['blank_cell']=blank
                result['blank_cell_manual_review_required']=True
        except Exception as error:
            result['reader_error']=f'{type(error).__name__}: {error}'
            raise
        finally:
            save()
    save()
    print(json.dumps({'saved':str(output),'screenshot_pages':result['screenshot']['returned_pages'],'text_cosine_after_image':cosine,
                      'reader_executed':reader,'reader_answers':{k:result[k]['answer'] for k in ('price_with_zoom','blank_cell') if k in result}},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reader',action='store_true')
    parser.add_argument('--resume-reader',action='store_true',help='reuse the saved reader calls and remaining three-call budget')
    args=parser.parse_args()
    if args.resume_reader and not args.reader:
        parser.error('--resume-reader requires --reader')
    run(args.reader,args.resume_reader)
