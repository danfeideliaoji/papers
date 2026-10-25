"""Check original immutability and rendered translation content (no network)."""
from pathlib import Path
import hashlib,json,re
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'中文翻译';SAMPLE=OUT/'RCSC'
SHA='923aabbbcdd7605a235c49cc0bd607731cc5c9ccd5e581d99478bed6369b4d9d'

def main():
    original=next((ROOT/'01_Multi-View').glob('*Robust Consensus*/paper.pdf'))
    pdf=SAMPLE/'RCSC_中文全文译文.pdf';r=PdfReader(pdf)
    build=json.loads((SAMPLE/'build_report.json').read_text(encoding='utf-8'))
    texts=[p.extract_text() or '' for p in r.pages]
    # Headers, line wraps, spaces, math and page numbers do not affect Han order.
    body=''.join(re.findall(r'[\u4e00-\u9fff]',re.sub(r'RCSC \| 非官方中文全文译文 \| CC BY 4.0','', '\n'.join(texts[:-3]))))
    blocks=re.split(r'\n\s*\n',(SAMPLE/'translation.zh.md').read_text(encoding='utf-8').strip())
    missing=[]
    for i,b in enumerate(blocks):
        h=''.join(re.findall(r'[\u4e00-\u9fff]',b))
        if h and h not in body:missing.append(dict(block=i,text=b[:100]))
    assets=json.loads((SAMPLE/'assets.json').read_text(encoding='utf-8'))
    assert not missing,missing
    assert set(f'eq{i:02}' for i in range(1,35)).issubset(assets)
    assert hashlib.sha256(original.read_bytes()).hexdigest()==SHA
    assert all((SAMPLE/a['path']).is_file() for a in assets.values())
    assert len(r.pages)==build['output_pages'] and len(r.pages)>=14
    assert all('\ufffd' not in t for t in texts)
    # All original reference glyphs remain on the 3 vector pages. Original body
    # is visually masked, not removed from the PDF text stream; check that every
    # original reference-page text is retained in those page streams.
    orig=PdfReader(original)
    refs_unchanged=[]
    for i in [8,9,10]:
        before=re.sub(r'\s+','',orig.pages[i].extract_text() or '')
        after=re.sub(r'\s+','',texts[i-8+len(texts)-3])
        refs_unchanged.append(before in after)
    assert all(refs_unchanged)
    fonts=[]
    for page in r.pages[:build['body_pages']]:
        for font in page['/Resources']['/Font'].get_object().values():
            f=font.get_object();name=str(f.get('/BaseFont',''))
            if 'MicrosoftYaHei' in name or 'Cambria' in name:
                desc=f['/FontDescriptor'].get_object();assert '/FontFile2' in desc or '/FontFile3' in desc
                fonts.append(name)
    report=dict(pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),pages=len(r.pages),source_han_blocks_checked=sum(bool(re.search(r'[\u4e00-\u9fff]',b)) for b in blocks),missing_han_blocks=missing,formula_assets=34,other_assets=7,all_assets_present=True,all_reference_page_text_streams_preserved=all(refs_unchanged),chinese_and_math_fonts_embedded=sorted(set(fonts)),original_unchanged=True,extraction_replacement_characters=0,visual_review_required=True)
    (SAMPLE/'qa_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
