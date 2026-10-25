"""Read-only inventory of original repository files; writes only under 中文翻译/."""
from pathlib import Path
import csv, hashlib, json, re, subprocess, os
from datetime import datetime, timezone
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
if os.name=='nt' and not str(ROOT).startswith('\\\\?\\'):
    ROOT=Path('\\\\?\\'+str(ROOT))
OUT = ROOT / '中文翻译'
BASE = 'c9839663157a00ec3160ae1521757d5a4184caba'

def save_csv(name, rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def main():
    prior_rights={};prior_inventory={}
    for filename,target in [('rights_review.csv',prior_rights),('inventory.csv',prior_inventory)]:
        if (OUT/filename).exists():
            with (OUT/filename).open(encoding='utf-8-sig',newline='') as f:
                target.update({r['id']:r for r in csv.DictReader(f)})
    manifest=(ROOT/'论文清单.md').read_text(encoding='utf-8')
    cards=re.split(r'(?=^### P\d{3}\.)',manifest,flags=re.M)[1:]
    inventory=[];rights=[];pdfs=[];cc=[];existing=[]
    # core.quotePath=false preserves non-ASCII paths in git output.
    originals=subprocess.check_output(['git','-c','core.quotePath=false','ls-tree','-r','--name-only',BASE],cwd=ROOT).decode('utf-8').splitlines()
    baseline={}
    for rel in originals:
        rel=rel.strip('"')
        p=ROOT/rel
        if p.suffix.lower()!='.pdf':continue
        sha=digest(p);baseline[rel]=sha
        try:
            r=PdfReader(p);texts=[page.extract_text() or '' for page in r.pages]
            text='\n'.join(texts);pages=len(r.pages);error=''
        except Exception as e:pages='';text='';texts=[];error=str(e)
        row=dict(path=rel,pages=pages,bytes=p.stat().st_size,sha256=sha,error=error)
        pdfs.append(row)
        hits=[]
        for i,t in enumerate(texts):
            for m in re.finditer(r'creative\s*commons|creativecommons|CC\s*BY|permission|copyright|all rights reserved',t,re.I):
                hits.append(dict(page=i+1,excerpt=t[max(0,m.start()-120):m.end()+240].replace('\n',' ')))
        cc.append(dict(path=rel,notices=hits))
        if '中文' in p.name or '译' in p.name:
            existing.append(dict(**row,extracted_chars=len(text),han_chars=len(re.findall(r'[\u4e00-\u9fff]',text)),image_xobjects=sum(len(page.images) for page in r.pages) if not error else '',headings=' | '.join(s.strip() for s in re.findall(r'^.{0,10}(?:摘要|引言|参考文献|附录).{0,35}$',text,re.M))))
        if 'Robust Consensus Anchor' in rel:
            (OUT/'RCSC').mkdir(exist_ok=True)
            (OUT/'RCSC'/'original_extracted.txt').write_text('\n'.join(f'\n--- PAGE {i+1} ---\n{t}' for i,t in enumerate(texts)),encoding='utf-8')
    by_path={r['path']:r for r in pdfs}
    for card in cards:
        mid,title=re.match(r'### (P\d{3})\. ([^\n]+)',card).groups()
        folder=re.search(r'\*\*文件夹位置\*\*[^\n]*<\./([^>]+)>',card).group(1).rstrip('/')
        pdf_match=re.search(r'打开正文 PDF[^\n]*<\./([^>]+)>',card)
        rel=pdf_match.group(1) if pdf_match else ''
        detail=by_path.get(rel,{})
        venue=re.search(r'\*\*期刊／会议\*\*：([^\n]+)',card)
        row=dict(id=mid,title=title,venue=venue.group(1) if venue else '',folder=folder,primary_pdf=rel,pages=detail.get('pages',''),bytes=detail.get('bytes',''),sha256=detail.get('sha256',''),source_status='present' if rel else 'missing',translation_status='rights_pending')
        old=prior_inventory.get(mid,{})
        if old.get('sha256')==row['sha256']:row['translation_status']=old.get('translation_status','rights_pending')
        inventory.append(row)
        notes=[]
        for note in (ROOT/folder).glob('*.md'):
            if '中文' in note.name:continue
            for url in re.findall(r'https?://[^\s<>\])]+',note.read_text(encoding='utf-8',errors='replace')):
                notes.append(url)
        rights.append(dict(id=mid,title=title,source_pdf=rel,source_sha256=detail.get('sha256',''),license='UNVERIFIED',evidence_url='',source_urls=' | '.join(dict.fromkeys(notes)),decision='pending_permission_or_license',figure_review='not_started',reason='Repository presence, download access and code licenses do not establish adaptation/republication rights.'))
        old=prior_rights.get(mid,{})
        if old.get('source_sha256')==rights[-1]['source_sha256']:
            for key in ['license','evidence_url','decision','figure_review','reason']:
                if key in old:rights[-1][key]=old[key]
    save_csv('inventory.csv',inventory);save_csv('all_original_pdfs.csv',pdfs);save_csv('rights_review.csv',rights)
    if existing:save_csv('existing_translations.csv',existing)
    (OUT/'baseline_pdf_sha256.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Some original PDFs have unpaired UTF-16 surrogates; escaped JSON preserves
    # the extracted evidence without making the UTF-8 manifest unwritable.
    (OUT/'license_notices.json').write_text(json.dumps(cc,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    counts=dict(entries=len(inventory),primary_present=sum(x['source_status']=='present' for x in inventory),primary_missing=sum(x['source_status']=='missing' for x in inventory),primary_pages=sum(x['pages'] for x in inventory if isinstance(x['pages'],int)),all_original_pdfs=len(pdfs),all_original_pdf_pages=sum(x['pages'] for x in pdfs if isinstance(x['pages'],int)),pdf_errors=sum(bool(x['error']) for x in pdfs),existing_chinese_pdfs=len(existing))
    progress=dict(baseline_commit=BASE,branch='translations/zh-pdf-audit-20261025',updated_utc=datetime.now(timezone.utc).isoformat(),counts=counts,phase='inventory_complete_rights_review_in_progress',next=['Review license_notices.json manually','Verify RCSC official PDF hash and figures','Audit existing translations','Build one licensed full translation and render all pages'])
    (OUT/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(counts))

if __name__=='__main__':main()
