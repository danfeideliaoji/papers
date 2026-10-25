"""Download only known official publisher URLs; compare versions, save evidence.

This script uses no credentials and never uploads a paper. Network access required.
It does not grant publication clearance: figure exceptions still require review.
"""
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlparse,urljoin
from html import unescape
import concurrent.futures,csv,hashlib,json,re
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'中文翻译'
HOSTS={'proceedings.mlr.press','raw.githubusercontent.com'}

def get(url):
    if url.startswith('http:'):url='https:'+url[5:]
    assert urlparse(url).hostname in HOSTS,url
    with urlopen(Request(url,headers={'User-Agent':'papers-translation-license-audit/1.0'}),timeout=60) as r:
        assert urlparse(r.url).hostname in HOSTS,r.url
        return r.read()

def normalize(t):return re.sub('[^a-z0-9]','',t.lower())

def main():
    mapping=json.loads((OUT/'evidence/pmlr_sources.json').read_text(encoding='utf-8'))
    with (OUT/'inventory.csv').open(encoding='utf-8-sig',newline='') as f:inventory={x['id']:x for x in csv.DictReader(f)}
    tmp=OUT/'tmp/official_sources';tmp.mkdir(parents=True,exist_ok=True)
    def one(item):
        pid,url=item;rec=dict(id=pid,article_url=url,checked_utc=datetime.now(timezone.utc).isoformat())
        try:
            raw=get(url);html=raw.decode('utf-8');rec['html_sha256']=hashlib.sha256(raw).hexdigest()
            rec['license_agreement_linked_from_article']='pmlr-license-agreement.html' in html
            metas={}
            for tag in re.findall(r'<meta\b[^>]*>',html,re.I):
                pairs=dict(re.findall(r'([\w_-]+)\s*=\s*["\']([^"\']*)["\']',tag))
                if 'name' in pairs and 'content' in pairs:metas[pairs['name']]=unescape(pairs['content'])
            title=metas.get('citation_title','');rec['official_title']=title
            rec['title_matches']=normalize(title)==normalize(inventory[pid]['title'])
            assert rec['title_matches'],(inventory[pid]['title'],title)
            pdfurl=metas.get('citation_pdf_url')
            if not pdfurl:
                found=re.search(r'href=["\']([^"\']+\.pdf)["\'][^>]*>\s*Download PDF',html,re.I)
                assert found,'PDF link not found';pdfurl=urljoin(url,found.group(1))
            if pdfurl.startswith('http:'):pdfurl='https:'+pdfurl[5:]
            rec['pdf_url']=pdfurl
            dest=tmp/f'{pid}.pdf'
            if not dest.exists():dest.write_bytes(get(pdfurl))
            pdf=dest.read_bytes();assert pdf.startswith(b'%PDF-'),'Not a PDF'
            rec['official_pdf_sha256']=hashlib.sha256(pdf).hexdigest();rec['official_pdf_bytes']=len(pdf)
            rec['repository_pdf_sha256']=inventory[pid]['sha256']
            rec['byte_identical_to_repository']=rec['official_pdf_sha256']==rec['repository_pdf_sha256']
            rec['result']='matched_official_version' if rec['byte_identical_to_repository'] else ('official_available_repository_missing' if not inventory[pid]['primary_pdf'] else 'different_version_requires_review')
            rec['license_evidence']='https://proceedings.mlr.press/pmlr-license-agreement.html'
            rec['figure_exceptions']='not_reviewed' if pid!='P004' else 'reviewed_no_exception_notices_found'
        except Exception as e:rec['result']='verification_error';rec['error']=str(e)
        return rec
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(one,mapping.items()))
    (OUT/'evidence/official_version_checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ['id','result'] if k in r}|({'error':r['error']} if 'error' in r else {}) for r in results],ensure_ascii=False))

if __name__=='__main__':main()
