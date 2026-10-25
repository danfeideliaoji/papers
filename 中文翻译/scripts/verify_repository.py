"""Verify every baseline PDF and check that only 中文翻译/ has changed."""
from pathlib import Path
import hashlib,json,subprocess,os,csv

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'中文翻译'

def main():
    baseline=json.loads((OUT/'baseline_pdf_sha256.json').read_text(encoding='utf-8'));changed=[]
    for rel,sha in baseline.items():
        p=ROOT/rel
        if os.name=='nt':p=Path('\\\\?\\'+str(p))
        if hashlib.sha256(p.read_bytes()).hexdigest()!=sha:changed.append(rel)
    tracked=subprocess.check_output(['git','-c','core.quotePath=false','diff','--name-only','c9839663157a00ec3160ae1521757d5a4184caba'],cwd=ROOT).decode('utf-8').splitlines()
    outside=[p for p in tracked if not p.startswith('中文翻译/')]
    assert not changed,changed
    assert not outside,outside
    with (OUT/'inventory.csv').open(encoding='utf-8-sig',newline='') as f:rs=list(csv.DictReader(f))
    assert len(rs)==174 and len({r['id'] for r in rs})==174
    assert sum(r['source_status']=='present' for r in rs)==138
    report=dict(baseline_commit='c9839663157a00ec3160ae1521757d5a4184caba',baseline_pdfs_checked=len(baseline),changed_original_pdfs=changed,tracked_changes_outside_translation_directory=outside,manifest_entries=174,primary_present=138,primary_missing=36)
    (OUT/'repository_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
