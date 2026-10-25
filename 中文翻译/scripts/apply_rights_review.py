"""Apply the first manually reviewed publisher evidence; never modify originals."""
from pathlib import Path
import csv,json
from collections import Counter
from datetime import datetime,timezone

OUT=Path(__file__).resolve().parents[1]

def read(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    rows=read('rights_review.csv');inv=read('inventory.csv');byid={r['id']:r for r in rows}
    versions=json.loads((OUT/'evidence/official_version_checks.json').read_text(encoding='utf-8'))
    for v in versions:
        if v['result']=='verification_error':continue
        r=byid[v['id']]
        r.update(license='CC BY 4.0 (PMLR publication agreement; official version)',evidence_url=v['article_url']+' | '+v['license_evidence'],figure_review='pending_per_figure_review')
        if v['byte_identical_to_repository']:
            r.update(decision='license_verified_figure_review_pending',reason='Title and SHA-256 match official PMLR PDF; publication agreement clauses 2-3 permit attribution-compliant adaptation. Check figure credit-line exceptions before translation release.')
        else:
            r.update(decision='use_official_version_after_comparison',reason='Official PMLR version is available under publisher publication agreement, but repository PDF hash differs. Do not transfer that license to the different repository version; compare and translate the official version in the Chinese directory. No original files replaced.')
    byid['P004'].update(decision='translated_sample_qa_passed',figure_review='figures_1_3_tables_1_3_reviewed_no_exception_notices',reason='Official PDF byte-identical; CC BY 4.0 PMLR agreement verified; attribution, source, license and unofficial-translation/modification notice included. All 16 rendered pages checked; 34 equations, 3 figures, 3 tables and all references retained.')
    byid['P131'].update(license='CC BY 4.0',evidence_url='https://link.springer.com/article/10.1007/s00362-023-01468-3',decision='license_verified_figure_review_pending',figure_review='pending_per_figure_review',reason='Article-specific publisher rights section and repository PDF p.23 explicitly allow adaptation and distribution under CC BY 4.0. Figure credit-line exceptions still need review.')
    nd={'P141':('https://doi.org/10.1038/s41467-026-77212-w',11),'P175':('https://link.springer.com/article/10.1007/s44443-026-01262-9',18),'P123':('https://doi.org/10.1007/s44443-026-00875-4',17),'P149':('https://doi.org/10.1007/s00362-024-01550-4',31)}
    for pid,(url,page) in nd.items():
        byid[pid].update(license='CC BY-NC-ND 4.0',evidence_url=url,decision='adaptation_permission_required',figure_review='not_cleared_for_adaptation',reason=f'Repository PDF p.{page} explicitly states CC BY-NC-ND 4.0 and that sharing adapted material is not permitted. Need separate permission for translation and public distribution; downloadable/open access is insufficient.')
    for r in inv:
        rr=byid[r['id']]
        if r['venue']=='AAAI' and rr['license']=='UNVERIFIED':
            rr.update(license='UNVERIFIED; AAAI republication permission rule applies unless separately licensed',evidence_url='https://ojs.aaai.org/index.php/AAAI/about',reason='AAAI publisher policy requires permission for republication. No article-specific adaptation/republication permission verified for this repository copy; check an explicitly open-licensed author version or obtain permission.')
        if r['id']=='P004':r['translation_status']='complete_sample_qa_passed'
        elif rr['decision']=='license_verified_figure_review_pending':r['translation_status']='licensed_queue_figure_review_pending'
        elif rr['decision']=='use_official_version_after_comparison':r['translation_status']='official_version_comparison_pending'
        elif rr['decision']=='adaptation_permission_required':r['translation_status']='blocked_nd_permission_required'
        elif r['source_status']=='missing':r['translation_status']='missing_source_and_rights_pending'
    save('rights_review.csv',rows);save('inventory.csv',inv)
    p=json.loads((OUT/'progress.json').read_text(encoding='utf-8'))
    p.update(updated_utc=datetime.now(timezone.utc).isoformat(),phase='first_sample_complete_ready_for_checkpoint',rights_counts=dict(Counter(r['decision'] for r in rows)),translation_counts=dict(Counter(r['translation_status'] for r in inv)),completed_samples=['P004'],next=['Commit and push first checkpoint; create draft PR; verify remote downloadable PDF','Review figures and supplementary scope for licensed queue','Compare P065/P082 to official PMLR versions','Continue article-specific licensing review for unverified entries; request only necessary permissions'])
    (OUT/'progress.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(p['rights_counts']))

if __name__=='__main__':main()
