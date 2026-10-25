"""Reproducible local RCSC translation build, with original formula/figure crops.

Requirements: Python 3.12+, reportlab, pypdf, Pillow, pdfplumber; pdftoppm.
Run from any directory: python build_rcsc.py --poppler PATH --font PATH
Only writes into 中文翻译/RCSC and 中文翻译/tmp. Never edits source PDFs.
"""
from pathlib import Path
import argparse, hashlib, json, re, subprocess, io
from xml.sax.saxutils import escape
from PIL import Image as PILImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, KeepTogether
from reportlab.lib.pagesizes import A4
from reportlab import rl_config
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen.canvas import Canvas

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'中文翻译'
SAMPLE=OUT/'RCSC'
ORIGINAL=next((ROOT/'01_Multi-View').glob('*Robust Consensus*/paper.pdf'))
SHA='923aabbbcdd7605a235c49cc0bd607731cc5c9ccd5e581d99478bed6369b4d9d'
# page number, bounding box in PDF points (top-origin); all math stays untouched.
CROPS={
 'eq01':(3,(60,608,292,661)), 'eq02':(3,(310,211,543,231)),
 'eq03':(3,(310,346,543,391)), 'eq04':(3,(310,542,543,596)),
 'eq05':(3,(310,655,543,722)), 'eq06':(4,(60,142,292,196)),
 'eq07':(4,(60,321,292,375)), 'eq08':(4,(110,395,292,429)),
 'eq09':(4,(130,482,292,516)), 'eq10':(4,(85,564,292,585)),
 'eq11':(4,(85,589,292,609)), 'eq12':(4,(59,628,292,657)),
 'eq13':(4,(115,658,292,678)), 'eq14':(4,(128,700,292,720)),
 'eq15':(4,(310,94,543,125)), 'eq16':(4,(380,150,543,189)),
 'eq17':(4,(316,380,543,399)), 'lemma01':(4,(307,588,527,624)),
 'eq18':(5,(54,94,292,139)), 'eq19':(5,(64,230,292,263)),
 'eq20':(5,(62,297,292,376)), 'eq21':(5,(60,582,292,688)),
 'eq22':(5,(309,205,543,255)), 'eq23':(5,(320,288,543,312)),
 'eq24':(5,(308,387,543,454)), 'eq25':(5,(310,487,543,526)),
 'eq26':(5,(312,605,543,642)), 'eq27':(5,(345,663,543,687)),
 'eq28':(6,(85,97,292,157)), 'eq29':(6,(54,190,292,228)),
 'eq30':(6,(108,251,292,275)), 'eq31':(6,(88,315,292,339)),
 'eq32':(6,(95,361,292,385)), 'eq33':(6,(63,450,292,488)),
 'eq34':(6,(110,524,292,562)),
 'fig01':(7,(54,66,543,260)), 'tab01':(7,(54,318,543,408)),
 'fig02':(7,(54,416,296,485)), 'fig03':(7,(60,510,286,631)),
 'tab02':(8,(54,88,543,177)), 'tab03':(8,(54,200,543,290)),
}

def font_setup(path,math_path):
    pdfmetrics.registerFont(TTFont('ZH',path,subfontIndex=0))
    pdfmetrics.registerFont(TTFont('Math',math_path,subfontIndex=0))
    pdfmetrics.registerFontFamily('ZH',normal='ZH',bold='ZH',italic='ZH',boldItalic='ZH')

def reference_pages(original, font_path):
    """Preserve all bibliography as selectable vector text, without English body."""
    r=PdfReader(original);pages=[]
    for number in [9,10,11]:
        p=r.pages[number-1]
        buf=io.BytesIO();c=Canvas(buf,pagesize=(612,792))
        c.setFillColor(colors.white)
        # Mask original running title/page number; retain every reference.
        c.rect(0,732,612,60,fill=1,stroke=0)
        c.rect(0,0,612,63,fill=1,stroke=0)
        if number==9:c.rect(45,450,255,275,fill=1,stroke=0)
        c.setFillColor(colors.HexColor('#173c53'));c.setFont('ZH',11)
        c.drawString(55,750,'参考文献（原文书目，完整保留）')
        c.setFont('ZH',8);c.drawRightString(543,38,f'原文参考文献第 {number-8}/3 页')
        c.save();overlay=PdfReader(buf).pages[0];p.merge_page(overlay);pages.append(p)
    return pages

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--poppler',default='pdftoppm');ap.add_argument('--font',default='C:/Windows/Fonts/msyh.ttc');ap.add_argument('--math-font',default='C:/Windows/Fonts/cambria.ttc');args=ap.parse_args()
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()==SHA,'Unexpected original PDF version'
    rl_config.invariant=1
    font_setup(args.font,args.math_font)
    tmp=OUT/'tmp'/'rcsc';tmp.mkdir(parents=True,exist_ok=True)
    assets=SAMPLE/'assets';assets.mkdir(exist_ok=True)
    prefix=tmp/'original'
    if not (tmp/'original-08.png').exists():
        subprocess.run([args.poppler,'-r','240','-f','3','-l','8','-png',str(ORIGINAL),str(prefix)],check=True,capture_output=True)
    cropmeta={}
    for key,(pn,box) in CROPS.items():
        im=PILImage.open(tmp/f'original-{pn:02}.png');sx=im.width/612;sy=im.height/792
        im.crop(tuple(round(v*(sx if i%2==0 else sy)) for i,v in enumerate(box))).save(assets/f'{key}.png')
        cropmeta[key]=dict(source_page=pn,bbox_points=box,path=f'assets/{key}.png')
    (SAMPLE/'assets.json').write_text(json.dumps(cropmeta,indent=2)+'\n',encoding='utf-8')
    style=ParagraphStyle('body',fontName='ZH',fontSize=10.5,leading=17,spaceAfter=8,wordWrap='CJK',textColor=colors.HexColor('#202a31'),splitLongWords=True)
    h1=ParagraphStyle('h1',parent=style,fontSize=19,leading=29,spaceAfter=16,textColor=colors.HexColor('#173c53'))
    h2=ParagraphStyle('h2',parent=style,fontSize=14,leading=23,spaceBefore=13,spaceAfter=9,keepWithNext=True,textColor=colors.HexColor('#173c53'))
    h3=ParagraphStyle('h3',parent=style,fontSize=11.5,leading=19,spaceBefore=10,spaceAfter=7,keepWithNext=True,textColor=colors.HexColor('#173c53'))
    caption=ParagraphStyle('caption',parent=style,fontSize=9,leading=14,textColor=colors.HexColor('#455968'))
    story=[];used=[];paragraphs=0
    def safe(t):
        # Source permits only super/sub markup; escape every other literal.
        t=escape(t)
        for tag in ['super','sub']:t=t.replace('&lt;'+tag+'&gt;','<'+tag+'>').replace('&lt;/'+tag+'&gt;','</'+tag+'>')
        cmap=pdfmetrics.getFont('ZH').face.charToGlyph
        for ch in sorted(set(t)):
            if ord(ch)>127 and ord(ch) not in cmap:
                assert ord(ch) in pdfmetrics.getFont('Math').face.charToGlyph,(ch,hex(ord(ch)))
                t=t.replace(ch,'<font name="Math">'+ch+'</font>')
        return t
    blocks=re.split(r'\n\s*\n',(SAMPLE/'translation.zh.md').read_text(encoding='utf-8').strip())
    skip=set()
    for index,block in enumerate(blocks):
        if index in skip:continue
        if block=='{{references}}':continue
        mat=re.fullmatch(r'\{\{([a-z0-9]+)\}\}',block)
        if mat:
            key=mat.group(1);used.append(key);pn,box=CROPS[key]
            width=(box[2]-box[0])*(1.35 if key.startswith('eq') or key.startswith('lemma') else 1)
            if key in ['fig02','fig03']:width*=1.6
            width=min(width,491)
            im=PILImage.open(assets/f'{key}.png');height=width*im.height/im.width
            flow=Image(str(assets/f'{key}.png'),width=width,height=height,hAlign='CENTER')
            if key.startswith(('fig','tab')):
                cap=blocks[index+1]
                assert re.match(r'^(图|表) [1-3]\.',cap)
                story.append(KeepTogether([flow,Spacer(1,6),Paragraph(safe(cap),caption)]));skip.add(index+1)
            else:
                story.extend([flow,Spacer(1,6)])
            continue
        if block.startswith('### '):st=h3;block=block[4:]
        elif block.startswith('## '):st=h2;block=block[3:]
        elif block.startswith('# '):st=h1;block=block[2:]
        elif re.match(r'^(图|表) [1-3]\.',block):st=caption
        else:st=style
        story.append(Paragraph(safe(block.replace('\n',' ')),st));paragraphs+=1
    assert set(used)==set(CROPS),f'Unused/missing assets: {set(CROPS)^set(used)}'
    def furniture(c,doc):
        c.saveState();w,h=A4;c.setStrokeColor(colors.HexColor('#cbd6dc'));c.line(52,h-42,w-52,h-42)
        c.setFont('ZH',8);c.setFillColor(colors.HexColor('#567080'));c.drawString(52,h-32,'RCSC | 非官方中文全文译文 | CC BY 4.0')
        c.drawRightString(w-52,29,str(doc.page));c.restoreState()
    body=tmp/'body.pdf'
    doc=SimpleDocTemplate(str(body),pagesize=A4,rightMargin=52,leftMargin=52,topMargin=56,bottomMargin=48,title='面向高效多视图子空间聚类的鲁棒共识锚点学习 | 非官方中文全文译文',author='Original: Yalan Qin, Nan Pu, Guorui Feng, Nicu Sebe; translation assisted by Codex')
    doc.build(story,onFirstPage=furniture,onLaterPages=furniture)
    writer=PdfWriter();bodyreader=PdfReader(body)
    for p in bodyreader.pages:writer.add_page(p)
    for p in reference_pages(ORIGINAL,args.font):writer.add_page(p)
    writer.add_metadata({'/Title':'RCSC 非官方中文全文译文','/Author':'Yalan Qin; Nan Pu; Guorui Feng; Nicu Sebe / Codex assisted translation','/Subject':'CC BY 4.0; full Chinese translation; original equations and figures retained'})
    final=SAMPLE/'RCSC_中文全文译文.pdf'
    with final.open('wb') as f:writer.write(f)
    verify=dict(source_sha256=SHA,pdf_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),source_pages=11,body_pages=len(bodyreader.pages),bibliography_pages=3,output_pages=len(writer.pages),assets=len(used),numbered_equations=34,figures=3,tables=3,source_blocks=paragraphs,source_characters=len((SAMPLE/'translation.zh.md').read_text(encoding='utf-8')),omissions=[],unreadable_regions=[],visual_review='pending')
    (SAMPLE/'build_report.json').write_text(json.dumps(verify,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(verify,ensure_ascii=False))

if __name__=='__main__':main()
