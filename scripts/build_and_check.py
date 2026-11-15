"""Tectonic + full pdftoppm rendering, with final-pass reference/layout audits."""
from pathlib import Path
import os,subprocess,re,json,xml.etree.ElementTree as ET
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];PAPERS=ROOT.parent/'mestrado-papers';OUT=ROOT/'results/dissertation'
TARGETS=[('latex-dissertacao','main'),('artigo','main'),('latex-problema-pesquisa','main'),('latex-problema-pesquisa','main-resultados')]
ENV=os.environ.copy();ENV['TECTONIC_CACHE_DIR']=str(OUT/'cache/tectonic')

def build(folder,stem):
    directory=PAPERS/folder;log=directory/f'build-{stem}-gbm13_1.txt'
    with log.open('w') as stream:
        result=subprocess.run(['tectonic','-X','compile','--only-cached','--keep-logs','--keep-intermediates',stem+'.tex'],cwd=directory,env=ENV,stdout=stream,stderr=subprocess.STDOUT)
    final=(directory/(stem+'.log')).read_text(errors='replace') if (directory/(stem+'.log')).exists() else ''
    bib=(directory/(stem+'.blg')).read_text(errors='replace') if (directory/(stem+'.blg')).exists() else ''
    overfull=[float(x) for x in re.findall(r'Overfull \\hbox \(([\d.]+)pt too wide\)',final)]
    warnings=[line for line in (final+'\n'+bib).splitlines() if re.search(r'undefined|I didn.t find a database entry|multiply defined',line,re.I)]
    record=dict(document=folder,target=stem,exit_code=result.returncode,undefined_or_duplicate=warnings,overfull_hbox_pt=overfull,max_overfull_pt=max(overfull,default=0))
    with (PAPERS/'V5_1_CHANGELOG.md').open('a') as f:f.write(f'\n- Build `{folder}/{stem}.tex`: retorno {result.returncode}; referências/citações pendentes {len(warnings)}; maior overfull horizontal {max(overfull,default=0):.2f} pt.\n')
    if result.returncode or warnings or any(v>10 for v in overfull):
        (OUT/f'build_{folder}_{stem}.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2));return record
    qa=directory/'qa'/stem;qa.mkdir(parents=True,exist_ok=True)
    for stale in [*qa.glob('page-*.png'),*qa.glob('contact-*.jpg')]:stale.unlink()
    subprocess.run(['pdftoppm','-r','75','-png',str(directory/(stem+'.pdf')),str(qa/'page')],check=True)
    subprocess.run(['pdftotext','-layout',str(directory/(stem+'.pdf')),str(qa/'text.txt')],check=True)
    subprocess.run(['pdftotext','-bbox',str(directory/(stem+'.pdf')),str(qa/'bbox.html')],check=True)
    tree=ET.parse(qa/'bbox.html');bounds=[];pages=tree.findall('.//{http://www.w3.org/1999/xhtml}page')
    for num,page in enumerate(pages,1):
        width=float(page.attrib['width']);height=float(page.attrib['height'])
        for word in page.findall('.//{http://www.w3.org/1999/xhtml}word'):
            b={k:float(word.attrib[k]) for k in ['xMin','xMax','yMin','yMax']}
            if b['xMin']<10 or b['xMax']>width-10 or b['yMin']<5 or b['yMax']>height-5:bounds.append(dict(page=num,text=word.text,**b))
    images=sorted(qa.glob('page-*.png'));record['rendered_pages']=len(images);record['words_outside_safe_page']=bounds
    contacts=[]
    for start in range(0,len(images),9):
        canvas=Image.new('RGB',(1500,2160),'#d6d6d6');draw=ImageDraw.Draw(canvas)
        for pos,p in enumerate(images[start:start+9]):
            im=Image.open(p).convert('RGB');im.thumbnail((480,680));left=(pos%3)*500+(500-im.width)//2;top=(pos//3)*720+25;canvas.paste(im,(left,top));draw.text(((pos%3)*500+15,(pos//3)*720+6),p.stem,fill='black')
        dest=qa/f'contact-{start//9+1:02d}.jpg';canvas.save(dest,quality=88);contacts.append(str(dest))
    record['contact_sheets']=contacts;(OUT/f'build_{folder}_{stem}.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k!='contact_sheets'},indent=2));return record

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--folder');p.add_argument('--stem',default='main');args=p.parse_args()
    targets=[(args.folder,args.stem)] if args.folder else TARGETS
    records=[build(*target) for target in targets];(OUT/'build_summary.json').write_text(json.dumps(records,indent=2))
    if any(r['exit_code'] or r['undefined_or_duplicate'] or r['max_overfull_pt']>10 for r in records):raise SystemExit(1)
