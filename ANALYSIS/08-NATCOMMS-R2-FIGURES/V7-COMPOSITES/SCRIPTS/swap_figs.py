"""Swap figure images in a docx. usage: swap_figs.py in.docx out.docx
Replaces Figures 2-5 and Extended Data Figs 1-3 (by the caption that follows each image), keeps width, sets height by aspect."""
import re, sys, shutil, subprocess, tempfile
from pathlib import Path
from PIL import Image

FIN = Path(__file__).parent / "final"
NEW = {'Figure 2.': 'Fig2.png', 'Figure 3.': 'Fig3.png', 'Figure 4.': 'Fig4.png', 'Figure 5.': 'Fig5.png',
       'Extended Data Fig. 1.': 'ED1.png', 'Extended Data Fig. 2.': 'ED2.png', 'Extended Data Fig. 3.': 'ED3.png'}

src, dst = sys.argv[1], sys.argv[2]
w = Path(tempfile.mkdtemp()); subprocess.run(['unzip', '-q', src, '-d', str(w)], check=True)
doc = (w / 'word/document.xml').read_text(encoding='utf8')
rels_p = w / 'word/_rels/document.xml.rels'; rels = rels_p.read_text(encoding='utf8')
target = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="(media/[^"]+)"', rels))

done = {}
for m in list(re.finditer(r'<w:drawing>.*?</w:drawing>', doc, re.S)):
    rid = re.search(r'r:embed="(rId\d+)"', m.group(0)).group(1)
    after = re.sub(r'<[^>]+>', '', doc[m.end():m.end() + 80000]).lstrip()
    key = next((k for k in NEW if after.startswith(k)), None)
    if key: done[key] = (rid, m.group(0))
missing = set(NEW) - set(done); assert not missing, missing

for key, (rid, drawing) in done.items():
    png = FIN / NEW[key]; iw, ih = Image.open(png).size
    old = target[rid]; new = 'media/r2_' + NEW[key]
    shutil.copy(png, w / 'word' / new)
    rels = re.sub(r'(Id="%s"[^>]*Target=")%s"' % (rid, re.escape(old)), r'\g<1>%s"' % new, rels)
    cx = int(re.search(r'<wp:extent cx="(\d+)"', drawing).group(1)); cy = round(cx * ih / iw)
    MAXH = 8229600  # 9 in usable height on Letter with 1 in margins, as the submitted Fig 4
    if cy > MAXH: cy = MAXH; cx = round(cy * iw / ih)
    nd = re.sub(r'<wp:extent cx="\d+" cy="\d+"', '<wp:extent cx="%d" cy="%d"' % (cx, cy), drawing)
    nd = re.sub(r'<a:ext cx="\d+" cy="\d+"', '<a:ext cx="%d" cy="%d"' % (cx, cy), nd)
    assert doc.count(drawing) == 1; doc = doc.replace(drawing, nd)
    if old not in rels: (w / 'word' / old).unlink()
    print(f'{key:24s} {old} -> {new}  {iw}x{ih}  cx {cx} cy {cy}')

(w / 'word/document.xml').write_text(doc, encoding='utf8'); rels_p.write_text(rels, encoding='utf8')
ct = (w / '[Content_Types].xml').read_text(); assert 'Extension="png"' in ct
out = Path(dst).resolve()
if out.exists(): out.unlink()
subprocess.run(['zip', '-qXrD', str(out), '[Content_Types].xml'], cwd=w, check=True)
subprocess.run(['zip', '-qXrD', str(out), '.', '-x', '[Content_Types].xml'], cwd=w, check=True)
print('wrote', out)
