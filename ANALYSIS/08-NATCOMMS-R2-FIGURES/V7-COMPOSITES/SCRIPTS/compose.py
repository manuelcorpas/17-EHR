"""Assemble Figure 2 (a over b|c) and Figure 4 (UMAP a over b-e) from regenerated panels, with panel letters."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

S = Path(__file__).parent
F = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/PUBLICATIONS/07-EHR-LINKED-BIOBANKS/ANALYSIS/08-NATCOMMS-R2-FIGURES"
OUT = S / "final"; OUT.mkdir(exist_ok=True)
FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 120)


def scale_w(im, w):
    return im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)


def letter(canvas, xy, s):
    ImageDraw.Draw(canvas).text(xy, s, fill="black", font=FONT)


def stack(rows, gap=80):
    W = max(sum(i.width for i in r) for r in rows)
    H = sum(max(i.height for i in r) for r in rows) + gap * (len(rows) - 1)
    c = Image.new("RGB", (W, H), "white"); y = 0; pos = []
    for r in rows:
        x = (W - sum(i.width for i in r)) // 2
        for i in r:
            c.paste(i, (x, y)); pos.append((x, y)); x += i.width
        y += max(i.height for i in r) + gap
    return c, pos


# Figure 2: map on top, distribution and pie below at a common height
a = Image.open(F / "fig2a_world_biobank_equity.png").convert("RGB")
b = Image.open(F / "fig2b_biobank_eas_distribution.png").convert("RGB")
c = Image.open(F / "fig2c_publication_share.png").convert("RGB")
W = 6000
a = scale_w(a, W)
h = 2400
b = b.resize((round(b.width * h / b.height), h), Image.LANCZOS)
c = c.resize((round(c.width * h / c.height), h), Image.LANCZOS)
k = W / (b.width + c.width)
b, c = scale_w(b, round(b.width * k)), scale_w(c, round(c.width * k))
fig2, pos = stack([[a], [b, c]])
for (x, y), s in zip(pos, "abc"): letter(fig2, (x + 10, y), s)
fig2.save(OUT / "Fig2.png", dpi=(300, 300))

# Figure 4: UMAP on top (letter a added), b-e panels below at the same width
u = Image.open(S / "new/Fig4a.png").convert("RGB")
be = Image.open(S / "new/05_Fig5_Semantic_Structure_Analysis.png").convert("RGB")
W = be.width
u = scale_w(u, W)
pad = Image.new("RGB", (u.width, u.height + 160), "white"); pad.paste(u, (0, 160)); u = pad
fig4, pos = stack([[u], [be]], gap=120)
letter(fig4, (pos[0][0] + 10, pos[0][1]), "a")
fig4.save(OUT / "Fig4.png", dpi=(300, 300))

for name, src in [("Fig3.png", S / "new/Fig3.png"), ("Fig5.png", F / "fig6_unified_score_ranking.png"),
                  ("ED1.png", S / "ed1/Extended_Data_Fig_01_Temporal_Trends.png"),
                  ("ED2.png", F / "Extended_Data_Fig_02_Sensitivity_Analysis.png"),
                  ("ED3.png", F / "fig_regional_comparison.png")]:
    Image.open(src).convert("RGB").save(OUT / name, dpi=(300, 300))
for p in sorted(OUT.glob("*.png")):
    im = Image.open(p); print(p.name, im.size)
