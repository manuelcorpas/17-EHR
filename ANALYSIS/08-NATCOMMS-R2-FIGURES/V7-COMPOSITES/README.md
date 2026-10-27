# V7 figure composites (NCOMMS-26-040266-T, round 2), 27 October 2026

Figures inserted into `DOCX/.../R2-REVISION/05-HEIM_NatComms_Manuscript_v7_TRACKED.docx` (Figures 2 to 5, Extended Data Figs 1 to 3; Figure 1 unchanged, its panel c weights still need the design-file edit).

Regenerate:
1. `SCRIPTS/studio_figs.sh` on the Mac Studio (python3.11 has umap/geopandas; embeddings materialised there) writes Fig 3, the Fig 4a UMAP and the Fig 4 b-e panels (relettered) to /tmp/heimfig. Copy them to the scratch `new/` folder.
2. `SCRIPTS/patch.py` makes a copy of `PYTHON/06-13-extended-data-fig1-temporal.py` whose ED Fig 1c annotation reports counts (4 NTD publications of 36,416) instead of 0.0% to 0.0%.
3. `SCRIPTS/compose.py` assembles Fig 2 (a over b|c) and Fig 4 (a over b-e) and copies the single-panel figures.
4. `SCRIPTS/swap_figs.py in.docx out.docx` replaces the images by caption, keeps width and caps height at 9 in.

Checks run: centroids (`heim_centroids.npz`, 175 diseases, 13,158,048 abstracts) reproduce deposited SII (max diff 3e-7); accept and reject text of v7 identical to the 11:05 v6 save; LibreOffice render inspected.
Known limitation kept from the submitted design: Fig 3a colours only the top 15 countries by sites.
