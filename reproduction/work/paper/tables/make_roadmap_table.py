"""Rebuild the cumulative-roadmap table with the same row selection as its figure."""
from pathlib import Path
import os,runpy
WORK=Path(__file__).resolve().parents[2]
old=os.environ.get('STACKMFF_TABLE_ONLY')
os.environ['STACKMFF_TABLE_ONLY']='1'
try:
 runpy.run_path(str(WORK/'figures/src/make_roadmap.py'),run_name='__main__')
 generated=WORK/'figures/out/tab_roadmap_regen.tex'
 with open(generated,encoding='utf8') as source:
  text=source.read()
 with open(Path(__file__).resolve().parent/'tab_roadmap.tex','w',encoding='utf8') as target:
  target.write(text)
finally:
 if old is None:os.environ.pop('STACKMFF_TABLE_ONLY',None)
 else:os.environ['STACKMFF_TABLE_ONLY']=old
