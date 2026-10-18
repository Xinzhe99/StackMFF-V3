import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import pytest

spec=importlib.util.spec_from_file_location('download_results',Path(__file__).resolve().parents[1]/'tools/download_results.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)

def archive(path, entries):
    with tarfile.open(path,'w') as tf:
        for name,data in entries:
            member=tarfile.TarInfo(name);member.size=len(data);tf.addfile(member,io.BytesIO(data))

def test_reassembles_verified_parts_and_preserves_existing_results(tmp_path):
    source=tmp_path/'source.tar';archive(source,[('results/Method/Set/scene.png',b'example image bytes')])
    data=source.read_bytes();cache=tmp_path/'cache';cache.mkdir()
    parts=[]
    for i,content in enumerate([data[:1000],data[1000:]]):
        name=f'part{i}';(cache/name).write_bytes(content)
        parts.append(dict(name=name,size=len(content),sha256=hashlib.sha256(content).hexdigest(),url='https://invalid.example/unused'))
    record=dict(name='assembled.tar',size=len(data),sha256=hashlib.sha256(data).hexdigest(),parts=parts)
    rebuilt=D.obtain(record,cache)
    dest=tmp_path/'dest';D.extract(rebuilt,dest);D.extract(rebuilt,dest)
    assert (dest/'results/Method/Set/scene.png').read_bytes()==b'example image bytes'
    (dest/'results/Method/Set/scene.png').write_bytes(b'user result')
    with pytest.raises(ValueError,match='Refusing to overwrite'):D.extract(rebuilt,dest)
    assert (dest/'results/Method/Set/scene.png').read_bytes()==b'user result'

def test_rejects_path_escape_before_extracting_any_file(tmp_path):
    source=tmp_path/'bad.tar';archive(source,[('safe.txt',b'safe'),('../escape.txt',b'bad')])
    with pytest.raises(ValueError,match='Unsafe archive member'):D.extract(source,tmp_path/'dest')
    assert not (tmp_path/'escape.txt').exists()
    assert not (tmp_path/'dest/safe.txt').exists()

def test_rejects_corrupt_cached_part(tmp_path):
    (tmp_path/'part').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Cached file differs'):
        D.fetch(dict(name='part',size=8,sha256='0'*64,url='https://invalid.example/unused'),tmp_path)
