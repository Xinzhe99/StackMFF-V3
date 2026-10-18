"""Download selected verified research archives, resume transfers and extract safely."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
from urllib.request import Request, urlopen
from urllib.error import URLError

ROOT=Path(__file__).resolve().parents[1]
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def fetch(record, cache):
    name=record['name']
    if Path(name).name!=name or '/' in name or '\\' in name:raise ValueError('Unsafe asset name')
    dest=cache/name
    if dest.exists():
        if dest.stat().st_size==record['size'] and sha256(dest)==record['sha256']:return dest
        raise ValueError('Cached file differs from the published checksum: '+str(dest))
    partial=dest.with_name(dest.name+'.partial')
    for attempt in range(6):
        start=partial.stat().st_size if partial.exists() else 0
        if start==record['size']:break
        if start>record['size']:raise ValueError('Oversized partial file: '+str(partial))
        request=Request(record['url'],headers={'Range':f'bytes={start}-'} if start else {})
        try:
            with urlopen(request,timeout=45) as response:
                if start and response.status!=206:start=0
                if start and not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                    raise ValueError('Server returned an unexpected download range')
                with partial.open('ab' if start else 'wb') as f:shutil.copyfileobj(response,f,1024*1024)
            break
        except (OSError,URLError):
            if attempt==5:raise
            time.sleep(2+attempt*2)
    if partial.stat().st_size!=record['size'] or sha256(partial)!=record['sha256']:
        raise ValueError('Download checksum failed: '+name)
    partial.replace(dest)
    return dest

def obtain(record, cache):
    if 'download' in record:return fetch(record['download'],cache)
    dest=cache/record['name']
    if dest.exists():
        if dest.stat().st_size==record['size'] and sha256(dest)==record['sha256']:return dest
        raise ValueError('Existing archive differs: '+str(dest))
    partial=dest.with_name(dest.name+'.partial')
    with partial.open('wb') as output:
        for number,part in enumerate(record['parts'],1):
            print(f"  {number}/{len(record['parts'])}",flush=True)
            path=fetch(part,cache)
            with path.open('rb') as source:shutil.copyfileobj(source,output,1024*1024)
    if partial.stat().st_size!=record['size'] or sha256(partial)!=record['sha256']:
        raise ValueError('Reassembled archive checksum failed')
    partial.replace(dest)
    return dest

def extract(archive, destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:') as tf:
        targets=[]
        for member in tf.getmembers():
            parts=PurePosixPath(member.name).parts
            if (not parts or member.name.startswith('/') or '..' in parts or '\\' in member.name
                    or ':' in parts[0] or not (member.isfile() or member.isdir())):
                raise ValueError('Unsafe archive member: '+member.name)
            relative=Path(*parts)
            if member.name=='DATA_CONDITIONS.md':relative=Path(Path(archive).stem+'_DATA_CONDITIONS.md')
            target=(destination/relative).resolve()
            if os.path.commonpath([str(destination),str(target)])!=str(destination):raise ValueError('Archive path escapes destination')
            targets.append((member,target))
        # Validate every path before writing any content.
        for member,target in targets:
            if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with tf.extractfile(member) as source:
                if target.exists():
                    h=hashlib.sha256()
                    for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
                    if h.hexdigest()!=sha256(target):raise ValueError('Refusing to overwrite a different file: '+str(target))
                    continue
                temporary=target.with_name(target.name+'.extracting')
                with temporary.open('xb') as output:shutil.copyfileobj(source,output,1024*1024)
                temporary.replace(target)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index',type=Path,default=ROOT/'benchmark/archives.json')
    ap.add_argument('--groups',nargs='+',help='select groups; default all available groups')
    ap.add_argument('--dest',type=Path,default=ROOT,help='root containing results/ and real_results/')
    ap.add_argument('--cache',type=Path,default=ROOT/'download_cache')
    ap.add_argument('--list',action='store_true')
    ap.add_argument('--archives-only',action='store_true')
    args=ap.parse_args();index=json.loads(args.index.read_text(encoding='utf8'))
    records={r['group']:r for r in index['archives']}
    if args.list:
        for group,r in records.items():print(group,r['images'],f"{r['size']/1e9:.2f} GB")
        return
    selected=args.groups or list(records)
    unknown=set(selected)-set(records)
    if unknown:ap.error('Unknown groups: '+', '.join(sorted(unknown)))
    args.cache.mkdir(parents=True,exist_ok=True)
    for group in selected:
        record=records[group];print('Downloading',group,flush=True)
        archive=obtain(record,args.cache)
        if not args.archives_only:extract(archive,args.dest/record.get('extract_prefix',''))
        print('Verified',group,flush=True)

if __name__=='__main__':main()
