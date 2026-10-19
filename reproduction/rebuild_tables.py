"""Rebuild numerical tables in isolation and verify the published metric cells."""
from pathlib import Path
import json,os,re,shutil,subprocess,sys,tempfile

HERE=Path(__file__).resolve().parent

def numbers(text):
    rows=[]
    for line in text.splitlines():
        if ' & ' not in line or line.lstrip().startswith(r'\caption'):
            continue
        cells=line.split(' & ')[1:]
        values=[]
        for cell in cells:
            cell=re.sub(r'\\cite\{[^}]+\}','',cell)
            values.extend(re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?',cell))
        if values:rows.append(values)
    return rows

def main():
    report=[]
    with tempfile.TemporaryDirectory(prefix='stackmff_reproduction_') as temp:
        root=Path(temp);shutil.copytree(HERE/'work',root/'work')
        tables=root/'work/paper/tables'
        for old in tables.glob('tab_*.tex'):
            assert old.resolve().is_relative_to(root.resolve())
            old.unlink()
        producers=sorted(tables.glob('make_*.py'))
        for producer in producers:
            before={p.name:p.read_bytes() for p in tables.glob('*.tex')}
            env=dict(os.environ,PYTHONUTF8='1')
            result=subprocess.run([sys.executable,str(producer)],cwd=root,capture_output=True,env=env)
            if result.returncode:
                raise RuntimeError(producer.name+': '+result.stderr.decode(errors='replace')[-1600:])
            for generated in tables.glob('*.tex'):
                if generated.read_bytes()==before.get(generated.name):continue
                expected=HERE/'expected_tables'/generated.name
                if not expected.exists():
                    report.append({'producer':producer.name,'table':generated.name,'status':'historical output, not in current manuscript'})
                    continue
                actual=numbers(generated.read_text(encoding='utf8'))
                wanted=numbers(expected.read_text(encoding='utf8'))
                if actual!=wanted:
                    differences=[{'row':i,'generated':a,'expected':b} for i,(a,b) in enumerate(zip(actual,wanted)) if a!=b]
                    raise AssertionError(json.dumps({'table':generated.name,'row_counts':[len(actual),len(wanted)],'differences':differences[:5]}))
                report.append({'producer':producer.name,'table':generated.name,'status':'metric cells match'})
        subprocess.run([sys.executable,str(tables/'add_method_venues.py')],cwd=root,check=True,capture_output=True)
    print(json.dumps({'producers':len(producers),'tables':report},indent=2))

if __name__=='__main__':main()
