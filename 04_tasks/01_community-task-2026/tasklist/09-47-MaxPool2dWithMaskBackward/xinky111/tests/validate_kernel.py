"""Run the compiled Ascend C library against the supplied golden (CPU twin mode)."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'operator/python'))
from max_pool_backward import MaxPoolBackward
spec = importlib.util.spec_from_file_location('golden', ROOT / 'max_pool2d_with_mask_backward_golden.py')
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)
DTYPES = {'float32': np.float32, 'float16': np.float16, 'bfloat16': ml_dtypes.bfloat16}


def view(a):
    storage = np.empty((*a.shape[:-1], a.shape[-1]*2+3), dtype=a.dtype)
    result = storage[..., 1:1+2*a.shape[-1]:2]
    result[...] = a
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', default='/tmp/pool-cpu/libmax_pool_backward.so')
    parser.add_argument('--suite', choices=['smoke', 'full', 'edges', 'random', 'tails'], default='smoke')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--dtypes', nargs='+', choices=list(DTYPES), default=list(DTYPES))
    args = parser.parse_args()
    op = MaxPoolBackward(args.library)
    records = []
    scope = 'Atlas 800T A2 / Atlas 300V Pro NPU hardware execution via ACL runtime' if 'npu' in str(args.library) else 'actual Ascend C kernel CPU twin execution'
    report = {'scope': scope, 'dtypes':args.dtypes,
              'suite': args.suite, 'records': records,
              'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((ROOT/'operator').rglob('*')) if p.is_file()
                         and p.suffix in ('.asc', '.h', '.py', '.txt')},
              'library_sha256': hashlib.sha256(Path(args.library).read_bytes()).hexdigest()}

    def run(name, shape, dtype, attrs, seed, blocks=1, strided=False, ties=False, numeric=None):
        start = time.perf_counter()
        try:
            x = np.random.default_rng(seed).uniform(-5, 5, size=shape).astype(dtype)
            if ties:
                x.fill(1)
            pooled, mask = golden.maxpool2d_with_argmax_expect(x, *attrs)
            grad = np.random.default_rng(seed+1000000).uniform(-5, 5, size=pooled.shape).astype(dtype)
            if numeric is not None:
                grad[...] = np.resize(np.asarray(numeric, dtype=dtype), grad.shape)
            expected = golden.max_pool2d_with_mask_backward_golden(grad,x,mask,*attrs)[0]
            if strided:
                x, grad, mask = map(view, (x,grad,mask))
            # Dirty bytes outside global prefix are deliberately not zero.
            if not strided:
                mask.reshape(-1)[4*grad.size:] = -17
            output = view(np.zeros(shape, dtype=dtype)) if strided else None
            actual = op(grad,x,mask,*attrs,out=output,blocks=blocks)
            np.testing.assert_array_equal(actual,expected)
            assert actual.tobytes() == expected.tobytes(), 'bitwise mismatch'
            records.append({'name':name,'dtype':np.dtype(dtype).name,'shape':list(shape),'attrs':attrs,
                            'seed':seed,'blocks':blocks,'noncontiguous':strided,'status':'PASS',
                            'bitwise_equal':True,'seconds':round(time.perf_counter()-start,3)})
        except Exception as exc:
            records.append({'name':name,'status':'FAIL','error':repr(exc)})
            raise
        print(f"{len(records)} PASS {name} {records[-1]['seconds']}s",flush=True)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    try:
        if args.suite == 'full':
            for c in json.loads((ROOT/'max_pool2d_with_mask_backward_cases.json').read_text()):
                desc={d['name']:d for d in c['input_desc']}; attr={a['name']:a['value'] for a in c['attr_desc']}
                if desc['self']['data_type'] not in args.dtypes: continue
                run(c['case_name'],tuple(desc['self']['shape']),DTYPES[desc['self']['data_type']],
                    [attr[k] for k in ['kernelSize','stride','padding','dilation','ceilMode']],
                    20260830+int(c['case_name'].split('_')[1]),blocks=4)
        elif args.suite=='random':
            rng=np.random.default_rng(20260919)
            for i in range(50):
                h,w=map(int,rng.integers(4,19,size=2))
                kh=int(rng.integers(1,min(h,7)+1));kw=int(rng.integers(2,min(w,7)+1))
                sh,sw=map(int,rng.integers(1,8,size=2))
                ph,pw=int(rng.integers(0,kh//2+1)),int(rng.integers(0,kw//2+1))
                attrs=[[kh,kw],[] if i%4==0 else [sh,sw],[ph,pw],[1],bool(i%2)]
                shape=(int(rng.integers(1,3)),int(rng.integers(1,4)),h,w)
                for dtype_name,dtype in DTYPES.items():
                    if dtype_name in args.dtypes:
                        run(f'random_{i}_{dtype_name}',shape,dtype,attrs,20260919+i,
                            blocks=(1,3,8)[i%3],strided=bool(i%2),ties=i%5==0)
        else:
            scenarios = [('overlap',(2,3,7,9),[[3],[1],[1],[1],False]),
                         ('empty_stride',(1,3,7,9),[[2],[],[0],[1],False]),
                         ('rect_ceil',(2,2,7,9),[[3,2],[2,3],[1,0],[1],True]),
                         ('tail',(1,1,3,5),[[1,2],[1],[0],[1],False]),
                         ('gaps',(2,2,9,11),[[2],[3],[0],[1],False])]
            if args.suite == 'smoke': scenarios=scenarios[:1]
            if args.suite == 'tails':
                scenarios=[(f'tile_boundary_{n}',(1,1,1,n),[[1,2],[1],[0],[1],False]) for n in (8191,8192,8193)]
            for name,shape,attrs in scenarios:
                for dtype_name,dtype in DTYPES.items():
                    if dtype_name not in args.dtypes: continue
                    for blocks in ([1] if args.suite=='smoke' else [1,3,8]):
                        for strided in ([False] if args.suite=='smoke' else [False,True]):
                            run(f'{name}_{dtype_name}_b{blocks}_strided{strided}',shape,dtype,attrs,20260917,
                                blocks=blocks,strided=strided,ties=(blocks==3))
            if args.suite=='edges':
                for dtype_name,dtype in DTYPES.items():
                    if dtype_name not in args.dtypes: continue
                    run('numeric_'+dtype_name,(1,2,5,5),dtype,[[3],[1],[1],[1],False],20260917,
                        numeric=[32768,1,-32768,2**-20,-2**-21,0,0.5,-0.5],blocks=3)
    finally:
        report['passed']=sum(r['status']=='PASS' for r in records)
        report['failed']=sum(r['status']=='FAIL' for r in records)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"DONE {report['passed']} passed, {report['failed']} failed",flush=True)

if __name__=='__main__': main()
