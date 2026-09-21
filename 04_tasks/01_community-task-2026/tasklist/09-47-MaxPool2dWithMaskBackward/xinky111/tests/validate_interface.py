"""Public adapter validation, independent expected results, and repeatability."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import ml_dtypes

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'operator/python'))
from max_pool_backward import MaxPoolBackward


def main():
    p=argparse.ArgumentParser();p.add_argument('--library',default='/tmp/pool-cpu/libmax_pool_backward.so');p.add_argument('--report',type=Path,required=True)
    args=p.parse_args();op=MaxPoolBackward(args.library);records=[]
    def fixture(dtype=np.float32):
        x=np.zeros((1,1,5,5),dtype=dtype);g=np.ones((1,1,3,3),dtype=dtype)
        mask=np.zeros((1,1,9,64),np.int8)
        mask.reshape(-1)[:36]=np.full(9,12,dtype='<i4').view(np.int8)
        return [g,x,mask,[3],[1],[0],[1],False]
    def test(name,fn):
        try:fn();records.append({'name':name,'status':'PASS'})
        except Exception as e:records.append({'name':name,'status':'FAIL','error':repr(e)});raise
        print('PASS',name,flush=True)
    def reject(name,mutation,kwargs=None):
        a=fixture();mutation(a)
        try:op(*a,**(kwargs or {}))
        except ValueError:return
        raise AssertionError(name+' unexpectedly accepted')
    try:
        mutations={
            'kernel_zero':lambda a:a.__setitem__(3,[0]),
            'stride_zero':lambda a:a.__setitem__(4,[0]),
            'negative_padding':lambda a:a.__setitem__(5,[-1]),
            'too_much_padding':lambda a:a.__setitem__(5,[4]),
            'dilation_two':lambda a:a.__setitem__(6,[2]),
            'empty_kernel':lambda a:a.__setitem__(3,[]),
            'oversized_list':lambda a:a.__setitem__(3,[3,3,3]),
            'fractional_attribute':lambda a:a.__setitem__(3,[3.0]),
            'boolean_attribute':lambda a:a.__setitem__(3,[True]),
            'nonbool_ceil':lambda a:a.__setitem__(7,1),
            'grad_shape':lambda a:a.__setitem__(0,a[0][...,:2]),
            'input_dtype':lambda a:a.__setitem__(1,a[1].astype(np.float16)),
            'mask_dtype':lambda a:a.__setitem__(2,a[2].astype(np.int16)),
            'mask_shape':lambda a:a.__setitem__(2,a[2][...,:32]),
            'nan_grad':lambda a:a[0].fill(np.nan),
            'negative_inf':lambda a:a[0].fill(-np.inf),
            'negative_index':lambda a:a[2].reshape(-1).__setitem__(slice(0,4),np.array([-1],'<i4').view(np.int8)),
            'outside_plane':lambda a:a[2].reshape(-1).__setitem__(slice(0,4),np.array([25],'<i4').view(np.int8)),
            'outside_window':lambda a:a[2].reshape(-1).__setitem__(slice(0,4),np.array([24],'<i4').view(np.int8)),
        }
        for name,fn in mutations.items():test(name,lambda n=name,f=fn:reject(n,f))
        test('invalid_blocks',lambda:reject('blocks',lambda a:None,{'blocks':0}))
        def output_tests():
            a=fixture()
            for out in (a[1],np.zeros((1,1,5,5),np.float16),np.broadcast_to(np.zeros(1,np.float32),(1,1,5,5))):
                try:op(*a,out=out)
                except ValueError:continue
                raise AssertionError('bad output accepted')
        test('invalid_outputs',output_tests)
        for dtype in (np.float32,np.float16,ml_dtypes.bfloat16):
            def known(dtype=dtype):
                a=fixture(dtype);a[0][...]=np.array([32768,1,-32768,0.5,-0.5,2,-2,2**-20,-2**-21],dtype=dtype).reshape(1,1,3,3)
                acc=np.float32(0)
                for value in a[0].flat:acc=np.float32(acc+np.float32(value))
                expected=np.zeros(a[1].shape,dtype=dtype);expected[0,0,2,2]=acc
                prior=None
                for blocks in (1,3,8,1):
                    actual=op(*a,blocks=blocks)
                    assert actual.tobytes()==expected.tobytes()
                    if prior is not None:assert actual.tobytes()==prior
                    prior=actual.tobytes()
                # Negative strides preserve logical values through reversed backing arrays.
                def negative(v):return v[...,::-1].copy()[...,::-1]
                a[:3]=map(negative,a[:3])
                target=negative(np.zeros_like(expected))
                assert op(*a,out=target).tobytes()==expected.tobytes()
            test('known_accumulation_repeatability_'+np.dtype(dtype).name,known)
    finally:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        scope = 'Atlas 800T A2 / Atlas 300V Pro NPU host adapter rejection and hardware execution tests' if 'npu' in str(args.library) else 'host adapter rejection and actual CPU twin known-result tests'
        args.report.write_text(json.dumps({'scope':scope,'records':records,
                'passed':sum(x['status']=='PASS' for x in records),'failed':sum(x['status']=='FAIL' for x in records)},indent=2))
    print('DONE',len(records),'passed',flush=True)
if __name__=='__main__':main()
