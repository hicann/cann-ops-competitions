import ctypes
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'operator/python'))
from max_pool_backward import Tiling

lib=ctypes.CDLL('/tmp/libpool_geometry.so')
ptr=ctypes.POINTER(ctypes.c_int64)
lib.test_plan.argtypes=[ptr,ptr,ptr,ptr,ctypes.c_int64,ptr,ctypes.c_int64,ptr,ctypes.c_int64,ptr,ctypes.c_int64,ctypes.c_bool,ctypes.POINTER(Tiling)]
lib.test_plan.restype=ctypes.c_int

def check(x,g,m,attrs):
    arrays=[(ctypes.c_int64*len(v))(*v) for v in [x,g,m,*attrs[:4]]]
    t=Tiling()
    status=lib.test_plan(*arrays[:3],arrays[3],len(attrs[0]),arrays[4],len(attrs[1]),arrays[5],len(attrs[2]),arrays[6],len(attrs[3]),attrs[4],ctypes.byref(t))
    return status,t

records=[]
complete=False
try:
    for case in json.loads((ROOT/'max_pool2d_with_mask_backward_cases.json').read_text()):
        desc={d['name']:d for d in case['input_desc']};at={a['name']:a['value'] for a in case['attr_desc']}
        x,g,m=[desc[n]['shape'] for n in ['self','gradOutput','indices']]
        attrs=[at[n] for n in ['kernelSize','stride','padding','dilation','ceilMode']]
        status,t=check(x,g,m,attrs)
        assert status==0 and t.planes==x[0]*x[1] and (t.height,t.width)==tuple(x[2:]) and (t.outHeight,t.outWidth)==tuple(g[2:]),case['case_name']
        records.append({'name':case['case_name'],'status':'PASS'})
    x=[1,1,5,5];g=[1,1,3,3];m=[1,1,9,64];attrs=[[3],[1],[0],[1],False]
    import copy
    for i in range(4):
        for kind,original in [('input',x),('gradient',g),('mask',m)]:
            altered=list(original);altered[i]=0
            status,_=check(altered if kind=='input' else x,altered if kind=='gradient' else g,altered if kind=='mask' else m,attrs)
            assert status!=0
            records.append({'name':f'reject_{kind}_{i}','status':'PASS'})
    for name,idx,value in [('zero_kernel',0,[0]),('zero_stride',1,[0]),('negative_padding',2,[-1]),('large_padding',2,[4]),('dilation',3,[2]),('oversized_attribute',0,[3,3,3])]:
        a=copy.deepcopy(attrs);a[idx]=value
        assert check(x,g,m,a)[0]!=0
        records.append({'name':name,'status':'PASS'})
    # Equivalent compact/default attributes; ceil's right-padding correction.
    for name,x,g,m,a in [
        ('empty_stride',[1,1,5,5],[1,1,2,2],[1,1,4,64],[[2],[],[0],[1],False]),
        ('ceil_correction',[1,1,4,4],[1,1,2,2],[1,1,4,64],[[2],[3],[1],[1],True]),
        ('oversized_kernel_rejected',[1,1,2,2],[1,1,1,1],[1,1,100,64],[[10],[1],[0],[1],False])]:
        status,_=check(x,g,m,a)
        assert (status!=0)==(name=='oversized_kernel_rejected')
        records.append({'name':name,'status':'PASS'})
    complete=True
finally:
    path=ROOT/'operator/reports/tiling.json';path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({'scope':'same MakePoolPlan C++ function used by registered host tiling','passed':len(records),
                               'failed':0 if complete else 1,'complete':complete,'records':records},indent=2))
print('HOST_TILING',len(records),'PASS')
