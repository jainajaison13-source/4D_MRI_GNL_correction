import numpy as np, re, sys
def read(path, want):
    out={}
    with open(path,'rb') as f:
        hdr={}
        while True:
            line=f.readline()
            if not line: break
            s=line.decode('latin1').strip()
            if s.startswith('DIMENSIONS'): hdr['dims']=tuple(int(x) for x in s.split()[1:])
            elif s.startswith('ORIGIN'): hdr['origin']=tuple(float(x) for x in s.split()[1:])
            elif s.startswith('SPACING'): hdr['spacing']=tuple(float(x) for x in s.split()[1:])
            elif s.startswith('POINT_DATA'): npts=int(s.split()[1])
            elif s.startswith('VECTORS') or s.startswith('SCALARS'):
                kind,name=s.split()[0],s.split()[1]
                if kind=='SCALARS':
                    pos=f.tell(); nxt=f.readline().decode('latin1')
                    if not nxt.startswith('LOOKUP_TABLE'): f.seek(pos)
                nc=3 if kind=='VECTORS' else 1
                nbytes=npts*nc*4
                if name in want:
                    a=np.frombuffer(f.read(nbytes),dtype='>f4').astype(np.float32)
                    nx,ny,nz=hdr['dims']
                    a=a.reshape(nz,ny,nx,nc) if nc==3 else a.reshape(nz,ny,nx)
                    a=np.moveaxis(a,[0,1,2],[2,1,0]) if nc==1 else np.moveaxis(a,[0,1,2,3],[3,2,1,0])
                    out[name]=a  # vectors -> (3,nx,ny,nz) ; scalars -> (nx,ny,nz)
                else:
                    f.seek(nbytes,1)
                f.readline()
    return hdr,out
if __name__=='__main__':
    h,o=read(sys.argv[1],{'velocity_measured','velocity_corrected','flow_coherence','speed_timeavg','anatomy_magnitude'})
    print(h,{k:v.shape for k,v in o.items()})
    np.savez('cow.npz',**o,origin=h['origin'],spacing=h['spacing'])
