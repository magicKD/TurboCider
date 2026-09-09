"""Compare native 8-bit RGB/RGBA PNG outputs without extra image dependencies."""
import argparse,json,math,struct,zlib
from pathlib import Path
import numpy as np

def read_png(path):
 data=path.read_bytes();assert data[:8]==b'\x89PNG\r\n\x1a\n'
 pos=8;compressed=b''
 while pos<len(data):
  size=struct.unpack('>I',data[pos:pos+4])[0];kind=data[pos+4:pos+8];chunk=data[pos+8:pos+8+size];pos+=12+size
  if kind==b'IHDR':w,h,depth,color,compression,filtering,interlace=struct.unpack('>IIBBBBB',chunk)
  if kind==b'IDAT':compressed+=chunk
 assert depth==8 and color in (2,6) and interlace==0
 channels=3 if color==2 else 4;stride=w*channels;raw=zlib.decompress(compressed)
 rows=[];previous=bytearray(stride)
 for row in range(h):
  start=row*(stride+1);method=raw[start];current=bytearray(raw[start+1:start+1+stride])
  for i in range(stride):
   a=current[i-channels] if i>=channels else 0;b=previous[i];c=previous[i-channels] if i>=channels else 0
   if method==1:v=a
   elif method==2:v=b
   elif method==3:v=(a+b)//2
   elif method==4:
    q=a+b-c;da=abs(q-a);db=abs(q-b);dc=abs(q-c);v=a if da<=db and da<=dc else b if db<=dc else c
   elif method==0:v=0
   else:raise ValueError('PNG filter')
   current[i]=(current[i]+v)&255
  rows.append(current);previous=current
 return np.frombuffer(b''.join(rows),np.uint8).reshape(h,w,channels)[:,:,:3].astype(np.float32)

def compare(a,b):
 x=read_png(a);y=read_png(b);delta=x-y;mse=float(np.mean(delta**2))
 return {'reference':str(a),'candidate':str(b),'pixels_equal':bool(np.array_equal(x,y)),
         'mae':float(np.mean(abs(delta))),'psnr_db':10*math.log10(255**2/mse) if mse else None,
         'shape':list(x.shape)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();r=[]
 for x,y in [('fixed1056-1-short.png','fixed-1-short.png'),('fixed1056-1-short.png','enumerated-1-short.png'),('fixed1056-1-short.png','range-1-short.png'),('fixed-4-long.png','enumerated-4-long.png'),('fixed-4-long.png','range-4-long.png'),('gpu-1-short.png','enumerated-1-short.png'),('gpu-4-long.png','enumerated-4-long.png')]:
  if (a.root/x).exists() and (a.root/y).exists():r.append(compare(a.root/x,a.root/y))
 (a.root/'image-comparison.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
