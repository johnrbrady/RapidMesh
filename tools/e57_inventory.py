import sys, re, os, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from e57_xml import read_e57_xml

def g(x, tag):
    m = re.search(r'<%s[^>]*>([^<]*)</%s>' % (tag,tag), x)
    return m.group(1) if m else None
def gi(x, tag):
    m = re.search(r'<%s type="Integer"[^/>]*/>' % tag, x)
    if m: return 0
    v = g(x,tag)
    return int(v) if v not in (None,'') else None

for p in sys.argv[1:]:
    meta, xmlb = read_e57_xml(p)
    x = xmlb.decode('utf-8','replace')
    scans = re.findall(r'<vectorChild type="Structure">.*?</vectorChild>', x, re.S)
    d3 = re.search(r'<data3D.*?</data3D>', x, re.S)
    d3s = d3.group(0) if d3 else ''
    scans = re.findall(r'<vectorChild type="Structure">(.*?)</vectorChild>(?=\s*(?:<vectorChild|</data3D))', d3s, re.S)
    imgs = re.search(r'<images2D[^>]*>(.*?)</images2D>', x, re.S)
    nimg = len(re.findall(r'<vectorChild', imgs.group(1))) if imgs else 0
    print('='*70)
    print(os.path.basename(p), '| %.1f MB'%(meta['file_len']/1e6), '| xml %d B'%meta['xml_len'],
          '| lib', (g(x,'e57LibraryVersion') or '?'), '| scans', len(scans), '| images2D', nimg)
    tot=0
    for i,s in enumerate(scans):
        proto = re.search(r'<points type="CompressedVector"[^>]*recordCount="(\d+)".*?<prototype type="Structure">(.*?)</prototype>', s, re.S)
        rc = int(proto.group(1)) if proto else -1
        fields = re.findall(r'<(\w+) type="(?:Float|Integer|ScaledInteger)"', proto.group(2)) if proto else []
        ib = re.search(r'<indexBounds.*?</indexBounds>', s, re.S)
        rows=cols=None
        if ib:
            rmax = gi(ib.group(0),'rowMaximum'); cmax = gi(ib.group(0),'columnMaximum')
            rmin = gi(ib.group(0),'rowMinimum') or 0; cmin = gi(ib.group(0),'columnMinimum') or 0
            if rmax is not None: rows = rmax-rmin+1
            if cmax is not None: cols = cmax-cmin+1
        tr = re.search(r'<translation type="Structure">\s*<x[^>]*>([^<]*)</x>\s*<y[^>]*>([^<]*)</y>\s*<z[^>]*>([^<]*)</z>', s)
        pose = tuple(round(float(v),3) for v in tr.groups()) if tr else None
        name = g(s,'name')
        lat = (rows*cols) if (rows and cols) else None
        fill = ('%.1f%%'%(100*rc/lat)) if lat else '-'
        tot += rc if rc>0 else 0
        print('  [%d] %s pts=%s lattice=%sx%s fill=%s pose=%s' % (i, (name or '')[:28], f'{rc:,}', cols, rows, fill, pose))
        print('      fields:', ','.join(fields))
        if i>=3 and len(scans)>5:
            print('      ... (%d more scans)'%(len(scans)-i-1)); break
    print('  TOTAL points (listed):', f'{tot:,}')
