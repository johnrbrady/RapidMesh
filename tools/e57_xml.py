import struct, sys, io

def read_e57_xml(path):
    with open(path,'rb') as f:
        hdr = f.read(48)
        vmaj, vmin = struct.unpack('<II', hdr[8:16])
        flen, xoff, xlen, page = struct.unpack('<QQQQ', hdr[16:48])
        if page == 0: page = 1024
        payload = page - 4
        out = io.BytesIO(); pos = xoff; remaining = xlen
        while remaining > 0:
            p_index, p_off = divmod(pos, page)
            if p_off >= payload:
                pos = (p_index+1)*page
                continue
            n = min(payload - p_off, remaining)
            f.seek(p_index*page + p_off)
            out.write(f.read(n))
            remaining -= n; pos += n
        return dict(magic=hdr[:8], ver=(vmaj,vmin), file_len=flen,
                    xml_off=xoff, xml_len=xlen, page=page), out.getvalue()

if __name__ == '__main__':
    meta, xml = read_e57_xml(sys.argv[1])
    print(meta, file=sys.stderr)
    sys.stdout.write(xml.decode('utf-8','replace'))
