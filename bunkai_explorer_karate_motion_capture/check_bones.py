import json, struct

with open(r'static/models/karateka_lowpoly.glb', 'rb') as f:
    f.read(12)
    clen, ctype = struct.unpack('<II', f.read(8))
    gltf = json.loads(f.read(clen).decode('utf-8'))

nodes = gltf['nodes']
pmap = {c: i for i, n in enumerate(nodes) for c in n.get('children', [])}
for i, n in enumerate(nodes):
    name = n.get('name', '')
    if not any(k in name for k in ['Index', 'Thumb', 'Middle', 'Ring', 'Pinky']):
        p_name = nodes[pmap[i]].get('name', 'ROOT') if i in pmap else 'ROOT'
        trans = [round(x, 4) for x in n.get('translation', [0, 0, 0])]
        rot = [round(x, 4) for x in n.get('rotation', [0, 0, 0, 1])]
        print(f"{name:20s} <- {p_name:20s} trans={trans} rot={rot}")
