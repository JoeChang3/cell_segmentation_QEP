from config import ROOT,REPO
from zipfile import ZipFile
from io import BytesIO
from pathlib import Path
from PIL import Image
import hashlib,json,re
import numpy as np

def image_key(a):
    return (a.shape,hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
with ZipFile(ROOT/"data/images.zip") as archive,ZipFile(ROOT/"data/cells.zip") as gt:
    old=set()
    for image in REPO.glob("data/*test_images/*/original_fig.*"):
        a=np.array(Image.open(image))
        old.add(image_key(a[...,0] if a.ndim==3 else a))
    fields=[]; excluded=[]; refs={}; keys=set()
    for image in sorted(n for n in archive.namelist() if n.lower().endswith("_c1.tif")):
        name=Path(image).stem
        masks=[n for n in gt.namelist() if re.fullmatch(re.escape(name)+r'_\d+\.tif',Path(n).name,re.I)]
        if not masks: excluded.append([name,"no supplied cell mask"]);continue
        rgb=np.array(Image.open(BytesIO(archive.read(image))))
        assert rgb.ndim==3 and np.array_equal(rgb[:,:,0],rgb[:,:,1]) and not rgb[:,:,2].any()
        a=rgb[:,:,0];key=image_key(a)
        if key in old: excluded.append([name,"legacy duplicate"]);continue
        assert key not in keys
        keys.add(key); Image.fromarray(a).save(ROOT/"data"/(name+".png"))
        fields.append(name);refs[name]=sorted(masks)
    (ROOT/"data/manifest.json").write_text(json.dumps(dict(fields=fields,excluded=excluded,reference_masks=refs),indent=2))
    print(len(fields),len(excluded),sum(map(len,refs.values())))
