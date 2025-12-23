set -euo pipefail
REPO="$(pwd)"
F="$REPO/unified-av-data-loader/src/trajdata/dataset_specific/nusc/nusc_utils.py"
python - <<'PY'
import re, pathlib
p=pathlib.Path("unified-av-data-loader/src/trajdata/dataset_specific/nusc/nusc_utils.py")
s=p.read_text()
if "_ensure_cols" not in s:
    s=s.replace("import numpy as np","import numpy as np\n\ndef _ensure_cols(arrs,k):\n    out=[]\n    for a in arrs:\n        import numpy as np\n        a=np.asarray(a)\n        if a.ndim==1:\n            a=a.reshape(-1,a.shape[0])\n        if a.ndim==2 and a.shape[1]<k:\n            pad=np.zeros((a.shape[0],k-a.shape[1]))\n            a=np.concatenate([a,pad],axis=1)\n        elif a.ndim==2 and a.shape[1]>k:\n            a=a[:, :k]\n        else:\n            a=a.reshape(-1,k)\n        out.append(a)\n    return out\n")
pat_block=r"translation_list\s*=\s*\[.*?for a in translation_list]\s*\n\s*translations_np\s*=\s*np\.concatenate\(\s*translation_list\s*,\s*axis\s*=\s*0\s*\)"
if re.search(pat_block,s,flags=re.S):
    s=re.sub(pat_block,"translation_list=_ensure_cols(translation_list,3)\n    translations_np=np.concatenate(translation_list,axis=0)",s,1,flags=re.S)
else:
    pat_line=r"translations_np\s*=\s*np\.concatenate\(\s*translation_list\s*,\s*axis\s*=\s*0\s*\)"
    s=re.sub(pat_line,"translation_list=_ensure_cols(translation_list,3)\n    translations_np=np.concatenate(translation_list,axis=0)",s,1)
s=s.replace("velocities_np = np.concatenate(velocity_list, axis=0)","velocity_list=_ensure_cols(velocity_list,3)\n    velocities_np=np.concatenate(velocity_list,axis=0)")
s=s.replace("velocities_np = np.concatenate(velocities_list, axis=0)","velocities_list=_ensure_cols(velocities_list,3)\n    velocities_np=np.concatenate(velocities_list,axis=0)")
p.write_text(s)
print("PATCH_APPLIED")
PY
