set -euo pipefail
REPO="$(pwd)"
TARGET="$REPO/unified-av-data-loader/src/trajdata/dataset_specific/nusc/nusc_utils.py"
cp "$TARGET" "$TARGET.bak_$(date +%s)"
python - <<'PY'
import re,sys,pathlib
p=pathlib.Path("unified-av-data-loader/src/trajdata/dataset_specific/nusc/nusc_utils.py")
s=p.read_text()
pat=r"translations_np\s*=\s*np\.concatenate\(\s*translation_list\s*,\s*axis\s*=\s*0\s*\)"
rep=("translation_list = [np.asarray(a)[:, :2] if np.asarray(a).ndim==2 and np.asarray(a).shape[1]>=2 else np.asarray(a).reshape(-1,2) for a in translation_list]\n    translations_np = np.concatenate(translation_list, axis=0)")
if re.search(pat,s) is None: print("PATTERN_NOT_FOUND"); sys.exit(2)
s=re.sub(pat,rep,s,1)
p.write_text(s)
print("PATCHED")
PY
python - <<'PY'
import trajdata.dataset_specific.nusc.nusc_utils as nu
print("IMPORT_OK", nu.__file__)
PY
