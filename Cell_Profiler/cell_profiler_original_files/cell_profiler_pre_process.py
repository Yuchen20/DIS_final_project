import os, glob, re, csv
import numpy as np
import tifffile

# ── CONFIG ──────────────────────────────────────────────────────────────
src_folder = "Cell_Profiler/cell_profiler_original_files/deephcs"          # your .npy input
sample_name = src_folder.split("/")[-1]
dst_folder = f"Cell_Profiler/cell_profiler_input/{sample_name}_tiff"    # where TIFFs+CSV go

os.makedirs(dst_folder, exist_ok=True)
IMG_DIR   = os.path.join(dst_folder, "images")
ILLUM_DIR = os.path.join(dst_folder, "illum")
os.makedirs(IMG_DIR,   exist_ok=True)
os.makedirs(ILLUM_DIR, exist_ok=True)
csv_path  = os.path.join(dst_folder, "load_data.csv")

# The five channels in order:
channels = ["Mito","AGP","RNA","ER","DNA"]

# Regex to parse filenames like r01c01f01p01_pred.npy
pat = re.compile(r"r(\d{2})c(\d{2})f(\d{2})p(\d{2})_(pred|target)\.npy")

# Helper to map "01"->"A", etc.
def row2letter(r): return chr(ord('A') + int(r) - 1)

# Step 1+2: split arrays to TIFFs and write illum images once
print("Splitting .npy → images/ and writing illum/ …")
seen_illum = set()
for npy_file in glob.glob(os.path.join(src_folder, "*.npy")):
    if npy_file.endswith("target.npy"):
        continue
    m = pat.search(os.path.basename(npy_file))
    if not m:
        print(f"Skipping (no match): {npy_file}")
        continue

    row, col, fov, p, kind = m.groups()
    base = f"r{row}c{col}f{fov}_{kind}"
    arr = np.load(npy_file)

    # ensure channels-last
    if arr.ndim == 3 and arr.shape[0] == len(channels):
        arr = np.moveaxis(arr, 0, -1)

    H, W, C = arr.shape
    if C != len(channels):
        print("Skipping:", npy_file, "has shape", arr.shape)
        continue

    # Normalize each channel to [0,1] and cast to float32
    for i, ch in enumerate(channels):
        data = arr[..., i].astype(np.float32)
        mn, mx = data.min(), data.max()
        # avoid division by zero if flat image
        if mx > mn:
            data = (data - mn) / (mx - mn)
        else:
            data = np.zeros_like(data)

        out_fn = f"{base}_{ch}.tif"
        out_path = os.path.join(IMG_DIR, out_fn)
        # write as true float32 TIFF
        tifffile.imwrite(out_path, data, dtype=np.float32)

    # write one illum-per-channel (float=1.0)
    for ch in channels:
        key = (ch, H, W)
        if key in seen_illum:
            continue
        illum = np.ones((H, W), dtype=np.float32)
        tifffile.imwrite(os.path.join(ILLUM_DIR, f"Illum{ch}.tif"),
                         illum, dtype=np.float32)
        seen_illum.add(key)

print("Done splitting + illum.")

# Step 3: write the CSV
print(f"Writing CSV → {csv_path}")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    # Header: metadata + 5 image channels + 5 illum functions
    header = ["Metadata_Plate","Metadata_Well","Metadata_Site"]
    for ch in channels:
        header += [f"PathName_pred_{ch}", f"FileName_pred_{ch}"]
    for ch in channels:
        header += [f"PathName_Illum{ch}", f"FileName_Illum{ch}"]
    w.writerow(header)

    # Collect one row per site (row,col,fov) keyed by plate/well/site
    sites = {}
    for fn in os.listdir(IMG_DIR):
        m = re.match(r"(r\d{2}c\d{2}f\d{2})_(pred)_(\w+)\.tif", fn)
        if not m:
            continue
        base, kind, ch = m.groups()
        # parse base again
        m2 = pat.match(base + "p01_" + kind + ".npy")
        print(base + "p01_" + kind + ".npy", m2)
        if not m2:
            continue
        row, col, fov, p, _ = m2.groups()
        key = (row, col, fov)
        sites.setdefault(key, {})[f"{kind}_{ch}"] = fn

    # Now write each site
    for (row,col,fov), d in sorted(sites.items()):
        plate = "1"
        well  = f"{row2letter(row)}{int(col):02d}"
        site  = int(fov)
        row_vals = [plate, well[-1], site]
        # pred images only
        for ch in channels:
            fn = d.get(f"pred_{ch}")
            row_vals += [f"{sample_name}_tiff/images", fn if fn else ""]
        # illum
        for ch in channels:
            row_vals += [f"{sample_name}_tiff/illum", f"Illum{ch}.tif"]
        w.writerow(row_vals)

print("All done! You can now point CellProfiler's LoadData at:")
print("  Default Input Folder =", dst_folder)
print("  Sub‐folder            = '' (leave blank)")
print("  Name of the file      =", os.path.basename(csv_path))
