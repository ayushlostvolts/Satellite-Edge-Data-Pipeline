import os
import glob
import xarray as xr

files = sorted(glob.glob("Dataset/RF25_ind*.nc"))
print(f"Found {len(files)} NetCDF files:")
total_days = 0
for f in files:
    with xr.open_dataset(f) as ds:
        t_len = len(ds["TIME"])
        total_days += t_len
        yr = str(ds["TIME"].values[0])[:4]
        print(f"  {os.path.basename(f)}: Year {yr}, Days={t_len}, Shape={ds['RAINFALL'].shape}")
print(f"\nTotal days across all files: {total_days}")
