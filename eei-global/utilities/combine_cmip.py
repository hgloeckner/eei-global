# %%
import xarray as xr
import numpy as np
from pathlib import Path

directory = Path("../../data/CMIP")
files = directory.glob("*.nc")

dss = []
dsmets = []
for file in files:
    ds = xr.open_dataset(file).convert_calendar("standard", align_on="year")
    model = file.stem.split("_")[1]
    forcing = file.stem.split("_")[2]
    if np.datetime64("2010-06-02") in ds["time"].values:
        dsmets.append(ds.expand_dims(model=[model], forcing=[forcing]))
    else:
        dss.append(ds.expand_dims(model=[model], forcing=[forcing]))

# %%
mapping = {
    "net": "net",
    "cre": "net cre",
    "clear": "net cs",
    "lw": "lw net",
    "lwclear": "lw cs",
    "lwcre": "lw cre",
    "sw": "sw net",
    "swclear": "sw cs",
    "swcre": "sw cre",
}
rearanged = []
for ds in dsmets:
    rearanged.extend(
        ds[var].rename("N").expand_dims(rad=[mapping[var]]).compute()
        for var in ds.variables
        if var not in ds.coords and (var != "bound_time")
    )

dsmet = xr.merge(rearanged, combine_attrs="drop_conflicts")
dsmet.to_netcdf("../data/HadAM3.nc")
# %%

rearanged = []
for ds in dss:
    rearanged.extend(
        ds[var].rename("N").expand_dims(rad=[mapping[var]]).compute()
        for var in ds.variables
        if var not in ds.coords and (var != "bound_time")
    )
ds = xr.merge(rearanged, combine_attrs="drop_conflicts")
ds.to_netcdf("../data/CMIP.nc")
# ds = xr.concat(dss, dim="model")
