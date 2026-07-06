import pandas as pd
import xarray as xr


def open_berkeley():

    names = [
        "year",
        "month",
        "monthly_anomaly",
        "monthly_unc",
        "annual_anomaly",
        "annual_unc",
        "5yr_anomaly",
        "5yr_unc",
        "10yr_anomaly",
        "10yr_unc",
        "20yr_anomaly",
        "20yr_unc",
    ]
    df = pd.read_csv(
        "../data/berkeley.txt",
        sep="\s+",
        index_col=0,
        skiprows=94,
        header=None,
        names=names,
    )
    df.index.name = "year"
    df["time"] = pd.to_datetime(dict(year=df.index, month=df["month"], day=15))
    return df.set_index("time")[
        [
            "monthly_anomaly",
            "monthly_unc",
            "annual_anomaly",
            "annual_unc",
        ]
    ]


def open_hadcrut():
    hadcrut_T = pd.read_csv(
        "../data/HadCRUT.5.0.2.0_Globalannual_1850-2024.txt",
        sep=",",
        skiprows=1,
        header=None,
    ).T
    hadcrut_T.columns = ["year", "Tsfc"]
    return hadcrut_T.set_index("year")


def open_ipcc_forcing():
    ipcc_forcing = pd.read_csv("../data/ERF_best_aggregates.csv")
    ipcc_forcing.columns.values[0] = "year"
    return ipcc_forcing.set_index("year")


def open_ceres():

    ceres_all_cs = xr.open_dataset(
        "../data/CERES_EBAF-TOA_Ed4.2.1_200003-202601.nc", engine="netcdf4"
    )
    ceres_cre = xr.open_dataset(
        "../data/CERES_EBAF_Ed4.2.1_Subset_200003-202512.nc", engine="netcdf4"
    )
    return xr.merge([ceres_all_cs, ceres_cre], compat="override")


def open_ceres_grid():
    return xr.open_dataset("../data/CERES_EBAF-TOA_gridded.nc", engine="netcdf4")


def open_cmip():
    dsmet = xr.open_dataset("../data/HadAM3.nc")
    dscmip = xr.open_dataset("../data/CMIP.nc")
    return dsmet, dscmip
