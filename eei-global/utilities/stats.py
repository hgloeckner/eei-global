import scipy.stats as sstats
import numpy as np
from scipy.stats import t
import statsmodels.api as sm
import warnings
import xarray as xr


def assign_year(ds):
    return ds.assign_coords(
        year=ds.time.dt.year + (ds.time.dt.dayofyear - 1) / ds.time.dt.days_in_year
    )


def normalize_by_climatology(ds):
    if "time" in ds.dims:
        ds = ds.groupby("time.month") - ds.groupby("time.month").mean()
        return assign_year(ds)
    else:
        return ds


def linear_trend_np(years, values, alpha, nlags, only_slope=True, calc_cierr=True):
    reg = sstats.linregress(years, values)
    if calc_cierr:
        ei = values - reg.intercept - reg.slope * years
        n = len(years)
        autocorr = sm.tsa.acf(ei, nlags=nlags)
        sigmak = np.sqrt(1 / n * (1 + 2 * np.cumsum(autocorr[1:] ** 2)))
        cik = sigmak * [t.ppf(0.8, n - k) for k in range(1, len(sigmak) + 1)]
        maxauto = np.argmax((autocorr[:-1] < 0) & ((autocorr[1:] + autocorr[:-1]) < 0))
        if maxauto == 0:
            print("autocorrelation is positive at all lags, increase number of lags!")
        if np.any(autocorr[1:] > cik):
            Neff = n / (1 + 2 * np.sum(np.abs(autocorr[1:maxauto])))
        else:
            print("no significant autocorrelation, using n for effective sample size")
            Neff = n
        se = np.sqrt(np.sum(ei**2) / (Neff - 2))
        sb = se / np.sqrt(np.sum((years - years.mean()) ** 2))
        tstat = t.ppf(alpha, Neff - 2)
        if only_slope:
            return reg.slope, tstat * sb
        else:
            return reg, tstat * sb
    else:
        if only_slope:
            return reg.slope, np.nan
        else:
            return reg


def linear_trend(da, calc_cierr=True, alpha=0.975, nlags=25, only_slope=True):
    Ndet = normalize_by_climatology(da)
    if np.any(np.isnan(Ndet.values)):
        warnings.warn(
            "NaN values found in data - probably not covering the full period, dropping NaN values for linear regression.\n"
        )
        Ndet = Ndet.dropna("year")
    return linear_trend_np(
        Ndet.year,
        Ndet.values,
        alpha,
        nlags,
        only_slope=only_slope,
        calc_cierr=calc_cierr,
    )


def trend_unc_from_meas_unc(uncsigma):
    w = 1 / uncsigma**2
    return np.sqrt(1 / (np.sum(w * (uncsigma.year - uncsigma.year.mean()) ** 2)))


def get_trends_for_cmip(
    ds, startyear, endyear, varname="N", alpha=0.975, nlags=48, calc_cierr=True
):
    ds = normalize_by_climatology(ds)
    return xr.apply_ufunc(
        linear_trend_np,
        ds.sel(year=slice(startyear, endyear)).year,
        ds.sel(year=slice(startyear, endyear))[varname],
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],
        kwargs=dict(alpha=alpha, nlags=nlags, calc_cierr=calc_cierr),
        vectorize=True,
    )
