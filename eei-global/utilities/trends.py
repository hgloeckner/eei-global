import numpy as np
import xarray as xr
import utilities.stats as ustats


def calc_fclrcon(da, cloud_fraction):
    return (1 - cloud_fraction / 100) * da


def add_resolved_fclr(ds):
    # keep signs of OLR and Fsw out
    return ds.assign(
        Fclrsw=calc_fclrcon(ds.toa_sw_clr_c_mon, ds.cldarea_total_daynight_mon),
        Fclrlw=calc_fclrcon(ds.toa_lw_clr_c_mon, ds.cldarea_total_daynight_mon),
    )


def add_weighted(ds):
    # weighted cloud and cs contributions as in Loeb et al 2024
    weights = np.cos(np.deg2rad(ds.lat))
    weights = weights.expand_dims({"lon": ds.lon})
    clr = add_resolved_fclr(ds)
    clrlwmean = clr.Fclrlw.weighted(weights).mean(["lat", "lon"])
    clrswmean = clr.Fclrsw.weighted(weights).mean(["lat", "lon"])
    solarmean = ds.solar_mon.weighted(weights).mean(["lat", "lon"])

    return clr.assign(
        Fclrnetmean=solarmean - clrswmean - clrlwmean,
        Fclrlwmean=clrlwmean,
        Fclrswmean=clrswmean,
        solarmean=solarmean,
        Fcldnetmean=ds.toa_net_all_mon.weighted(weights).mean(["lat", "lon"])
        - (solarmean - clrswmean - clrlwmean),
        Fcldswmean=(ds.toa_sw_all_mon).weighted(weights).mean(["lat", "lon"])
        - clrswmean,
        Fcldlwmean=(ds.toa_lw_all_mon).weighted(weights).mean(["lat", "lon"])
        - clrlwmean,
    )


def create_trend_ds(startyears, endyears):
    return xr.Dataset(
        coords=dict(
            rad=[
                "net",
                "net cre",
                "net cs",
                "lw net",
                "lw cs",
                "lw cre",
                "sw net",
                "sw cs",
                "sw cre",
            ],
            value=["mean", "err"],
            startyear=startyears,
            endyear=endyears,
        ),
    )


def add_cmip_to_data(data, dscmip, forcing, newvar, **kwargs):
    ftrend = []
    for startyear in data.startyear.values:
        for endyear in data.endyear.values:
            res = ustats.get_trends_for_cmip(
                dscmip.sel(forcing=forcing), startyear, np.ceil(endyear), **kwargs
            )
            ftrend.append(
                xr.concat(
                    [
                        res[0].mean("model").expand_dims({"value": ["mean"]}),
                        1.645 * res[0].std("model").expand_dims({"value": ["err"]}),
                    ],
                    dim="value",
                ).expand_dims({"startyear": [startyear], "endyear": [endyear]})
            )
    return data.assign(
        **{newvar: 10 * xr.merge(ftrend).to_dataarray().squeeze()}
    ).drop_vars(["variable"])


def add_ceres(data, ceres, alpha):
    ceres_enddates = np.ceil(data.endyear.values)
    ceres_trends = xr.Dataset(
        dict(
            ceres=(
                ("rad", "startyear", "endyear", "value"),
                np.array(
                    [
                        [
                            [
                                ustats.linear_trend(
                                    da.sel(year=slice(startyear, endyear)),
                                    alpha=alpha,
                                    nlags=48,
                                )
                                for endyear in ceres_enddates
                            ]
                            for startyear in data.startyear.values
                        ]
                        for da in [
                            ceres.gtoa_net_all_mon,
                            ceres.gtoa_cre_net_mon,
                            ceres.gtoa_net_clr_c_mon,
                            -ceres.gtoa_lw_all_mon,
                            -ceres.gtoa_lw_clr_c_mon,
                            ceres.gtoa_cre_lw_mon,
                            ceres.gsolar_mon - ceres.gtoa_sw_all_mon,
                            ceres.gsolar_mon - ceres.gtoa_sw_clr_c_mon,
                            ceres.gtoa_cre_sw_mon,
                        ]
                    ]
                ),
                dict(
                    long_name="dN / dt from CERES ",
                    units="W m-2 dec-1",
                ),
            )
        ),
        coords=dict(
            rad=[
                "net",
                "net cre",
                "net cs",
                "lw net",
                "lw cs",
                "lw cre",
                "sw net",
                "sw cs",
                "sw cre",
            ],
            startyear=data.startyear.values,
            endyear=data.endyear.values,
            value=["mean", "err"],
        ),
    )

    return xr.merge([data, ceres_trends * 10])


def add_weighted_ceres(data, ceres, alpha):
    ceres_enddates = np.ceil(data.endyear.values)
    ceres_trends = xr.Dataset(
        dict(
            ceres_weighted=(
                ("rad", "startyear", "endyear", "value"),
                np.array(
                    [
                        [
                            [
                                ustats.linear_trend(
                                    da.sel(year=slice(startyear, endyear)),
                                    alpha=alpha,
                                    nlags=48,
                                )
                                for endyear in ceres_enddates
                            ]
                            for startyear in data.startyear.values
                        ]
                        for da in [
                            ceres.Fcldnetmean,
                            ceres.Fclrnetmean,
                            -ceres.Fclrlwmean,
                            -ceres.Fcldlwmean,
                            ceres.solarmean - ceres.Fclrswmean,
                            -ceres.Fcldswmean,
                        ]
                    ]
                ),
                dict(
                    long_name="dN / dt cld and clr contributions from CERES ",
                    units="W m-2 dec-1",
                ),
            )
        ),
        coords=dict(
            rad=[
                "net cre",
                "net cs",
                "lw cs",
                "lw cre",
                "sw cs",
                "sw cre",
            ],
            startyear=data.startyear.values,
            endyear=data.endyear.values,
            value=["mean", "err"],
        ),
    )
    return xr.merge([data, ceres_trends * 10])


def add_data_trend(data, ds, varname, quantity, alpha, attrs=None):
    return xr.merge(
        [
            data,
            xr.DataArray(
                np.array(
                    [
                        [
                            np.array(
                                ustats.linear_trend(
                                    ds[varname].sel(year=slice(startyear, endyear)),
                                    alpha=alpha,
                                )
                            )
                            * 10
                            for endyear in data.endyear.values
                        ]
                        for startyear in data.startyear.values
                    ]
                ),
                dims=("startyear", "endyear", "value"),
                name=quantity + varname,
                attrs=attrs,
                coords=dict(
                    startyear=data.startyear.values,
                    endyear=data.endyear.values,
                    value=["mean", "err"],
                ),
            ),
        ]
    )


def add_manual(data, factor):
    for varname, values in [
        (
            "lambda_4co2",
            [
                (-0.92, 0.29),
                (0.2, 0.26),
                (-1.12, 0.18),
                (-1.8, 0.31),
                (-1.8, 0.12),
                (0.01, 0.27),
                (0.88, 0.38),
                (0.68, 0.13),
                (0.19, 0.43),
            ],
        ),
        (
            "amiprp",
            [
                (-0.27, 0.07),
                (-0.02, 0.06),
                (-0.25, 0.05),
                (-0.5, 0.1),
                (-0.47, 0.09),
                (-0.03, 0.05),
                (0.23, 0.08),
                (0.21, 0.08),
                (0.02, 0.09),
            ],
        ),
        (
            "cnnrp",
            [
                (-0.35, 0.2),
                (np.nan, np.nan),
                (np.nan, np.nan),
                (-0.44, np.nan),
                (np.nan, np.nan),
                (np.nan, np.nan),
                (0.21, np.nan),
                (np.nan, np.nan),
                (np.nan, np.nan),
            ],
        ),
    ]:
        data = xr.merge(
            [
                data,
                xr.DataArray(
                    [[values]],
                    dims=("startyear", "endyear", "rad", "value"),
                    coords=dict(
                        rad=[
                            "net",
                            "net cre",
                            "net cs",
                            "lw net",
                            "lw cs",
                            "lw cre",
                            "sw net",
                            "sw cs",
                            "sw cre",
                        ],
                        startyear=[2001],
                        endyear=[2023.6],
                        value=["mean", "err"],
                    ),
                    name=varname,
                ),
            ]
        )

        if varname != "lambda_4co2":
            data[varname].loc[dict(value="err")] *= factor
    return data
