# %%
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import seaborn as sns
import utilities.open_data as od
import utilities.stats as ustats


# %%

berkeleyT = od.open_berkeley().to_xarray()
berkeleyT = ustats.assign_year(berkeleyT)
hadcrutT = od.open_hadcrut().to_xarray()
ipccF = od.open_ipcc_forcing().to_xarray()
ceres = od.open_ceres()
ceres = ustats.assign_year(ceres)

_, cmip = od.open_cmip()
cmip = ustats.assign_year(cmip)
# %%
cmip_rtrend = []
startyear = 2001
endyear = 2024
cmip_ftrend = []
res = ustats.get_trends_for_cmip(
    cmip.sel(forcing="piClim-histall"), startyear, np.ceil(endyear)
)
cmip_ftrend.append(
    xr.concat(
        [
            res[0].mean("model").expand_dims({"value": ["mean"]}),
            1.645 * res[0].std("model").expand_dims({"value": ["err"]}),
        ],
        dim="value",
    ).expand_dims({"startyear": [startyear], "endyear": [endyear]})
)
rfmipf = xr.merge(cmip_ftrend).to_dataarray().squeeze().drop_vars(["variable"])
# %%
alpha = 0.95
startyear = 2001
endyear = 2026
cw = 190 / 25.4
fig, axes = plt.subplots(figsize=(cw, 0.6 * cw), nrows=3, sharex=True)

for da, idx, color, label, unit in [
    # (-ceres.gtoa_lw_all_mon, 0, "orangered", "N$_{lw}$", "W m$^{-2}$"),
    # (ceres.gsolar_mon - ceres.gtoa_sw_all_mon, 0, "teal", "N$_{sw}$", "W m$^{-2}$"),
    (ceres.gtoa_net_all_mon, 0, "k", "N", "W m$^{-2}$"),
    (ipccF.total, 1, "#003E74", "F", "W m$^{-2}$"),
    (berkeleyT.monthly_anomaly, 2, "#B9D532", "T", "K"),
]:
    da = ustats.normalize_by_climatology(da.sel(year=slice(startyear, endyear)))
    reg, err = ustats.linear_trend(da, alpha=alpha, nlags=48, only_slope=False)
    axes[idx].plot(
        da.year,
        da.values,
        color=color,
    )
    axes[idx].plot(
        da.year,
        da.year * reg.slope + reg.intercept,
        color=color,
        lw=1,
        label=f"d{label} / dt: {reg.slope * 10:.2f} $\pm$ {err * 10:.2f} "
        + f"{unit} dec$^{{-1}}$",
    )
    axes[idx].set_ylabel(f"$\Delta$ {label} / {unit}")
    axes[idx].yaxis.set_major_formatter(FuncFormatter(lambda x, p: "{:.1f}".format(x)))


lines = (
    cmip.swap_dims({"time": "year"})
    .sel(rad="net", forcing="piClim-histall", year=slice(startyear, endyear))
    .N.plot.line(
        ax=axes[1],
        x="year",
        color="grey",
        alpha=0.5,
        lw=0.5,
        ls="-",
        add_legend=False,
    )
)
lines[0].set_label(
    f"RFMIP dF / dt : {rfmipf.sel(rad='net', value='mean').values * 10:.2f} $\pm$ {rfmipf.sel(rad='net', value='err').values * 10:.2f} W m$^{{-2}}$ dec$^{{-1}}$"
)
for ax in axes:
    ax.legend()
axes[1].set_title("")
axes[1].set_xlabel("")
axes[1].set_ylabel("F / W m$^{-2}$")
axes[2].set_xlabel("Year")
sns.despine(offset=5)
# fig.savefig("../graphics/trend_overview.pdf", bbox_inches="tight")
# %%
norm_ceres = (
    ustats.normalize_by_climatology(ceres)
    .swap_dims({"time": "year"})
    .sel(year=slice(2001, 2026))
)


fig, axes = plt.subplots(figsize=(cw, 0.5 * cw), nrows=3, sharex=True, sharey=True)


for idx, (da, color, label) in enumerate(
    [
        (norm_ceres.gtoa_net_all_mon, "black", "$N$"),
        (-norm_ceres.gtoa_lw_all_mon, "orangered", r"$N_\mathrm{lw}$"),
        (
            norm_ceres.gsolar_mon - norm_ceres.gtoa_sw_all_mon,
            "teal",
            r"$N_\mathrm{sw}$",
        ),
    ]
):
    ax = axes[idx]
    da.plot(ax=ax, color=color)
    reg, uncertainty = ustats.linear_trend(da, alpha=alpha, nlags=48, only_slope=False)
    ax.plot(
        norm_ceres.year,
        reg.intercept + reg.slope * norm_ceres.year,
        color=color,
        label=f"trend: {reg.slope * 10:.2f} ± {uncertainty * 10:.2f} W m$^{{-2}}$ dec$^{{-1}}$",
    )

    ax.set_ylabel(label + " / W m$^{-2}$")
    ax.set_xlabel("")
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.legend()
axes[2].set_xlabel("Year")
sns.despine()
fig.savefig("../graphics/ceres_trends.pdf", bbox_inches="tight")
