# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import utilities.open_data as od
import utilities.stats as ustats
import utilities.trends as trends
# %%


berkeleyT = od.open_berkeley().to_xarray()
berkeleyT = ustats.assign_year(berkeleyT)
hadcrutT = od.open_hadcrut().to_xarray()
ipccF = od.open_ipcc_forcing().to_xarray()
ceres = od.open_ceres()
ceres = ustats.assign_year(ceres)
ceres_grid = od.open_ceres_grid()
ceres_grid = ustats.assign_year(ceres_grid)
weighted_ceres = trends.add_weighted(ceres_grid)

dsmet, dscmip = od.open_cmip()
dsmet = ustats.assign_year(dsmet)
dscmip = ustats.assign_year(dscmip)

# %%

alpha = 0.95  # 90% confidence interval
factor = 1.645  # factor for sigma for 90% confidence interval

data = trends.create_trend_ds(startyears=[2001, 2005], endyears=[2023.6, 2025.6])
data = trends.add_cmip_to_data(data, dscmip, forcing="piClim-histall", newvar="rfmipf")
data = trends.add_ceres(data, ceres, alpha)
data = trends.add_weighted_ceres(data, weighted_ceres, alpha)
data = trends.add_data_trend(
    data,
    ipccF,
    varname="total",
    quantity="F",
    alpha=alpha,
    attrs=dict(long_name="dF / dt from ERF forcing", units="W m-2 dec-1"),
)
data = trends.add_data_trend(
    data,
    ipccF,
    varname="aerosol",
    quantity="F",
    alpha=alpha,
    attrs=dict(long_name="dF / dt from ERF aerosol forcing", units="W m-2 dec-1"),
)
data = trends.add_data_trend(
    data,
    berkeleyT,
    varname="monthly_anomaly",
    quantity="T",
    alpha=alpha,
    attrs=dict(long_name="dT / dt from Berkeley", units="K dec-1"),
)
data = trends.add_data_trend(
    data,
    hadcrutT,
    varname="Tsfc",
    quantity="T",
    alpha=alpha,
    attrs=dict(long_name="dT / dt from HadCRUT", units="K dec-1"),
)
data = trends.add_manual(data, factor)

sherwood_lambda = xr.DataArray(
    np.array([[-1.3, np.nan], [-1.9, np.nan], [0.6, np.nan]]),
    dims=["rad", "value"],
    coords=dict(
        rad=["net", "lw net", "sw net"],
        value=["mean", "err"],
    ),
)
data = data.assign(
    theoryr=sherwood_lambda
    * data.Tmonthly_anomaly.expand_dims({"rad": sherwood_lambda.rad})
)

# %%
eps = 0.08
nxticks = 4
npanels = 3
nx = (nxticks - 1) * npanels
colors = {
    "net": "k",
    "lw net": "orangered",
    "sw net": "teal",
    "net cre": "k",
    "lw cre": "orangered",
    "sw cre": "teal",
    "net cs": "k",
    "lw cs": "orangered",
    "sw cs": "teal",
}
startyear = 2001
endyear = 2023.6
labels = dict(
    ceres=r"CERES $\frac{\mathrm{d}N}{\mathrm{d}t}$",
    Ftotal=r"ERF (IPCC AR6) $\frac{\mathrm{d}F}{\mathrm{d}t}$",
    rfmipf=r"RFMIP $\frac{\mathrm{d}F}{\mathrm{d}t}$, AMIP $\frac{\mathrm{d}R}{\mathrm{d}t}$",
    cnnrp=r"CNN $\frac{\mathrm{d}R}{\mathrm{d}t}$",
    amiprp="",
    theoryr=r"Sherwood $\lambda \frac{\mathrm{d}T}{\mathrm{d}t}$",
    ceres_weighted=r"CERES weighted Loeb et al (2024)",
)


kwargs = dict(
    linestyle="",
    markersize=6,
    markeredgewidth=1,
)
ekwargs = dict(
    elinewidth=1,
    linestyle="",
    markeredgewidth=1,
    markersize=6,
)

nxticks = 4
npanels = 1
nx = (nxticks - 1) * npanels

cw = 190 / 25.4
sns.set_context("paper")
fig, axes = plt.subplots(figsize=(cw, cw), nrows=3, sharex=True, sharey=True)

# plot CERES N, RFMIP F and AMIP R+P trends
for varname, marker, facecolor, xidx, alpha, xshift in [
    ("ceres", "s", None, 1, 1, -0.03),
    ("ceres_weighted", "s", "white", 1, 1, 0.03),
    ("rfmipf", None, None, 2, 1, 0),
    ("amiprp", None, None, 3, 1, 0),
    ("cnnrp", "o", None, 3, 0.5, 0.03),
    ("theoryr", "P", None, 3, 1, -0.03),
]:
    for idx, rad in enumerate(
        [
            "net",
            "lw net",
            "sw net",
            "net cre",
            "lw cre",
            "sw cre",
            "net cs",
            "lw cs",
            "sw cs",
        ]
    ):
        if idx == 3:
            label = labels[varname]
        else:
            label = ""
        ax = axes[idx // 3]
        ax.errorbar(
            idx % 3 + xidx / (nxticks + 1) + xshift,
            data[varname]
            .sel(rad=rad, value="mean")
            .sel(startyear=startyear, endyear=endyear),
            data[varname]
            .sel(rad=rad, value="err")
            .sel(startyear=startyear, endyear=endyear),
            marker=marker,
            color=colors[rad],
            alpha=alpha,
            markerfacecolor=facecolor,
            label=label,
            **ekwargs,
        )

for idx, cld in enumerate(["net", "cre", "cs"]):
    ax = axes[idx]
    for radidx, rad in enumerate(["net", "lw", "sw"]):
        if cld == "net" and rad == "net":
            varname = "net"
        else:
            varname = f"{rad} {cld}"
        err = np.sqrt(
            data.ceres.sel(
                rad=varname, value="err", startyear=startyear, endyear=endyear
            )
            ** 2
            + data.rfmipf.sel(
                rad=varname, value="err", startyear=startyear, endyear=endyear
            )
            ** 2
            + data.amiprp.sel(
                rad=varname, value="err", startyear=startyear, endyear=endyear
            )
            ** 2
        )
        ax.errorbar(
            (radidx + 1 - 1 / (nxticks + 1)),
            (data.ceres - data.rfmipf - data.amiprp)
            .sel(rad=f"{varname}", value="mean")
            .sel(startyear=startyear, endyear=endyear),
            err,
            marker="",
            color=colors[varname],
            **ekwargs,
        )
    if cld == "net":
        cld = "total"
    elif cld == "cre":
        cld = "CRE / cloud"
    else:
        cld = "clear-sky"

    ax.set_title(cld, loc="left", fontsize=12, pad=5)

varname, marker, facecolor, xidx, alpha, xshift = ("Ftotal", "D", None, 2, 1, -0.03)
axes[0].errorbar(
    xidx / (nxticks + 1) + xshift,
    data[varname].sel(value="mean").sel(startyear=startyear, endyear=endyear),
    data[varname].sel(value="err").sel(startyear=startyear, endyear=endyear),
    marker=marker,
    color=colors["net"],
    alpha=alpha,
    markerfacecolor=facecolor,
    label=label,
    **ekwargs,
)

for ax in axes:
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    for i in range(9):
        if i % 3 == 0:
            color = "k"
            lw = 0.0  # 7
        else:
            lw = 0.5
            color = "grey"
        ax.axvline(i, color=color, lw=lw, ls="-")

    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    ax.xaxis.set_tick_params(width=0.5, length=5)
    ax.yaxis.set_tick_params(width=0.5, length=5)
    ax.tick_params(axis="x", which="major", labelsize=14)
    ax.tick_params(axis="y", which="major", labelsize=10)
    ax.set_ylabel("W m$^{-2}$ dec$^{-1}$")

ax = axes[0]
ax.set_xlim(0, nx)
ax.set_xticks(
    np.concatenate(
        [i + (1 / (nxticks + 1)) * np.arange(1, (nxticks + 1)) for i in range(nx)]
    ),
    labels=[
        r"$\frac{\mathrm{d}N}{\mathrm{d}t}$",
        r"$\frac{\mathrm{d}F}{\mathrm{d}t}$",
        # "$\dot{T}$",
        # r"$\lambda\frac{\mathrm{d}T}{\mathrm{d}t}$",
        r"$\frac{\mathrm{d}R}{\mathrm{d}t}$",
        r"$\mathrm{r}$",
    ]
    * nx,
    fontsize=14,
)
for idx, (text, c) in enumerate(
    [
        ("Net", colors["net cs"]),
        ("Longwave", colors["lw cs"]),
        ("Shortwave", colors["sw cs"]),
    ]
):
    for i in range(npanels):
        ax.text(
            x=i * 3 + idx + 0.5,
            y=1.15,
            s=text,
            horizontalalignment="center",
            color=c,
            fontsize=12,
        )
sns.despine()

axes[1].legend(
    fontsize=10, bbox_to_anchor=(1.15, 1.26), ncols=2, frameon=True, loc="upper right"
)


ax.fill_between(
    [0, 1],
    -eps,
    eps,
    color="grey",
    alpha=0.2,
)
fig.savefig("../graphics/NFRr_rows.pdf", bbox_inches="tight")


# %%
kwargs = dict(
    linestyle="",
    markersize=4,
    markeredgewidth=1,
)
ekwargs = dict(
    elinewidth=0.8,
    linestyle="",
    markeredgewidth=1,
    markersize=4,
)

fig, ax = plt.subplots(figsize=(cw, 0.5 * cw))


# plot CERES N, RFMIP F and AMIP R+P trends
for varname, marker, facecolor, xidx, alpha, xshift in [
    ("ceres", "s", None, 1, 1, 0),
    ("rfmipf", None, None, 2, 1, 0),
    ("amiprp", None, None, 3, 1, 0),
    ("cnnrp", "o", None, 3, 0.5, 0.03),
    ("theoryr", "P", None, 3, 1, -0.03),
]:
    for idx, rad in enumerate(
        [
            "net",
            "lw net",
            "sw net",
            "net cre",
            "lw cre",
            "sw cre",
            "net cs",
            "lw cs",
            "sw cs",
        ]
    ):
        if idx == 0:
            label = labels[varname]
        else:
            label = ""
        ax.errorbar(
            idx + xidx / (nxticks + 1) + xshift,
            data[varname]
            .sel(rad=rad, value="mean")
            .sel(startyear=startyear, endyear=endyear),
            data[varname]
            .sel(rad=rad, value="err")
            .sel(startyear=startyear, endyear=endyear),
            marker=marker,
            color=colors[rad],
            alpha=alpha,
            label=label,
            **ekwargs,
        )

varname, marker, facecolor, xidx, alpha, xshift = ("Ftotal", "D", None, 2, 1, -0.03)
ax.errorbar(
    xidx / (nxticks + 1) + xshift,
    data[varname].sel(value="mean").sel(startyear=startyear, endyear=endyear),
    data[varname].sel(value="err").sel(startyear=startyear, endyear=endyear),
    marker=marker,
    color=colors["net"],
    alpha=alpha,
    label=labels[varname],
    **ekwargs,
)
# plot residual
for idx, cld in enumerate(["net", "cre", "cs"]):
    for radidx, rad in enumerate(["net", "lw", "sw"]):
        if cld == "net" and rad == "net":
            varname = "net"
        else:
            varname = f"{rad} {cld}"
        err = np.sqrt(
            data.ceres.sel(
                rad=varname, value="err", startyear=startyear, endyear=endyear
            )
            ** 2
            + data.rfmipf.sel(
                rad=varname, value="err", startyear=startyear, endyear=endyear
            )
            ** 2
            + data.amiprp.sel(
                rad=varname, value="err", startyear=startyear, endyear=endyear
            )
            ** 2
        )
        print(
            data.amiprp.sel(
                rad=varname, value="err", startyear=startyear, endyear=endyear
            ).values
        )
        print(err.values)
        ax.errorbar(
            idx * 3 + (radidx + 1 - 1 / (nxticks + 1)),
            (data.ceres - data.rfmipf - data.amiprp)
            .sel(rad=f"{varname}", value="mean")
            .sel(startyear=startyear, endyear=endyear),
            err,
            marker="",
            color=colors[varname],
            **ekwargs,
        )
    if cld == "net":
        cld = "total"
    elif cld == "cre":
        cld = "CRE"
    else:
        cld = "clear-sky"

    ax.text((idx * 3) + 1.5, 1.25, cld, ha="center")


ax.set_xlim(0, nx)
ax.legend()
ax.axhline(0, color="grey", lw=0.5, ls=":")
for i in range(9):
    if i % 3 == 0:
        color = "k"
        lw = 0.7
    else:
        lw = 0.5
        color = "grey"
    ax.axvline(i, color=color, lw=lw, ls="-")

ax.set_xticks(
    np.concatenate(
        [i + (1 / (nxticks + 1)) * np.arange(1, (nxticks + 1)) for i in range(nx)]
    ),
    labels=[
        r"$\frac{\mathrm{d}N}{\mathrm{d}t}$",
        r"$\frac{\mathrm{d}F}{\mathrm{d}t}$",
        # "$\dot{T}$",
        # r"$\lambda\frac{\mathrm{d}T}{\mathrm{d}t}$",
        r"$\frac{\mathrm{d}R}{\mathrm{d}t}$",
        r"$\mathrm{r}$",
    ]
    * nx,
    fontsize=8,
)


for idx, (text, c) in enumerate(
    [
        ("Net", colors["net cs"]),
        ("Longwave", colors["lw cs"]),
        ("Shortwave", colors["sw cs"]),
    ]
):
    for i in range(npanels):
        ax.text(
            x=i * 3 + idx + 0.5,
            y=1.15,
            s=text,
            horizontalalignment="center",
            color=c,
            fontsize=8,
        )

ax.legend(fontsize=8, loc="upper right", ncols=2)
ax.tick_params(axis="both", which="major", labelsize=8)
ax.set_ylabel("W m$^{-2}$ dec$^{-1}$")
sns.despine()

ax.spines["bottom"].set_linewidth(0.5)
ax.spines["left"].set_linewidth(0.5)
ax.xaxis.set_tick_params(width=0.5, length=5)
ax.yaxis.set_tick_params(width=0.5, length=5)
ax.fill_between(
    [0, 1],
    -eps,
    eps,
    color="grey",
    alpha=0.2,
)
