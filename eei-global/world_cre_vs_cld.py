# %%

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cf
import utilities.open_data as od
import utilities.stats as ustats
import utilities.trends as trends
import cmocean as cmo

# %%
ceres = od.open_ceres()
ceres = ustats.assign_year(ceres)
ceres_grid = od.open_ceres_grid()
ceres_grid = ustats.assign_year(ceres_grid)
weighted_ceres = trends.add_resolved_fclr(ceres_grid)

startyear = 2001
endyear = 2026

# %%
das = {
    "clrnet": weighted_ceres.solar_mon
    - weighted_ceres.Fclrsw
    - weighted_ceres.Fclrlw,  # solar - SW_clr - OLR_clr
    "clrlw": -weighted_ceres.Fclrlw,  # - OLR_clr
    "clrsw": weighted_ceres.solar_mon - weighted_ceres.Fclrsw,  # solar - SW_clr
    "cldnet": weighted_ceres.toa_net_all_mon
    - (
        weighted_ceres.solar_mon - weighted_ceres.Fclrsw - weighted_ceres.Fclrlw
    ),  # cld net = net - clr net
    "cldlw": -weighted_ceres.toa_lw_all_mon + weighted_ceres.Fclrlw,  # -(OLR - OLR_clr)
    "cldsw": -weighted_ceres.toa_sw_all_mon + weighted_ceres.Fclrsw,
}  # -(sw_all - sw_clr)

weight = {}
for name, da in das.items():
    weight[name] = ustats.get_trends_for_cmip(
        da.rename(name).to_dataset(),
        startyear=startyear,
        endyear=endyear,
        varname=name,
        alpha=0.975,
        nlags=48,
        calc_cierr=False,
    )[0]


# %%
das = {
    "allnet": ceres_grid.toa_net_all_mon,
    "allsw": ceres_grid.solar_mon - ceres_grid.toa_sw_all_mon,
    "alllw": -ceres_grid.toa_lw_all_mon,
    "clrnet": ceres_grid.toa_net_clr_c_mon,
    "clrlw": -ceres_grid.toa_lw_clr_c_mon,
    "clrsw": ceres_grid.solar_mon - ceres_grid.toa_sw_clr_c_mon,
    "cldnet": ceres_grid.toa_net_all_mon - ceres_grid.toa_net_clr_c_mon,
    "cldlw": -ceres_grid.toa_lw_all_mon + ceres_grid.toa_lw_clr_c_mon,
    "cldsw": -ceres_grid.toa_sw_all_mon + ceres_grid.toa_sw_clr_c_mon,
}

standard = {}
for name, da in das.items():
    standard[name] = ustats.get_trends_for_cmip(
        da.rename(name).to_dataset(),
        startyear=startyear,
        endyear=endyear,
        varname=name,
        alpha=0.975,
        nlags=48,
        calc_cierr=False,
    )[0]


# %%

fraction_trend = ustats.get_trends_for_cmip(
    ceres_grid,
    startyear=startyear,
    endyear=endyear,
    varname="cldarea_total_daynight_mon",
    alpha=0.975,
    nlags=48,
    calc_cierr=False,
)
# %%

# %%
kwargs = dict(
    vmin=-5,
    vmax=5,
    cmap="cmo.balance",
    add_colorbar=False,
    rasterized=True,
    transform=ccrs.PlateCarree(),
)
scene = "clr"

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(12, 4.5),
    subplot_kw={"projection": ccrs.Robinson(central_longitude=-135.58)},
    dpi=300,
)


for idx, label in enumerate([f"{scene}net", f"{scene}lw", f"{scene}sw"]):
    (standard[label] * 10).plot(ax=axes[0, idx], **kwargs)
    global_mean = (
        standard[label].weighted(np.cos(np.deg2rad(standard[label].lat))).mean().values
        * 10
    )
    axes[0, idx].text(0.01, 0.9, f"{global_mean:.2f}", transform=axes[0, idx].transAxes)
    im = (weight[label] * 10).plot(ax=axes[1, idx], **kwargs)
    axes[1, idx].text(
        0.01,
        0.9,
        f"{(weight[label] * 10).weighted(np.cos(np.deg2rad(weight[label].lat))).mean().values:.2f}",
        transform=axes[1, idx].transAxes,
    )


for ax in axes.flatten():
    ax.coastlines()
fig.tight_layout()
cax = fig.add_axes([0.27, 0.001, 0.5, 0.03])
fig.colorbar(
    im,
    cax=cax,
    orientation="horizontal",
    extend="both",
    label="dN/dt / W m$^{-2}$ dec$^{-1}$",
)
for ax, label in zip(axes[0], ["net", "lw", "sw"]):
    ax.set_title(f"{label}")
fig.savefig(f"../graphics/weighted_{scene}_trends.pdf", bbox_inches="tight", dpi=300)
# %%

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(12, 4.5),
    subplot_kw={"projection": ccrs.Robinson(central_longitude=-135.58)},
    dpi=300,
)
scene = "all"
for idx, label in enumerate([f"{scene}net", f"{scene}lw", f"{scene}sw"]):
    (standard[label] * 10).plot(ax=axes[0, idx], **kwargs)
    global_mean = (
        standard[label].weighted(np.cos(np.deg2rad(standard[label].lat))).mean().values
        * 10
    )
    axes[0, idx].text(0.01, 0.9, f"{global_mean:.2f}", transform=axes[0, idx].transAxes)

for ax in axes.flatten():
    ax.coastlines()
fig.tight_layout()
# %%
fontsize = 16
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    subplot_kw={"projection": ccrs.Robinson(central_longitude=-135.58)},
    figsize=(14, 8),
)
rad = "sw"

for rowidx, source in enumerate([standard, weight]):
    for colidx, scene in enumerate(["clr", "cld"]):
        pltds = source[scene + rad] * 10
        im = pltds.plot(ax=axes[rowidx, colidx], **kwargs)
        global_mean = (
            pltds.weighted(np.cos(np.deg2rad(standard[label].lat))).mean().values
        )
        axes[rowidx, colidx].text(
            0.9,
            0.9,
            f"{global_mean:.2f}",
            fontsize=fontsize - 2,
            transform=axes[rowidx, colidx].transAxes,
        )

axes[0, 0].set_title("Clear-Sky", fontsize=fontsize)
axes[0, 1].set_title("CRE", fontsize=fontsize)
axes[1, 0].set_title("Clear-Sky Con", fontsize=fontsize)
axes[1, 1].set_title("Cloud Con", fontsize=fontsize)


fig.tight_layout()
ax_pos = axes[0, 0].get_position()
ax_frac = fig.add_axes(
    [0.05 - ax_pos.width, 0.5 - ax_pos.height / 2, ax_pos.width, ax_pos.height],
    projection=ccrs.Robinson(central_longitude=-135.58),
)
frac = fraction_trend[0].plot(
    ax=ax_frac,
    add_colorbar=False,
    rasterized=True,
    transform=ccrs.PlateCarree(),
    vmin=-0.5,
    vmax=0.5,
    cmap="cmo.diff_r",
)

cax_frac = fig.add_axes([-0.35, 0.25, 0.3, 0.02])
cbar = fig.colorbar(frac, cax=cax_frac, orientation="horizontal", extend="max")
cbar.set_label("cloud fraction trend / dec$^{-1}$", size=fontsize - 1)

cax_cld = fig.add_axes([0.35, 0.01, 0.3, 0.02])
cbar = fig.colorbar(im, cax=cax_cld, orientation="horizontal", extend="both")
cbar.set_label(r"$N_{\mathrm{" + rad + "}}$ / dec$^{-1}$", size=fontsize - 1)

for ax in axes.flatten():
    ax.coastlines()
ax_frac.coastlines()
fig.savefig(
    f"../graphics/weighted_clr_cld_{rad}_trends.pdf", bbox_inches="tight", dpi=300
)
# %%
for idx, label in enumerate([f"{scene}net", f"{scene}lw", f"{scene}sw"]):
    axes[0, idx].plot(
        np.sin(np.deg2rad(standard[label].lat)),
        (standard[label] * 10).mean("lon").values,
    )
for axidx, scene in enumerate(["clr", "cld"]):
    for idx, label in enumerate([f"{scene}net", f"{scene}lw", f"{scene}sw"]):
        for dictionary, name in [(standard, "cs/cre"), (weight, "cs/cld")]:
            axes[1 + axidx, idx].plot(
                np.sin(np.deg2rad(dictionary[label].lat)),
                (dictionary[label] * 10).mean("lon").values,
                label=name,
            )

axes[0, 0].set_xticks(
    axes[0, 0].get_xticks()[1:-1],
    labels=np.round(np.rad2deg(np.arcsin(axes[0, 0].get_xticks()[1:-1])), 1),
)
axes[1, 0].legend()
for ax in axes.flatten():
    ax.axhline(0, color="k", ls="--", lw=0.5)
for ax, title in zip(axes[0], ["net", "lw", "sw"]):
    ax.set_title(title)
for ax, ylabel in zip(axes[:, 0], ["total", "clear-sky", "cloud"]):
    ax.set_ylabel(ylabel)

sns.despine()
