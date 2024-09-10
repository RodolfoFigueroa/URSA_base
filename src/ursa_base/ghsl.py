import os
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray as rxr
import ursa_base.utils as utils

from shapely.geometry import shape


def download_s3(
    bbox,
    ds,
    data_path=None,
    resolution=1000,
    s3_path="GHSL/",
    bucket="tec-expansion-urbana-p",
):
    """Downloads a GHSL windowed rasters for each available year.

    Takes a bounding box (bbox) and downloads the corresponding rasters from a
    the global COG stored on Amazon S3. Returns a single multiband raster,
    a band per year.

    Parameters
    ----------
    bbox : Polygon
        Shapely Polygon defining the bounding box.
    ds : str
        Data set to download, can be one of SMOD, BUILT_S, POP, or LAND.
    resolution : int
        Resolution of dataset to download, either 100 or 1000.
    data_path : Path
        Path to directory to store rasters.
        If none, don't write to disk.
    s3_dir : str
        Relative path to COGs on S3.
    bucket : str

    Returns
    -------
    raster : rioxarray.DataArray
        In memory raster.

    """

    assert ds in ["SMOD", "BUILT_S", "POP", "LAND"], "Data set not available."

    print(f"Downloading {ds} rasters ...")

    s3_path = f"{s3_path}/GHS_{ds}/"

    if ds == "LAND":
        year_list = [2018]
        fname = f"GHS_{ds}_E{{}}_GLOBE_R2022A_54009_{resolution}_V1_0.tif"
    else:
        fname = f"GHS_{ds}_E{{}}_GLOBE_R2023A_54009_{resolution}_V1_0.tif"
        year_list = list(range(1975, 2021, 5))

    array_list = []
    for year in year_list:
        subset, profile = utils.raster.np_from_bbox_s3(
            s3_path + fname.format(year), bbox, bucket, nodata_to_zero=True
        )
        array_list.append(subset)
    ghs_full = np.concatenate(array_list)

    # Create rioxarray
    profile["count"] = ghs_full.shape[0]
    tmp_name = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
    with rio.open(tmp_name, "w", **profile) as dst:
        dst.write(ghs_full)
    raster = rxr.open_rasterio(tmp_name)

    # Rename band dimension to reflect years
    raster.coords["band"] = year_list

    if data_path is not None:
        raster.rio.to_raster(data_path / f"GHS_{ds}_{resolution}.tif")

    print(f"Done: GHS_{ds}_{resolution}.tif")

    return raster


def load_or_download(
    bbox,
    ds,
    data_path=None,
    resolution=1000,
    s3_path="GHSL/",
    bucket="tec-expansion-urbana-p",
):
    """Searches for a GHS dataset to load, if not available,
    downloads it from S3 and loads it.

    Parameters
    ----------
    bbox : Polygon
        Shapely Polygon defining the bounding box.
    ds : str
        Data set to download, can be one of SMOD, BUILT_S, POP, or LAND.
    resolution : int
        Resolution of dataset to download, either 100 or 1000.
    data_path : Path
        Path to directory to store rasters.
        If none, don't write to disk.
    s3_dir : str
        Relative path to COGs on S3.
    bucket : str

    Returns
    -------
    raster : rioxarray.DataArray
        In memory raster.

    """
    fpath = data_path / f"GHS_{ds}_{resolution}.tif"
    if fpath.exists():
        raster = rxr.open_rasterio(fpath)
        if ds != "LAND":
            raster.coords["band"] = list(range(1975, 2021, 5))
        else:
            raster.coords["band"] = [2018]
    else:
        raster = download_s3(bbox, ds, data_path, resolution, s3_path, bucket)

    return raster


def clip_dataset(ds, polygons):
    ds = ds.rio.set_nodata(0)
    ds = ds.rio.clip(polygons)
    return ds


def load_plot_datasets(bbox_mollweide, path_cache, clip=False):
    smod = load_or_download(
        bbox_mollweide,
        "SMOD",
        data_path=path_cache,
        resolution=1000,
    )
    built = load_or_download(
        bbox_mollweide,
        "BUILT_S",
        data_path=path_cache,
        resolution=100,
    )
    pop = load_or_download(
        bbox_mollweide,
        "POP",
        data_path=path_cache,
        resolution=100,
    )

    if clip:
        smod = clip_dataset(smod, [bbox_mollweide])
        built = clip_dataset(built, [bbox_mollweide])
        pop = clip_dataset(pop, [bbox_mollweide])

    return smod, built, pop


def smod_polygons(smod, centroid):
    """Find SMOD polygons for urban centers and urban clusters.

    Parameters
    ----------
    smod : xarray.DataArray
        DataArray with SMOD raster data.
    centroid : shapely.Point
        Polygons containing centroid will be identified as
        the principal urban center and cluster.
        Must be in Mollweide proyection.

    Returns
    -------
    smod_polygons : GeoDataFrame
        GeoDataFrame with polygons for urban clusters and centers.
    """

    # Get DoU lvl 1 representation (1: rural, 2: cluster, 3: center)
    smod_lvl_1 = smod // 10

    smod_centers = (smod_lvl_1 == 3).astype(smod.dtype)
    smod_clusters = (smod_lvl_1 > 1).astype(smod.dtype)

    transform = smod.rio.transform()

    dict_list = []
    for year in smod["band"].values:
        centers = rio.features.shapes(
            smod_centers.sel(band=year).values, connectivity=8, transform=transform
        )
        clusters = rio.features.shapes(
            smod_clusters.sel(band=year).values, connectivity=8, transform=transform
        )

        center_list = [shape(f[0]) for f in centers if f[1] > 0]
        cluster_list = [shape(f[0]) for f in clusters if f[1] > 0]

        center_dicts = [
            {
                "class": 3,
                "year": year,
                "is_main": centroid.within(center),
                "geometry": center,
            }
            for center in center_list
        ]
        cluster_dicts = [
            {
                "class": 2,
                "year": year,
                "is_main": centroid.within(cluster),
                "geometry": cluster,
            }
            for cluster in cluster_list
        ]
        dict_list += center_dicts
        dict_list += cluster_dicts

    smod_polygons = gpd.GeoDataFrame(dict_list, crs=smod.rio.crs)

    return smod_polygons


def built_s_polygons(built):
    """Returns a polygon per pixel for GHS BUILT rasters."""

    resolution = built.rio.resolution()
    pixel_area = abs(np.prod(resolution))

    built_df = built.to_dataframe(name="b_area").reset_index()
    built_df = built_df.rename(columns={"band": "year"})
    built_df = built_df.drop(columns="spatial_ref")

    built_df = built_df[built_df.b_area > 0].reset_index(drop=True)

    built_df["fraction"] = built_df.b_area / pixel_area
    built_df["geometry"] = built_df.apply(
        utils.raster.row2cell, res_xy=resolution, axis=1
    )

    built_gdf = gpd.GeoDataFrame(built_df, crs=built.rio.crs).drop(columns=["x", "y"])

    return built_gdf


def get_urb_growth_df(smod, built, pop, centroid_mollweide, path_cache):
    built.rio.set_nodata(0)
    pop.rio.set_nodata(0)

    smod_gdf = smod_polygons(smod, centroid_mollweide)
    smod_gdf["Area"] = smod_gdf.area

    clusters_gdf = smod_gdf[smod_gdf["class"] == 2]

    main_cluster = clusters_gdf[clusters_gdf.is_main]

    # Total built-up area and pop per year
    # Built raster contains squared meters
    total_built = built.sum(axis=(1, 2)).values
    total_pop = pop.sum(axis=(1, 2)).values

    # Built and pop within center and cluster
    years = smod.coords["band"].values
    cluster_built = []
    cluster_pop = []
    cluster_built_all = []
    cluster_pop_all = []

    for year in years:
        # Check if main cluster is empty
        if main_cluster[main_cluster.year == year].empty:
            cluster_built.append(0)
            cluster_pop.append(0)
            cluster_built_all.append(0)
            cluster_pop_all.append(0)
        else:
            # Main cluster and center
            cluster = main_cluster[main_cluster.year == year].geometry.iloc[0]

            # All clusters and centers
            cluster_all = clusters_gdf[clusters_gdf.year == year].geometry

            # Series for main cluster and center
            cluster_built.append(
                np.nansum(
                    built.sel(band=year)
                    .rio.set_nodata(0)
                    .rio.clip([cluster], crs=built.rio.crs)
                    .values
                )
            )

            cluster_pop.append(
                np.nansum(
                    pop.sel(band=year)
                    .rio.set_nodata(0)
                    .rio.clip([cluster], crs=pop.rio.crs)
                    .values
                )
            )

            # Series for ALL clusters and centers
            cluster_built_all.append(
                np.nansum(
                    built.sel(band=year)
                    .rio.set_nodata(0)
                    .rio.clip(cluster_all, crs=built.rio.crs)
                    .values
                )
            )

            cluster_pop_all.append(
                np.nansum(
                    pop.sel(band=year)
                    .rio.set_nodata(0)
                    .rio.clip(cluster_all, crs=pop.rio.crs)
                    .values
                )
            )

    cluster_built = np.array(cluster_built)
    cluster_pop = np.array(cluster_pop)
    cluster_built_all = np.array(cluster_built_all)
    cluster_pop_all = np.array(cluster_pop_all)

    # Identify year that are not in main, i.e. years without a main cluster
    years_not_int_main = set(years) - set(main_cluster.year.values)

    # Create dummy dataframe with years without a main cluster with 'Area' set to zero
    dummy_dataframe = pd.DataFrame(years_not_int_main, columns=["year"])
    dummy_dataframe["Area"] = 0

    # Complete clusters_gdf and main_cluster with dummy_dataframe
    clusters_gdf = pd.concat([clusters_gdf, dummy_dataframe], ignore_index=True)
    main_cluster = pd.concat([main_cluster, dummy_dataframe], ignore_index=True)

    # Urban area for main cluster and center
    cluster_area = main_cluster.sort_values("year").Area.values

    # Total cluster and center area
    t_cluster_area = clusters_gdf.groupby("year").Area.sum().values

    df = pd.DataFrame(
        {
            "year": smod.coords["band"].values,
            "built_all": total_built / 1e6,
            "built_cluster_main": cluster_built / 1e6,
            "built_cluster_all": cluster_built_all / 1e6,
            "built_cluster_other": (cluster_built_all - cluster_built) / 1e6,
            "built_rural": (total_built - cluster_built) / 1e6,
            "urban_cluster_all": t_cluster_area / 1e6,
            "urban_cluster_main": cluster_area / 1e6,
            "urban_cluster_other": (t_cluster_area - cluster_area) / 1e6,
            "pop_total": total_pop,
            "pop_cluster_main": cluster_pop,
            "pop_cluster_all": cluster_pop_all,
            "pop_cluster_other": (cluster_pop_all - cluster_pop),
            "pop_rural": (total_pop - cluster_pop_all) / 1e6,
            "built_density_cluster_main": cluster_built / cluster_area,
            "built_density_cluster_all": cluster_built_all / t_cluster_area,
            "built_density_cluster_other": (
                (cluster_built_all - cluster_built) / (t_cluster_area - cluster_area)
            ),
            "pop_density_cluster_main": cluster_pop / (cluster_area / 1e6),
            "pop_density_cluster_all": cluster_pop_all / (t_cluster_area / 1e6),
            "pop_density_cluster_other": (
                (cluster_pop_all - cluster_pop)
                / ((t_cluster_area - cluster_area) / 1e6)
            ),
            "pop_b_density_cluster_main": cluster_pop / (cluster_built / 1e6),
            "pop_b_density_cluster_all": (cluster_pop_all / (cluster_built_all / 1e6)),
            "pop_b_density_cluster_other": (
                (cluster_pop_all - cluster_pop)
                / ((cluster_built_all - cluster_built) / 1e6)
            ),
        }
    )

    df.to_csv(path_cache / "urban_growth.csv")

    return df
