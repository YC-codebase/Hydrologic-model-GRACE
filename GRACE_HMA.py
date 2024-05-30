import os
import sys
from datetime import datetime, timedelta
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import pyepsg
import scipy.optimize
from shapely.geometry import Polygon
import xarray as xr
import regionmask
from dask.diagnostics import ProgressBar
import cartopy.crs as ccrs
from himatpy.GRACE_MASCON.pygrace import extract_grace, get_mascon_gdf, get_cmwe_trend_analysis
from himatpy.LIS import utils as LISutils
from geopandas import GeoDataFrame
from typing import List, Tuple, Any

def read_grace_data(fname: str) -> Tuple[dict, GeoDataFrame]:
    f = extract_grace(fname)
    mascon = f['mascon']
    mascon_gdf = get_mascon_gdf(mascon)
    mascon_gdf['mascon'] = mascon_gdf.index
    return f, mascon_gdf

def create_region_gdf(coordinates: List[Tuple[float, float]]) -> GeoDataFrame:
    poly = Polygon(coordinates)
    region_gdf = gpd.GeoDataFrame()
    region_gdf.loc[0, 'geometry'] = poly
    return region_gdf

def get_hma_region_gdf(mascon_gdf: GeoDataFrame, f: dict) -> GeoDataFrame:
    HMA = mascon_gdf[(mascon_gdf['location'] == 80.0) & ((mascon_gdf['basin'] > 5000.0) & (mascon_gdf['basin'] < 6100.0))]
    HMA_w_trend = get_cmwe_trend_analysis(HMA, f)
    return HMA_w_trend

def plot_mass_change(HMA_w_trend: GeoDataFrame) -> None:
    HMA_w_trend.plot(column='avg_mass_change_cm')

def intersect_gdf(HMA_region: GeoDataFrame, HMA_w_trend: GeoDataFrame, world: GeoDataFrame) -> GeoDataFrame:
    gpd_intersect = gpd.overlay(HMA_region, HMA_w_trend, how='intersection')
    gpd_intersect.plot(column='avg_mass_change_cm', ax=world[world['continent'] == 'Asia'].plot(), figsize=(20, 20))
    return gpd_intersect

def create_region_mask(gpd_intersect: GeoDataFrame) -> regionmask.Regions:
    numbers = gpd_intersect.index.values
    names = gpd_intersect['mascon'].values
    abbrevs = gpd_intersect['mascon'].values
    m = regionmask.Regions_cls('HMA_msk', numbers, names, abbrevs, gpd_intersect.geometry.values)
    return m

def plot_region_mask(m: regionmask.Regions) -> None:
    ax = m.plot()
    ax.set_extent([0, 140, 10, 70], crs=ccrs.PlateCarree())
    m.name = 'mascon'

def load_lis_data(datadir: str) -> xr.Dataset:
    ds = xr.open_mfdataset(os.path.join(datadir, '*.nc'))
    ds = ds.chunk({'time': 1})
    return ds

def calculate_mask(m: regionmask.Regions, da: xr.DataArray) -> xr.DataArray:
    m2 = m.mask(da.coords, lon_name='longitude', lat_name='latitude')
    m2.name = 'mask'
    return m2

def save_mask(m2: xr.DataArray, datadir: str) -> None:
    m2.to_netcdf(os.path.join(datadir, 'gracemsk.nc'))

def load_mask(datadir: str) -> xr.DataArray:
    return xr.open_dataset(os.path.join(datadir, 'maskdir', 'gracemsk.nc'))

def plot_mask(m2: xr.DataArray) -> None:
    m2.plot(x='longitude', y='latitude')

def process_lis_data(ds: xr.Dataset, m2: xr.DataArray) -> pd.DataFrame:
    ds2 = ds.groupby(m2).mean(dim='stacked_north_south_east_west')
    ds2.coords['mascon'] = ('mask', names[ds2.coords['mask'].values.astype('int')])
    with ProgressBar():
        df = ds2.to_dataframe()
    return df

def calculate_water_balance(df: pd.DataFrame) -> pd.DataFrame:
    df['waterbal'] = df['Rainf_tavg'] + df['Snowf_tavg'] - (df['Qsb_tavg'] + df['Qsb_tavg'] + df['Qsm_tavg'] + df['Evap_tavg'])
    df['waterbal_cumulative'] = df.groupby(['mask'])['waterbal'].apply(lambda x: x.cumsum())
    return df

def save_dataframe(df: pd.DataFrame, datadir: str) -> None:
    df.to_pickle(os.path.join(datadir, 'LIS_by_mascon.pkl'))

def plot_water_balance(df: pd.DataFrame, mask_value: int) -> None:
    df[df.index.get_level_values('mask') == mask_value].plot(y='waterbal_cumulative')

def get_mascon_time_series(f: dict, soln: dict, n: int) -> pd.DataFrame:
    t = pd.DataFrame(data=f['time']["ref_days_middle"][0, :])
    mc = names[numbers == n][0]
    g = pd.DataFrame(data=soln['cmwe'][mc][:] * 10)  # mm we
    g.index = t[0].apply(lambda x: datetime(2001, 12, 31) + timedelta(days=x))
    return g

def main() -> None:
    fname = '../../files/GRACE/GSFC.glb.200301_201607_v02.3b-ICE6G.h5'
    datadir = '../../LIS/'
    
    f, mascon_gdf = read_grace_data(fname)
    
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    
    # Define coordinates and create a GeoDataFrame for the region
    coordinates = [(62, 26), (62, 46), (106, 46), (106, 26)]
    HMA_region = create_region_gdf(coordinates)
    
    HMA_w_trend = get_hma_region_gdf(mascon_gdf, f)
    plot_mass_change(HMA_w_trend)
    
    gpd_intersect = intersect_gdf(HMA_region, HMA_w_trend, world)
    m = create_region_mask(gpd_intersect)
    plot_region_mask(m)
    
    ds = load_lis_data(datadir)
    da = ds['Snowf_tavg'].isel(time=0)
    
    m2 = calculate_mask(m, da)
    save_mask(m2, datadir)
    m2 = load_mask(datadir)
    plot_mask(m2)
    
    df = process_lis_data(ds, m2)
    df = calculate_water_balance(df)
    save_dataframe(df, datadir)
    plot_water_balance(df, 10)
    
    g = get_mascon_time_series(f, f['solution'], 10)
    g.plot()
    df['waterbal_cumulative'].loc[df.index.get_level_values('mask') == 10].plot(y='waterbal_cumulative')

if __name__ == "__main__":
    main()
