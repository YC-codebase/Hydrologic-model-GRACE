
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal, stats
import geopandas as gpd
import pandas as pd
import xarray as xr
import salem
from PyAstronomy import pyasl
import dask.array as da
from dask.diagnostics import ProgressBar
import boto3
from typing import Tuple, List, Dict, Any
import warnings
from himatpy.GRACE_MASCON.pygrace import extract_grace, get_mascon_gdf, trend_analysis, get_cmwe_trend_analysis, select_mascons, aggregate_mascons

warnings.filterwarnings('ignore')

def load_dgws(datadir: str) -> xr.Dataset:
    return xr.open_mfdataset(datadir + '*.nc') * 0.1

def load_dgws1(datadir: str) -> Tuple[xr.DataArray, xr.DataArray]:
    dgws1 = xr.open_dataset(datadir + 'LISMonthly_new.nc') * 0.1
    lat = dgws1.latitude.drop('longitude')[:,-1].rename({'north_south': 'lat'})
    long = dgws1.longitude.drop('latitude')[-1,:].rename({'east_west': 'long'})
    return lat, long

def preprocess_dgws(dgws: xr.Dataset, lat: xr.DataArray, long: xr.DataArray) -> xr.Dataset:
    dgws = dgws.drop(['lat', 'lon']).rename({'north_south': 'lat', 'east_west': 'long'})
    dgws = dgws.assign_coords(lat=lat, long=long, time=dgws.coords['time'])
    to_drop = [dim for dim in dgws.data_vars if dgws[dim].dims != ('time', 'lat', 'long')]
    dgws = dgws.drop(to_drop)
    return dgws

def filter_and_resample_lis(lis: xr.Dataset) -> xr.Dataset:
    lis = lis.sel(long=slice(65, 100), lat=slice(21, 35)).resample(time='1m').mean(dim='time')
    return lis

def calculate_anomalies(lis: xr.Dataset) -> xr.Dataset:
    monthly_mean = lis.groupby('time.month').mean('time')
    anomalies = lis.groupby('time.month') - monthly_mean
    return anomalies

def load_shapefile(fp: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    well_d = gpd.read_file(fp)
    shdf = salem.read_shapefile(fp)
    return well_d, shdf

def load_grace_data(fname: str) -> Tuple[Dict[str, Any], gpd.GeoDataFrame]:
    f = extract_grace(fname)
    mascon_gdf = get_mascon_gdf(f['mascon'])
    mascon_gdf['mascon'] = mascon_gdf.index
    return f, mascon_gdf

def select_and_aggregate_mascons(lis: xr.Dataset, mascon_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    masked_gdf4 = select_mascons(lis, mascon_gdf)
    agg_data = aggregate_mascons(lis, masked_gdf4)
    A3 = agg_data['data']
    index = pd.MultiIndex.from_product([agg_data[n] for n in ['products', 'mascons', 'time']], names=['products', 'mascons', 'time'])
    df4 = pd.DataFrame({'A3': A3.flatten()}, index=index)['A3']
    df4 = df4.unstack(level='products').swaplevel().sort_index()
    df4.columns = agg_data['products']
    df4.index.names = ['date', 'mascon']
    dfok4 = df4.dropna(how='any')
    return dfok4

def calculate_grace_time_series(f: Dict[str, Any], agg_data: Dict[str, Any]) -> pd.DataFrame:
    t = pd.DataFrame(data={'time': f['time']["ref_days_middle"][0,:]})
    grc_lst = []
    for m in agg_data['mascons']:
        df = pd.DataFrame(data={'cmwe': f['solution']['cmwe'][m][:], 'mascon': m})
        df.index = t['time'].apply(lambda x: datetime(2001, 12, 31) + timedelta(days=x))
        grc_lst.append(df)
    gracedf = pd.concat(grc_lst)
    return gracedf

def plot_time_series(gracedf: pd.DataFrame, dfok4: pd.DataFrame) -> None:
    grct = gracedf.groupby('time').mean()
    lisst = dfok4.groupby('date').mean()
    gok1 = grct.reset_index()
    gok1['time'] = pd.to_datetime(gok1['time']).dt.floor('d')
    Aok = lisst.reset_index()
    Aok['date'] = pd.to_datetime(Aok['date'])
    
    fig, ax = plt.subplots(1, figsize=(10, 8))
    gok1.plot(ax=ax, y='cmwe', x='time', linewidth=2, color='brown')
    Aok.plot(ax=ax, y='GWS_tavg', x='date', linewidth=2, color='blue')
    plt.ylabel('Ground Water Storage Anomaly (cm)', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.xticks(rotation=0)
    ax.legend(['GRACE', 'LIS'], fontsize=14, loc=1)
    plt.show()

def plot_trends(gracedf: pd.DataFrame, dfok4: pd.DataFrame, f: Dict[str, Any]) -> None:
    grct = gracedf.groupby('time').mean()
    lisst = dfok4.groupby('date').mean()
    GRACE_mass = grct.cmwe.values
    GRACE_decyear = f['time']['yyyy_doy_yrplot_middle'][2, :]
    LIS = lisst.GWS_tavg.values
    LIS_dy1 = pd.Series(lisst.index).apply(lambda x: pyasl.decimalYear(x))
    
    plt.figure(figsize=(10, 8))
    plt.plot(GRACE_decyear, signal.detrend(GRACE_mass, type='linear'), color='brown', linewidth=2)
    plt.plot(LIS_dy1, signal.detrend(LIS, type='linear'), color='blue', linewidth=2)
    plt.ylabel('Ground Water Storage Anomaly (cm)', fontsize=14)
    plt.xlabel('Year', fontsize=14)
    plt.legend(['GRACE', 'LIS'], fontsize=14, loc=1)
    plt.show()

def plot_grace_trends(masked_gdf4: gpd.GeoDataFrame, shdf: gpd.GeoDataFrame, f: Dict[str, Any]) -> None:
    t_grace = get_cmwe_trend_analysis(masked_gdf4, f)
    t_grace_cl = gpd.overlay(shdf, t_grace, how='intersection')
    
    plt.figure(figsize=(12, 8))
    t_grace_cl.plot(column='avg_mass_change_cm', cmap='viridis', legend=True)
    plt.ylabel('Latitude', fontsize=12)
    plt.xlabel('Longitude', fontsize=12)
    plt.show()

def main() -> None:
    datadir = '/mnt/c/Users/HCD/UW_work/git/HiMAT/files/LIS/new/original_NASA/original_monthly/'
    datadir2 = '/mnt/c/Users/HCD/UW_work/git/HiMAT/files/LIS/files/'
    shapefile_path = "/mnt/c/Users/HCD/UW_work/git/HiMAT/files/WBM/Watershed_boundaries/dugwells_districts_d.shp"
    grace_file = '../../files/GRACE/GSFC.glb.200301_201607_v02.4-ICE6G.h5'
    
    # Load and preprocess datasets
    dgws = load_dgws(datadir)
    lat, long = load_dgws1(datadir2)
    lis = preprocess_dgws(dgws, lat, long)
    
    # Filter and resample LIS data
    lisr = filter_and_resample_lis(lis)
    lisa = calculate_anomalies(lisr)
    
    # Load shapefiles
    well_d, shdf = load_shapefile(shapefile_path)
    
    # Load GRACE data
    f, mascon_gdf = load_grace_data(grace_file)
    
    # Select and aggregate mascons
    dfok4 = select_and_aggregate_mascons(lisa, mascon_gdf)
    
    # Calculate GRACE time series
    gracedf = calculate_grace_time_series(f, {'mascons': mascon_gdf.index})
    
    # Plot time series and trends
    plot_time_series(gracedf, dfok4)
    plot_trends(gracedf, dfok4, f)
    
    # Plot GRACE trends with shapefiles
    plot_grace_trends(mascon_gdf, shdf, f)

if __name__ == "__main__":
    main()