import xarray as xr
import datetime as dt
import numpy as np
import metpy.calc as mpcalc
import metpy.units as units
from scipy.ndimage import gaussian_filter

def kinematics(u, v, thetae, dx, dy, lats, smooth=False, sigma=1):

	'''
    Use metpy functions to calculate various kinematics, given 2d numpy arrays as inputs. X and y grid spacing, as well as a 2d array of latitudes is also required.
    
    Option to smooth the thetae field before calculating front diagnostics (recommended), with addition "sigma" parameter controlling the smoothness (see scipy docs)
    '''

	if smooth:
		thetae = gaussian_filter(thetae, sigma)
    
	#Kinematics
	ddy_thetae = mpcalc.first_derivative( thetae, delta=dy, axis=0)
	ddx_thetae = mpcalc.first_derivative( thetae, delta=dx, axis=1)
	mag_thetae = np.sqrt( ddx_thetae**2 + ddy_thetae**2)
	div = mpcalc.divergence(u, v, dx, dy)
	strch_def = mpcalc.stretching_deformation(u, v, dx, dy)
	shear_def = mpcalc.shearing_deformation(u, v, dx, dy)
	tot_def = mpcalc.total_deformation(u, v, dx, dy)
	psi = 0.5 * np.arctan2(shear_def, strch_def)
	beta = np.arcsin((-ddx_thetae * np.cos(psi) - ddy_thetae * np.sin(psi)) / mag_thetae)
	vo = mpcalc.vorticity(u, v, dx, dy)
	conv = -div * 1e5
	F = np.array(0.5 * mag_thetae * (tot_def * np.cos(2 * beta) - div) * 1.08e4 * 1e5)
	Fn = np.array(0.5 * mag_thetae * (div - tot_def * np.cos(2 * beta) ) * 1.08e4 * 1e5)
	Fs = np.array(0.5 * mag_thetae * (vo + tot_def * np.sin(2 * beta) ) * 1.08e4 * 1e5)
	icon = np.array(0.5 * (tot_def - div) * 1e5)
	vgt = np.array(np.sqrt( div**2 + vo**2 + tot_def**2 ) * 1e5)

	#TFP
	ddx_thetae_scaled = ddx_thetae / mag_thetae
	ddy_thetae_scaled = ddy_thetae / mag_thetae
	ddy_mag_thetae = -1 * mpcalc.first_derivative( mag_thetae, delta=dy, axis=0)
	ddx_mag_thetae = -1 * mpcalc.first_derivative( mag_thetae, delta=dx, axis=1)
	tfp = ddx_mag_thetae * ddx_thetae_scaled + ddy_mag_thetae * ddy_thetae_scaled
	ddy_tfp = mpcalc.first_derivative( tfp, delta=dy, axis=0)
	ddx_tfp = mpcalc.first_derivative( tfp, delta=dx, axis=1)
	mag_tfp = np.sqrt( ddx_tfp**2 + ddy_tfp**2 )
	v_f = (u * units.units("m/s")) * (ddx_tfp / mag_tfp) + (v * units.units("m/s")) * (ddy_tfp / mag_tfp)
    
    #Extra condition
	ddy_ddy_mag_te = mpcalc.first_derivative( -1*ddy_mag_thetae, delta=dy, axis=0)
	ddx_ddx_mag_te = mpcalc.first_derivative( -1*ddx_mag_thetae, delta=dx, axis=1)
	cond = ddy_ddy_mag_te + ddx_ddx_mag_te
    
	return [F, Fn, Fs, icon, vgt, conv, vo*1e5, \
            np.array(tfp.to("km^-2")*(100*100)), np.array(mag_thetae.to("km^-1"))*100, np.array(v_f), thetae, cond]

def era5_eg3_read(time):
    
	'''
    Read ERA5 data using xarray, which has been downloaded from the CDS in netcdf. Output is 3d numpy arrays (time x lon x lat) of 
    U, V, specific humidity, air temp. Also output 1d arrays of long and lat, as well as a list of datetime objects describing each time step
    '''

	f = xr.open_dataset("/g/data/eg3/ab4502/era5.nc").sel({"time":slice(time[0], time[1])})
	u = f["u"].values
	v = f["v"].values
	q = f["q"].values
	t = f["t"].values
	lon = f.longitude.values
	lat = f.latitude.values
	date_list = f.time.values

	return [u, v, q, t, lon, lat, date_list]

def calc_fronts(u, v, q, t, lon, lat, date_list):

	'''
    Parse era5 variables to kinematics(), to calculate various thermal and kinematic front parameters. 
    12 are computed, but for brevity, only four are returned for now.
    '''
    
	#Using MetPy, derive equivalent potential temperature
	ta_unit = units.units.K*t
	dp_unit = mpcalc.dewpoint_from_specific_humidity(q*units.units.dimensionless, ta_unit, 850 * units.units.hectopascals)
	thetae = np.array(mpcalc.equivalent_potential_temperature(850 * units.units.hectopascals, ta_unit, dp_unit))

	#From the lat-lon information accompanying the ERA5 data, resconstruct 2d coordinates and grid spacing (dx, dy)
	x, y = np.meshgrid(lon,lat)
	dx, dy = mpcalc.lat_lon_grid_deltas(x,y)

	#Derive various kinematic/thermal diagnostics for each time step
	kinemats = [ kinematics(u[i], v[i], thetae[i], dx, dy, y, smooth=True, sigma=2) for i in np.arange(len(date_list)) ]
	F = [kinemats[i][0] for i in np.arange(len(date_list))]	       #Frontogenesis function (degC / 100 km / 3 hr)
	Fn = [kinemats[i][1] for i in np.arange(len(date_list))]	   #Frontogenetical function (deg C / 100 km / 3 hr)
	Fs = [kinemats[i][2] for i in np.arange(len(date_list))]	   #Rotational component of frontogensis (deg C/ 100 km / 3 hr)
	icon = [kinemats[i][3] for i in np.arange(len(date_list))]	   #Instantaneous contraction rate (s^-1 * 1e5)
	vgt = [kinemats[i][4] for i in np.arange(len(date_list))]	   #Horizontal velovity gradient tensor magnitude (s^-1 * 1e5)
	conv = [kinemats[i][5] for i in np.arange(len(date_list))]	   #Convergence (s^-1 * 1e5)
	vo = [kinemats[i][6] for i in np.arange(len(date_list))]	   #Relative vorticity (s^-1 * 1e5)
	tfp = [kinemats[i][7] for i in np.arange(len(date_list))]	   #Thermal front parameter (km^-2)
	mag_te = [kinemats[i][8] for i in np.arange(len(date_list))]   #Magnitude of theta-e gradient (100 km ^-1)
	v_f = [kinemats[i][9] for i in np.arange(len(date_list))]      #Advection of TFP (m/s)
	thetae = [kinemats[i][10] for i in np.arange(len(date_list))]  #Theta-e (K), may be smoothed depending on arguments
	cond = [kinemats[i][11] for i in np.arange(len(date_list))]    #Extra condition

	return [thetae, mag_te, tfp, v_f]    