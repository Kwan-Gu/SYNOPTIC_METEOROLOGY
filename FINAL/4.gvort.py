import numpy as np
from netCDF4 import Dataset
from metpy.units import masked_array as ma
from metpy.calc import potential_temperature as pt

lonmin = 110.
lonmax = 160.
latmin = -10.
latmax = 30.

ncf = Dataset('./ght.20131105-20131107.500.nc','r')
lon = ncf.variables['longitude'][:]
lat = ncf.variables['latitude'][:]
ycc = np.squeeze(np.where((lat>latmin)&(lat<latmax)))
xcc = np.squeeze(np.where((lon>lonmin)&(lon<lonmax)))
lat = lat[ycc]
lon = lon[xcc]
data = ncf.variables['z'][:,ycc,xcc]
nt,ny,nx = np.shape(data)

lats, lons = np.zeros((2,ny,nx), dtype='float32')

for j in range(ny):
	lons[j,:] = lon
for i in range(nx):
	lats[:,i] = lat

latrad = np.deg2rad(lats)
sf = np.sin(latrad)
f = sf*2.*np.pi/86400. #planetary vorticity

dx = 2*np.pi*6400000.*np.cos(latrad)/1440.  ##radius of earth = 6400km
dy = np.ones((ny,nx),dtype='float32')*2*np.pi*6400000./1440.

###finite difference
def fdx(dat):
	rst = np.zeros((nt,ny,nx), dtype='float32')
	rst[:,:,0] = np.divide((dat[:,:,1]-dat[:,:,0]), dx[:,0][None,:])
	rst[:,:,-1] = np.divide((dat[:,:,-1]-dat[:,:,-2]), dx[:,-1][None,:])
	rst[:,:,1:-1] = np.divide((dat[:,:,2:]-dat[:,:,:-2]), dx[:,1:-1][None,:,:])/2.
	return rst

def fdy(dat):
	rst = np.zeros((nt,ny,nx), dtype='float32')
	rst[:,0,:] = np.divide((dat[:,1,:]-dat[:,0,:]), dy[0][None,None,:])
	rst[:,-1,:] = np.divide((dat[:,-1,:]-dat[:,-2,:]), dy[-1][None,None,:])
	rst[:,1:-1,:] = np.divide((dat[:,2:,:]-dat[:,:-2,:]), dy[1:-1,:][None,:,:])/2.
	return rst

ug = (-1.)*np.divide(fdy(data), f[None,:,:]) 
vg = np.divide(fdx(data), f[None,:,:])

gvort = fdx(vg) - fdy(ug)

file = open('gvort.20131105-20131107.hourly.500hPa.npy','wb')
np.save(file, gvort)
file.close()

file = open('longitude.npy','wb')
np.save(file, lons)
file.close()

file = open('latitude.npy','wb')
np.save(file, lats)
file.close()

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
vmin = -0.005
vmax = 0.02
dv = 0.0025

gvort[np.where(gvort<vmin)] = vmin
gvort[np.where(gvort>vmax)] = vmax

fig, ax = plt.subplots(figsize=(10,6))
plt.title('20131106_12utc', fontsize=15)
m = Basemap(projection='cyl', llcrnrlat=latmin, urcrnrlat=latmax, llcrnrlon=lonmin, urcrnrlon=lonmax, resolution='l')
m.drawcoastlines(linewidth=0.7)
m.drawcountries(linewidth=0.7)
m.drawparallels(np.arange(19)*5.+latmin, linewidth=0.3, labels=[1,0,0,0])
m.drawmeridians(np.arange(19)*10.+lonmin, linewidth=0.3,  labels=[0,0,0,1])
x,y = m(lons,lats)
m.contourf(x,y,gvort[36],levels=np.arange(vmin,vmax+dv,dv))#, cmap=plt.cm.coolwarm)
cb = plt.colorbar(orientation='horizontal',shrink = 0.5, pad=0.1,
label = 'Geostrophic relative vorticity')
plt.savefig('gvort.131106.png')
plt.show()





