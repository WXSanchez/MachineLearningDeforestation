from sklearn.cluster import KMeans
import gdal
import numpy as np

naip_fn = 'path/to/image.tif' #select image and read in to use with gdal
driverTiff = gdal.GetDriverByName('GTiff')
naip_ds = gdal.Open(naip_fn)
nbands = naip_ds.RasterCount
data = np.empty((naip_ds.RasterXSize*naip_ds.RasterYSize, nbands))

for i in range(1, nbands+1): #create an empty array to hold the bands and loop through the bands, adding each to the array
    band = naip_ds.GetRasterBand(i).ReadAsArray()
    data[:, i-1] = band.flatten()

km = KMeans(n_clusters=7) #setup kmeans 
km.fit(data)
km.predict(data)

out_dat = km.labels_.reshape((naip_ds.RasterYSize, naip_ds.RasterXSize)) #map the predictions onto the original image

clfds = driverTiff.Create('path/to/classified.tif', naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_Float32) #save the original image
clfds.SetGeoTransform(naip_ds.GetGeoTransform())
clfds.SetProjection(naip_ds.GetProjection())
clfds.GetRasterBand(1).SetNoDataValue(-9999.0)
clfds.GetRasterBand(1).WriteArray(out_dat)
clfds = None