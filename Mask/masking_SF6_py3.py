import h5py
from psana import *
import matplotlib.pyplot as plt
import numpy as np

class maskmaker_SF6(object):

	def __init__(self, expt, run, filename, oldmask, x0, y0, z0, xrayEnergy, qbinsize):
		self.run = run
		self.ds = DataSource('exp=%s:run=%d' %(expt, run))
		self.oldmask = oldmask
		self.evt0 = next(self.ds.events())

		hf = h5py.File(filename, 'r')
		for item in hf:
			print(item)
			exec('%s = np.array(hf[item])' %str(item))

		self.det = Detector('jungfrau4M')
		self.xray = np.array(hf['xray_front'])/np.array(hf['xray_shots'])    
		self.dark = np.array(hf['dark_front'])/np.array(hf['dark_shots'])
		self.intensity = np.array(hf['front_intensity'])   


		X = self.det.coords_x(self.evt0)
		Y = self.det.coords_y(self.evt0)
		self.rr = np.sqrt((X+x0)**2 + (Y+y0)**2) * 1e-6
		distance = z0*1e-6
		wavelength = 12400./xrayEnergy #Angstroms
		theta = np.arctan2(self.rr, distance)/2.0
		self.q = 4*np.pi*np.sin(theta)/wavelength
		qmin = np.min(self.q)
		qmax = np.max(self.q)
		self.qbins = np.arange(qmin, qmax + qbinsize, qbinsize) 
		return

#	def __call__(self, img, upperbound, lowerbound, upperbound_signal, lowerbound_signal, name, inds):
	def __call__(self, img, upperbound_signal, lowerbound_signal, name, inds):
		if img == 'dark':	image = self.dark
		if img == 'xray':	image = self.xray
		if img == 'SF6':	image = self.xray
		newmask = np.ones_like(image)
#		mask_ring = (image[inds] < upperbound) * (image[inds] > lowerbound) + (image[inds] < upperbound_signal) * (image[inds] > lowerbound_signal)
		mask_ring = (image[inds] < upperbound_signal) * (image[inds] > lowerbound_signal)
		newmask[inds] = mask_ring
		newmask *= self.oldmask
		np.save(name, newmask)
		return newmask


