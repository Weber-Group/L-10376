import h5py
from psana import *
import matplotlib.pyplot as plt
import numpy as np

class maskmaker(object):

	def __init__(self, expt, run, detector, filename, oldmask):
		self.run = run
		self.ds = DataSource('exp=%s:run=%d' %(expt, run))
		self.oldmask = oldmask
		self.evt0 = next(self.ds.events())

		hf = h5py.File(filename, 'r')
		for item in hf:
			print('%s' %str(item))            
			exec('%s = np.array(hf[item])' %str(item))
#		self.fee = fee_data
#		self.ibm = ibm_data

		if detector == 'front':
		        self.det = Detector('jungfrau4M')
		        self.xray = np.array(hf['xray_front'])/np.array(hf['xray_shots'])    
		        self.dark = np.array(hf['dark_front'])/np.array(hf['dark_shots'])
		        self.intensity = np.array(hf['front_intensity'])         
#		        try:
#		            self.intensity = np.array(hf['front_intensity'])
#		        except Exception:
#		            self.intensity = np.zeros(int(hf['xray_shots']))

		if detector == 'back':
		    self.det = Detector('DsdCsPad')
		    self.xray = xray_back/xray_shots
		    self.dark = dark_back/dark_shots
		    self.intensity = back_intensity 

		X = self.det.coords_x(self.evt0)
		Y = self.det.coords_y(self.evt0)
		self.rr = np.sqrt((X**2) + (Y**2)) * 1e-6

		return

	def __call__(self, img, upperbound, lowerbound, tolerance, name):
		if img == 'dark':	image = self.dark
		if img == 'xray': image = self.xray
		#newmask = (image < upperbound)
		newmask = (image < upperbound) * (image > lowerbound) * self.oldmask
	
		image *= newmask
		mean = np.average(image[image != 0])
		sigma = np.std(image[image != 0])
		ub = mean + tolerance*sigma
		lb = mean - tolerance*sigma
		newmask *= (image > lb) * (image < ub)
		np.save(name, newmask)
		return newmask


