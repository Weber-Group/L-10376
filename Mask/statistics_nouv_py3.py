import psana
import numpy as np

## Script for doing analysis of front and back detectors, and seeing the kapton ring, for building masks.

experiment = 'cxix1000021'
run = 100
Nevents =100000 #the maximimum number of shots you want to analyze
ds = psana.MPIDataSource('exp=%s:run=%d' % (experiment, run)) #parallelized "small data" object, best to read the psana docs
front = psana.Detector('jungfrau4M', ds.env()) #the front detector
frontmask = np.ones((8,512,1024))
#frontmask = np.load('/reg/d/psdm/cxi/cxilu9218/scratch/nag1647/fbase_mask.npy') # mask for front detector, can use a real mask once we make one
#lower_bound = -30. # lower bound for ADU per pixel
evr = psana.Detector('evr0')
diode1 = psana.Detector('HFX-DG2-BMMON', ds.env())
#diode2 = Detector('CXI-DG3-BMMON', ds.env())
fee       = psana.Detector('FEEGasDetEnergy', ds.env())
smldata = ds.small_data('/sdf/data/lcls/ds/cxi/cxix1000021/scratch/nag1647/Masks/run%d_%d_stats.h5' %(run, Nevents/1000.), gather_interval=800)

# EVR codes
LASERON		= 183
LASEROFF	= 184
XRAYOFF		= 162
XRAYOFF1	= 163
XRAYON          = 137

#Prepare variables, counting shots and saving images
xray_shots = 0
dark_shots = 0
xray_front = np.zeros_like(frontmask)
dark_front = np.zeros_like(frontmask)

def safe_get(det, evt):
        try:
                return det.get(evt)
        except Exception:
                return None

#looping over the shots
for n, evt in enumerate(ds.events()):
        ds.break_after(Nevents)
        if evt is None: 
                print('evt')
                continue
        img = front.calib_data(evt) # #grabbing detector images
        evt_diode1 = safe_get(diode1, evt)
#        evt_diode2 = safe_get(diode2, evt)
        if evt_diode1 is None:
                print('diode')
                continue
        diode1_intensity = evt_diode1.TotalIntensity()
#	diode2_intensity = evt_diode2.TotalIntensity()
        fee_data_pull = safe_get(fee, evt)
        if fee_data_pull is None:
                print('fee')
                continue
        fee_data = fee_data_pull.f_21_ENRC()
#	evt_pressure = sample_pressure(evt)	
#ipm_data = IPM(evt)
	#sometimes a bad shot will be saved, without any images
        if img is None: 
                print('img')
                continue
        img *= frontmask	
        evrcodes = evr(evt)
        if evrcodes is None:
                print('evrcodes')
                continue
#	dark = XRAYOFF in evrcodes
        dark = XRAYON not in evrcodes
        if not dark:
                xray_front += img # * (img > lower_bound)
                xray_shots += 1
                smldata.event(front_intensity = img.sum(), diode1_intensity=diode1_intensity, xray_energy = fee_data)
#                smldata.event(fee_data = fee_data)
#                smldata.event(ipm_data = ipm_data)
        elif dark:
                dark_front += img  # * float(img > lower_bound)
                dark_shots += 1
# summing up the images and shots across ranks
xrayshots = smldata.sum(xray_shots)
darkshots = smldata.sum(dark_shots)
front_xrays = smldata.sum(xray_front)
front_dark = smldata.sum(dark_front)
# saving the summed values
smldata.save(xray_front=front_xrays, dark_front=front_dark, xray_shots=xrayshots, dark_shots=darkshots)

