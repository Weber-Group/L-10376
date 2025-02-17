import numpy as np

dark_run = 418
blank_run = 429

mask_off = np.load('run%d_mask_xray.npy' % dark_run) #mask made with the dark run 
mask_on = np.load('run%d_mask_xray.npy' % blank_run) #mask made with the blank run 
line_mask = np.load('line_mask.npy')
shadow_mask = np.load('Mask_Jungfrau_Shadow.npy')

mask_tot = mask_off*mask_on*line_mask*shadow_mask
print(np.sum(mask_tot)/(8*512*1024))

np.save('Mask_Jungfrau_%d_%d.npy' % (dark_run_1, blank_run_1), mask_tot)
