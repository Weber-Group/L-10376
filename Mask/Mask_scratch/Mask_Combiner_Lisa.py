import numpy as np

dark_run_1 = 123
#dark_run_2 = 14
blank_run_1 = 125
#blank_run_2 = 14

mask_off_1 = np.load('run%d_mask_xray.npy' % dark_run_1) #mask made with the dark run 1
#mask_off_2 = np.load('run%d_mask_xray.npy' % dark_run_2) #mask made with the dark run 2
mask_on_1 = np.load('run%d_mask_xray.npy' % blank_run_1) #mask made with the blank run 1
#mask_on_2 = np.load('run%d_mask_xray.npy' % blank_run_2) #mask made with the blank run 2

#mask_tot = mask_off_1*mask_off_2*mask_on_1*mask_on_2
mask_tot = mask_off_1*mask_on_1
print(np.sum(mask_tot))

#np.save('Mask_Jungfrau_%d_%d_%d_%d_T_Edge.npy' % (dark_run_1, dark_run_2, blank_run_1, blank_run_2), mask_tot) 
np.save('Mask_Jungfrau_%d_%d_T_Edge.npy' % (dark_run_1, blank_run_1), mask_tot)
