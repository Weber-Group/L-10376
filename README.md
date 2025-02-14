# L-10376
 Code for the L-10376 beamtime

## Masking process for filtered regions of the detector
1. Using ubuntu, or another terminal that works with X11 forwarding, ssh into the cluster using the -Y tag to enable X11 forwarding
```bash
ssh -Y <username>@s3dflogin.slac.stanford.edu
```
then
```bash
ssh -Y psana
```
2. Navigate to the smalldata_tools folder within the experiemental directory
```bash
cd /sdf/data/lcls/ds/cxi/cxil1037623/results/smalldata_tools/
```
3. Run the following command, replacing the run number to a run with scattering data, so that you can see the masked portions of the detector. This will open up a ipython session where you will run a few more commands.
```bash
./producers/runSmallDataAna -e cxil1037623 -r <run number>
```
4. Run the following command to generate an average detector image from the run
```python
anaps.AvImage('jungfrau4M')
```
5. Run the following command to enter the mask masking script. GUIs should pop up on your screen if the X11 forwarding is set up properly. Instructions will appear in the terminal, and the averaged image will show up.
```python
anaps.MakeMask()
```
6. For this experiment, we will want to generate a mask using the polygon mode. Look at the image, and then estimate how many points you will need for the polygon masking. You can add more polygons to this mask in a later step. You can close the first plot Press p, then [Enter]. A new plot will show up, and enter the number of points of the polygon into the terminal. Then, using your mouse, select the points in the plot. Once the specified number of points has been added, the points will be displayed in the terminal, and a new plot will show up with the masked region. The last plot cam be closed. The terminal will ask you if you want to add this mask, to which you can reply y/n. It will ask you then if you are done with this specific mask. You can add more shapes to this mask if need be. Enter y/n accordingly.
7. Once you are done with the mask, it will ask if you want to save it within the calibration directory under the experimental directory. Enter n for this. We will be saving to local. By default, it will save the mask to the directory from which you ran the runSmallDataAna file, typically the smalldata_tools folder within the experimental directory. It will save the mask to a file called:
```bash
/sdf/data/lcls/ds/cxi/cxil1037623/results/smalldata_tools/Mask_AvImg_pedSub_jungfrau4M_cxil1037623_Run<run_number>.data
```
This is basically a text file which contains the mask you made. Rename this file to the appropriate name representing the region that is unmasked. We will need to make another mask for the other region, so re-run these steps to create the mask for the other region. Save it accoridngly.
8. Lastly, we need to add these masks to the producer file, which is at the location
```bash
/sdf/data/lcls/ds/cxi/cxil1037623/results/smalldata_tools/producers/smd.producer.py
```
Find the code within the producer file that looks something like this:
```python
    try:
        #output from anaps.MakeMask w/ saving to local
        examplemask = np.genfromtxt('/sdf/data/lcls/ds/cxi/cxil1015022/results/smalldata_tools/Mask_AvImg_pedSub_jungfrau4M_cxil1015022_Run036_dim.data').astype(bool)
        examplemask2 = np.genfromtxt('/sdf/data/lcls/ds/cxi/cxil1015022/results/smalldata_tools/Mask_AvImg_pedSub_jungfrau4M_cxil1015022_Run036_bright.data').astype(bool)
        func_dict['list_of_masks'] = [examplemask, examplemask2]
    except:
        pass
    ret_dict['jungfrau4M'] = func_dict
```
This is where the masks are loaded into the producer file so that the azimuthal averaging is done for each region. Change the directories to load the two different masks.
9. Lastly, we still need to generate the normal masks from a dark and pedestal run, which will take care of any hot pixels or bad regions in the detector. This mask will be applied normally over the other masks that you already made.