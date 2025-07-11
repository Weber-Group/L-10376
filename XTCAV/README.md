# Temporal ghost imaging & XTCAV analysis: TODO items

- [X] Dataset structure
  - [X] Define epochs
- [X] Reference data
  - [X] Identify reference datasets (dark, no lasing) for each epoch
  - [X] Select epoch, get average reference captures
  - [X] Variations in refs, outliers & variance
- [X] XTCAV Processor class
  - [X] Instantiate, set data sources, run, times, detectors, defaults
  - [X] Data iteration, data validation, validity logging, 2nd iterations
  - [X] Data visualization, basic (image, zoom, histogram)
  - [X] Dark reference class
  - [X] Preprocessing - dark ref, denoising, pulse splitting
  - [X] Pulse statitics, vis pulse statistics, all pulse statistics
  - [X] Lasing off class, average profiles, vis
  - [ ] Lasing on class, shot-by-shot reconstructions, vis
  - [ ] Parallelize
- [ ] Calibrations
  - [ ] Get all necessary metadata: need SLAC assistance...
  - [ ] Complete calibrations
- [ ] Set up ghost imaging
  - [ ] Normalize, get p(t)
  - [ ] Get autocorrelation
  - [ ] Set up $\mathbf{A}$ matrix
- [ ] Diffraction
  - [ ] Run producers, get 1D scattering profiles
  - [ ] Set up $\mathbf{m}$ vectors
- [ ] Ghost imaging
  - [ ] Solve $\mathbf{m} = \mathbf{A}\mathbf{m}$
  - [ ] Visualize results



## Questions
------------

### Data source psana.Bld type determination

Data getter - in the old xtcav code, several different psana detector types
are searched for each data type (ebeam, gas detector) for each event.
Why is it done this way?  Are there several detectors that need to be checked
for each event?  Specifically, for e-beam data it checks
- psana.Bld.BldDataEBeamV7
- psana.Bld.BldDataEBeamV6
- psana.Bld.BldDataEBeamV5
and for gas detector data it checks 
- psana.Bld.BldDataFEEGasDetEnergy
- psana.Bld.BldDataFEEGasDetEnergyV1
If only one of these is used in a given experimental run, it will save compute
to figure this out once instead of checking at each event.


