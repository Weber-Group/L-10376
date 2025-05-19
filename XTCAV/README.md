# TODO: Temporal ghost imaging & XTCAV analysis

- [ ] Dataset structure
  - [ ] Define epochs
- [ ] Reference data
  - [ ] Identify reference datasets (dark, no lasing) for each epoch
  - [ ] Select epoch, get average reference captures
  - [ ] Variations in refs, outliers & variance
- [ ] Run xtcav2
  - [ ] xtcavDark
  - [ ] xtcavLasingOff
  - [ ] xtcavLasingOn
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

