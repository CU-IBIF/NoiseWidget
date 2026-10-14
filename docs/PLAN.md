# NoiseWidget: design and model

A companion to MicroscopeWidget that answers one question with numbers: *for these
settings, where does the noise in my image come from, and how much does it corrupt
what I measure?*

---

## 1. Decisions

| # | Decision | Consequence |
|---|---|---|
| 1 | **Same stack and conventions as MicroscopeVuilder** | numpy model, PySide6 UI, QPainter plots (no matplotlib), one-file PyInstaller build, offscreen Qt tests, the same dark panes and status colours, and rows that carry their equation. |
| 2 | **The widget is a `QWidget`** | MicroscopeWidget can host it in a tab or dock; `NoiseWindow` only adds the toolbar and status bar. |
| 3 | **Model by physics, not by look** | Noise comes from simulating photon arrival, charge generation, gain and digitization, not from added "grain". The analytic budget the graph draws is tested against the simulation's measured variance. |
| 4 | **Ground truth is generated** | Specimens are fluorophore positions at known sizes, so every measurement has an answer to compare with. |
| 5 | **Two clocks** | The object-to-image FFT (~0.1 s) is cached on NA, wavelength and specimen. Everything after the PSF reruns in milliseconds, so sliders and live mode stay interactive on the GUI thread. |

## 2. The signal chain

```
fluorophores --(irradiance, exposure)--> photons emitted        k_em = QY sigma I/(h nu) / (1 + k_abs tau)
             --(NA, n)------------------> collected            (1 - cos asin(NA/n)) / 2
             --(transmission)-----------> at the sensor
             --(lambda, NA)-------------> blurred              analytic incoherent OTF, cutoff 2NA/lambda
             --(pixel x bin / M)--------> integrated           exact box integral over camera pixels
             --(QE)---------------------> photoelectrons
             + background, dark current D(T) = D_ref 2^((T - T_ref)/6.5), CIC
```

Photobleaching scales the mean signal by the fraction of fluorophores surviving the
acquisition, which is exact for the exposure windows of N averaged frames.

## 3. Noise sources

| Source | Model | Averages away? |
|---|---|---|
| Labelling stochasticity | Poisson number of labels, random positions on the structure's wall | no (it is the specimen) |
| Photon shot noise | Poisson on signal + background photoelectrons | yes, 1/√N |
| Background shot noise | as above; the background carries noise but no information | yes |
| Dark current (thermal) shot noise | Poisson on D(T) t | yes |
| Clock-induced charge | Poisson, EMCCD only | yes |
| EM excess noise | Gamma(n, G) cascade: F² = 2 | yes |
| Read noise | Gaussian; per-pixel lognormal spread for (s)CMOS; one read per superpixel for charge binning, b² in quadrature for digital | yes |
| PRNU | per-pixel gain map, proportional to signal | **no** (flat-field removes it) |
| DSNU, hot pixels | per-pixel dark-current map | **no** (dark subtraction removes the mean) |
| Quantization | rounding to ADU, 1/√12 ADU rms | yes |
| Clipping | full well (binning-aware), EM register, ADC bit depth | n/a: biases, doesn't scatter |

Fixed-pattern maps are seeded per camera and stay put when the noise is re-rolled.
That is how a user discovers that averaging frames does not remove them.

## 4. Quantitation

The reference structure at the centre is measured as at the bench. The profile across
it is averaged along a chosen filament length, or summed for a bead, then fitted with
a Gaussian plus background by batched Levenberg–Marquardt. The same noise-free scene is
re-recorded R times (only the measured crop, so it stays interactive), giving:

* **Width**: the noise-free fit is the optics' bias, and the spread over frames is the
  noise's. Subtracting the instrument in quadrature uses the model's own fitted image
  of a thin line (PSF plus pixels), which is the bead calibration done exactly. Below
  the diffraction limit, noise makes that subtraction meaningless, which is the point.
* **Label count**: the profile area divided by photoelectrons per fluorophore,
  corrected by the instrument's light recovery (Airy rings fall outside any Gaussian).
  It is shown by fit and by background-subtracted sum, against the labels actually
  placed.
* **Position**: scatter of the fitted centre against the shot-noise floor σ/√N.

## 5. What is not modelled

* Vectorial high-NA PSF (scalar paraxial; real PSFs above NA 1.2 are a few % wider) and
  aberrations, including spherical aberration from index mismatch.
* Defocus and 3D: one focal plane. Out-of-focus light enters only as uniform background.
* Wavelength-dependent QE, filter spectra, and spectral bleed-through.
* Dipole orientation and near-interface emission effects on collection efficiency.
* Blinking, triplet states, phototoxicity, drift, vibration, cosmic rays.
* sCMOS rolling shutter, and amplifier glow in long exposures.

## 6. Layout

```
noisewidget/
  physics/          headless, numpy only
    optics.py         OTF/PSF/LSF, resolution, Nyquist, collection, box integration
    fluorophores.py   photophysics and bleaching
    specimens.py      microtubules, actin, beads with known sizes and label geometry
    camera.py         sensor classes, dark current vs temperature, fixed pattern
    acquisition.py    levers -> recorded image; the analytic budget and sweeps
    quantitation.py   profiles, batched Gaussian fits, repeat-frame statistics
    report.py         every derived number, with its equation
    presets.py        starting points that each make one point
  ui/               PySide6
    widget.py         NoiseWidget (embeddable), NoiseWindow, graph and quantitation panes
    controls.py       the lever panel
    panes.py          image panes (nearest-neighbour, scale bar, overlays)
    plots.py          QPainter plots with hover read-out
```
