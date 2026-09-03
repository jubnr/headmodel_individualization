# Headmodel Individualization

> **This is a fork of [harmening/headmodel_individualization](https://github.com/harmening/headmodel_individualization).**
> The algorithm and the PCA bases are unchanged. What this fork adds is everything around them:
> automatic fiducial handling, a quality-control gate on every run, reproducible output, and a
> roughly 3x faster pipeline. See [Changes in this fork](#changes-in-this-fork) for the full list
> and the reasoning.

**The presented PCAwarp algorithm estimates individual head anatomies based on a large database of heads when structural MRI/CT scans are unavailable using scalp data from photogrammetry or digitized electrode positions. The final surfaces meshes can be used to construct a BEM volume conduction head model for source reconstruction with e.g. [OpenMEEG](https://openmeeg.github.io/).<br>
In our related scientific publication ["Data-driven head model individualization from digitized electrode positions or photogrammetry improves M/EEG source localization accuracy"](https://direct.mit.edu/imag/article/doi/10.1162/IMAG.a.1073/134446) we demonstrate that our individualized PCAwarp head model outperformes any other head model in terms of source localization error (synthetic EEG data, SNR=10):**
<img src="img/Simulation_study_raincloud.png"><br>
**Measured in residual variance (RV) in sensor space (left) and euklidian distance in source space (right).** <br>
<br>
<br>


## Practitioner's Guide
This tutorial demonstrates how to perform a 3D-scan of a single subject's head and how to construct individualized surface meshes using the PCAwarp algorithm.<br>

Also check out [my related video tutorial on photogrammetry](https://youtu.be/PMBUWHnLXUo?si=qxrb89QRhaXAVt0j) (fNIRS-cap, but same procedure)<br>
and the similar [tutorial on how to digitize electrodes by FieldTrip](https://www.fieldtriptoolbox.org/tutorial/electrode/).<br>
<br>
<img align="right" width="300" src="img/scaniverse_overview.jpeg">

### 1. Preparing the subject
- Measure the head circumference along RPA, IN, LPA, NAS, RPA.
- Adjust downward to choose an appropriate cap size (54cm, 56cm, 58cm, 60cm).
- Attach the cap and check and correct the cap positioning using the fiducials if necessary.<br>
  (Cz should be in the middle of LPA and RPA and NAS and IN.)
- Ensure thorough cable management so that they don't protrude or overlap electrodes.
- Instruct the subject to sit still and minimize head movements.
- Equal lightning from all sides is recommended.
<br>

### 2. Setting up the scanning device
This tutorial uses an iPhone 12 mini and the app [Scaniverse](https://scaniverse.com/). However, any other hardware/software scanning solution can replace this.<br>

- Download and install the app [Scaniverse](https://scaniverse.com/) on your smartphone.
<img align="right" width="430" src="img/scaniverse_settings_start.png">
  
- At the bottom, click on '+' -> 'Mesh' -> 'Small Object'.
- Press the red recording button in the button and follow the instructions.
<img align="right" width="390" src="img/scaniverse_settings_stop.png">
  
- Press the red square at the bottom to stop the recording if you are finished.
- Select 'Detail' as Processing Mode and 'Save' after the processing is finished.
- At the bottom right, click on 'Share' -> 'Export Model' -> 'OBJ' -> 'Mail' to your computer.

For details, also check [my scaniverse video tutorial (fNIRS-cap, but same procedure)](https://www.youtube.com/watch?v=PMBUWHnLXUo&t=410s) (starting at min 33:40).<br>
<br>

### 3. During scanning
- Move the scanning device slowly around and above the subject. (Slow movements are essential to allow the software to keep track of the already scanned points. As the scanner moves, new points are added to the point cloud.)
- The ideal scanning distance is about 40cm.
- Start with less difficult regions so you’re able to capture a lot of the surface before going to more detailed regions that need more time.
- Pan and tilt the scanning device in difficult regions like at electrodes to capture everything from all angles.
- Make sure to have good quality everywhere, but at the same time, try to be quick and avoid scanning regions multiple times. Taking around 3 minutes per scan gave the best results. (If the scan lasts too long, the risk of distortion by head movements increases.)
<br>


### 4. Post-processing
<img align="right" width="360" src="img/MeshLab_cropping.gif">

If necessary, crop the obj-mesh to the EEG cap only.<br>
This is only required if more objects than the head <br>
(e.g. from the background) were accidentally scanned or/and <br>
if due to data protection the subject's face needs to be removed.<br>
This can be done easily by using [MeshLab](https://www.meshlab.net/) as shown<br>
in the right or by any other mesh manipulation software.<br>
<br>

### 5. Picking fiducials
<img align="right" width="360" src="img/MeshLab_fiducial_picking.gif">

- For example in [MeshLab](https://www.meshlab.net/), see on the right.<br>
  Save the picks with `File -> Export Picked Points` next to the scan, as `<scan>.pp` —<br>
  PCAwarp finds that file on its own, so there is nothing to retype. Name the three<br>
  points NAS, LPA and RPA; unnamed points are read in picking order (NAS, LPA, RPA).
- Or in [cedalion](https://doc.ibs.tu-berlin.de/cedalion/doc/dev/examples/head_models/41_photogrammetric_optode_coregistration.html), automatic detection of marked the fiducials<br>
with colored stickers (manual correction possible).
- Or in [FieldTrip](https://www.fieldtriptoolbox.org/), by loading and plotting the mesh and simply <br>
clicking on the fiducials and writing the coordinates down:
  ```
  headshape = ft_read_headshape('scaniverse.obj')
  ft_plot_headshape(headshape)
  ```
<br>

## Headmodel individualization
### 6. Call the PCAwarp algorithm
> ### :warning: You need a second repository
> With the default `HARTMUT = True`, PCAwarp also warps the [HArtMuT](https://github.com/harmening/HArtMuT)
> artefact sources (eye and muscle dipoles) into the individual head. Those sources and their
> template meshes live in the **HArtMuT repository**, which is a *separate clone* and is not
> vendored here. `PCAwarp.py` expects it as a **sibling directory**:
>
> ```
> parent/
> ├── headmodel_individualization/   <- this repo
> └── HArtMuT/                       <- git clone https://github.com/harmening/HArtMuT
> ```
>
> Specifically it reads `HArtMuT/HArtMuTmodels/HArtMuT_NYhead_small.mat` and
> `HArtMuT/individualwarp/NYhead/{scalp,skull}.stl`. If your copy lives somewhere else, edit
> `HARTMUT_REPO` at the top of `PCAwarp.py`. If you do not want the artefact model at all, set
> `HARTMUT = False` — but note that this also switches the PCA basis from `data/pcas_hartmut`
> (neck-extended scalp, no cortex variance) to `data/pcas` (no neck, cortex variance present).
>
> The run stops immediately with instructions if the checkout is missing, rather than failing
> after a multi-minute fit.

```
# Clone this repository and its HArtMuT companion side by side
git clone https://github.com/jubnr/headmodel_individualization
git clone https://github.com/harmening/HArtMuT
cd headmodel_individualization
pip install -e .            # or: pip install -r requirements.txt
```
The script `PCAwarp.py` shows how to start the PCAwarp individualization algorithm. This is based on a low-dimensional representation (PCA) of head shape surface meshes trained on an equally segmented and triangulated MRI database of 316 subjects. Warping is done by finding weights for the PCs by minimizing the shape difference between electrodes / scalp proxies and fitted scalp. Exemplary call:<br>
```
python PCAwarp.py -scalp data/photogrammetry_test_data/cutscan.obj
```
The landmarks are picked up automatically from a file sitting next to the scalp file — a MeshLab `.pp`, a 3D Slicer `.mrk.json` or `.fcsv`, or a text file with one `LABEL x y z` line per landmark, named after the scan (`cutscan.pp`) or simply `fiducials.txt`. Point at one explicitly with `-fiducials`, or type the coordinates in as before:<br>
```
python PCAwarp.py -scalp data/photogrammetry_test_data/cutscan.obj
                  -nas   144.482786 129.291732 380.645666
                  -lpa   154.618663 59.192534 488.364463
                  -rpa   80.580458 21.190605 362.990298
```
Two things are checked before anything is warped, because both are silent killers: the scalp file is rescaled to mm if it was exported in metres or centimetres, and the landmarks are rejected if their spacing is not that of a human head (a mesh picker's vertex *indices* pasted in place of coordinates are recognised and looked up instead).<br>

Useful options:<br>
| flag | default | what it does |
|---|---|---|
| `--n-points` | 100 | how many scalp proxy points the warp is fitted to |
| `--sampling` | `fps` | `fps` picks them by farthest-point sampling (deterministic, even coverage); `random` is the old behaviour and needs `--seed` to repeat |
| `--seed` | – | seed for `--sampling random` |
| `--no-qc-gate` | off | export the model even if the quality checks fail |
| `--regularize` | off | experimental: penalize shells coming closer than 5 mm during the fit |
<br>

It contains the following steps:
<img align="right" width="300" src="img/PCAwarped_meshes.png">

* Using the fiducials (NAS, LPA, RPA), transform the scalp<br>
proxy points into the [CTF-coordinate system](https://www.fieldtriptoolbox.org/faq/coordsys/), since the<br> 
database on which the PCA was applied lives in CTF space.
* Cut the input scalp proxy mesh above the ears.
* Reduce the input scalp proxy to `--n-points` points by<br>
farthest-point sampling, to keep the shape-difference<br>
computation cheap without losing coverage.
* PCA warping, parameters are: number of PCs used for <br>
reconstruction, regularizer type (if meshes are intersecting).
* Transform back from CTF in original input space.
* Save surface meshes.

**Known limitation.** The HArtMuT PCA basis (`data/pcas_hartmut`, used when `HARTMUT = True`) ships without a cortex block, so the cortex cannot be fitted to the scalp. It is instead moved with the csf surface it sits inside — an approximation that keeps the shells nested, but the innermost surface is not individualized. Set `HARTMUT = False` to use `data/pcas`, which does carry cortex variance.

**Supported input file formats (scalp proxy):**
* Any [trimesh](https://trimesh.org/) supported mesh format like .stl, .obj, .ply, ....
* [BrainVision CapTrak](https://www.brainproducts.com/solutions/captrak/) files (.bvct).
* [Polhemus Fastrak](https://polhemus.com/scanning-digitizing/digitizing-products/) files (.hsp, .elp, .eeg).
* [Polaris Vicra](https://www.ndigital.com/optical-navigation-technology/polaris-vicra/) files(.elc).
* numpy array (.npy) containing (n x 3) coordinates of the scalp proxy points.
* simple .txt containing (n x 3) coordinates of the scalp proxy points.

**Supported output file formats (surface meshes):**
* .tri for [OpenMEEG](https://openmeeg.github.io/).
* .mat for [Matlab](https://de.mathworks.com/products/matlab.html) / [FieldTrip](https://www.fieldtriptoolbox.org/).
* .surf for [MNE-python](https://mne.tools/stable/index.html).
* .npy as simple dictionary containing the 4 output meshes.
* .nii segmentation masks for [cedalion](https://doc.ibs.tu-berlin.de/cedalion/doc/dev/).

### 7. Check the result
Every run measures its own output and writes `qc.json` and `qc.png` next to the meshes, then refuses to export a model that fails. The headline number is the scalp fit residual against the **full** input cloud, next to the same residual for the unwarped template — a warp that does not beat the population mean has not worked, however plausible the surfaces look:<br>
```
Quality control...
  scalp fit vs 9443 scan points: median 1.52 mm, p95 7.67 mm, max 12.76 mm
    unwarped template baseline:          median 4.52 mm, p95 13.72 mm   (3.0x better)
  shell        volume  watertight  euler   gap to outer
  scalp      4281.4 cm3        True      2
  skull      2037.1 cm3        True      2        0.77 mm
  csf        1525.2 cm3        True      2        0.05 mm
  cortex     1027.8 cm3        True      2        0.50 mm
  QC passed.
```
`qc.json` also records the fitted PC weights, how much of each shell fits inside the exported volume, and the provenance of the run (input file, resolved fiducials and where they came from, unit scaling, sampling, git SHA). Hard failures — a surface that is not watertight, shells that cross, or a fit no better than the template — stop the run; `--no-qc-gate` overrides.<br>
<br>

**Need more support/interfaces? Please contact me or open an issue on GitHub.**<br>

## Development
```
pip install -e ".[dev]"
pytest -m "not slow"    # unit tests, about a second
pytest -m slow          # full pipeline on the shipped test scan, several minutes
ruff check .
```





## Changes in this fork

The algorithm, the PCA bases and the export formats are unchanged. Everything below came out of
auditing one real run that produced a completely unusable head model without the pipeline
noticing: the fiducials had been pasted in as mesh **vertex indices** rather than coordinates, and
the scan was exported in **metres** while the pipeline assumes millimetres. The warp fitted its 16
components to what was effectively a single point and returned four watertight, properly nested,
entirely plausible surfaces that matched the scan **worse than no warping at all** (median 8.4 mm,
against 4.5 mm for the unwarped template). It exited 0 and said nothing.

### Fiducials and units are handled for you
* Landmarks are read from a file next to the scan — MeshLab `.pp`, 3D Slicer `.mrk.json`/`.fcsv`,
  or plain text — so they no longer have to be retyped. `-fiducials`, and `-nas`/`-lpa`/`-rpa`,
  still work and take precedence.
* Values that are really **vertex indices** are recognised and looked up on the mesh.
* A scan in **metres or centimetres** is rescaled to mm automatically.
* Landmark spacing outside human range is **refused**, printing the offending distances, and a
  scalp cloud that collapses after the CTF transform stops the run instead of being fitted.

### Every run checks its own output
`src/qc.py` writes `qc.json` and `qc.png` beside the meshes and **refuses to export a model that
fails**. The decisive figure is the scalp fit against the full input cloud shown next to the same
fit for the unwarped template — a warp that does not beat the population mean has not worked,
whatever the surfaces look like. It also records watertightness, Euler number, normal orientation,
inter-shell gaps, field-of-view clipping, the fitted PC coefficients, how far the warp moved the
scalp from the mean head, and the full provenance of the run. `--no-qc-gate` exports anyway.

### Results are reproducible
The scalp cloud used to be decimated with an unseeded `np.random.choice`. Three runs of one scan
through identical code gave scalp meshes differing by **mean 1.5 mm, max 5.9 mm** — the same order
as the total fit error, so two people running the same scan got different head models and
different source localizations. Farthest-point sampling replaces it: deterministic, and it halves
the worst coverage gap (26 mm vs 47–66 mm). On the reference scan the fit improved from 1.99 mm to
**1.52 mm** median as a side effect. Two runs now produce byte-identical meshes.

### About 3x faster: 5:17 → 1:49
* The warp optimizer rebuilt its ray caster on **every objective evaluation** by writing an ASCII
  STL to a fresh temp directory and reading it back through `vtkSTLReader` — 71 % of each
  evaluation, thousands of times per fit, leaking a directory each time (~14 000 of them and
  2.8 GB had accumulated in `/tmp` here). It is now built in memory from the numpy arrays, at
  float32 because that is what `vtkSTLReader` emitted, so ray results are **bit-identical**.
* `tri2nii` walked all 12.9 M voxels in nested Python loops twice per shell — 104 M iterations per
  run — and pushed 17 M points into VTK one at a time. Both are array operations now, against the
  same `vtkSelectEnclosedPoints` test.
* `tri_io.vertex_normals` was O(vertices × faces) via a per-vertex `np.argwhere` scan. A
  scatter-add gives bit-identical normals ~200x faster.

### Export bugs fixed
* **The cortex was never individualized.** `data/pcas_hartmut` ships with an all-zero cortex PCA
  block, so `inner_csf.surf` came back identical to the template mean in every run, and the csf
  surface — which does warp — crossed it by up to 6.4 mm. The basis cannot be rebuilt without the
  source MRI database, so the dead block is now detected and the cortex follows the csf
  displacement instead (the two share a triangulation and agree on vertex direction to ~4°), with
  the few remaining contacts clamped. It is announced on every run.
* **`mne/pcawarp/bem/*.surf` were written in voxel index space**, 127/127/94 mm away from the
  surface RAS that MNE reads them as, because `tri2nii` shifted the caller's vertex arrays in place
  and `export_cedalion` mutated the dict `export_mne` then wrote.
* **`T1.mgz` was written RAS-ordered** while nibabel's `get_vox2ras_tkr()` always assumes an
  LIA-conformed volume — deriving the surface RAS frame from it would have **mirrored the head
  left–right**. It is properly conformed now.
* **`ditigized2ras-trans.fif` carried a millimetre translation**; MNE transforms are metres, so the
  head sat 117 m away.
* **Every surface was wound inward**, which makes `mne.make_bem_solution` reject the BEM outright.
* **The neck was silently clipped.** Padding was a fixed 10 voxels on the *superior* side while the
  neck-extended scalp runs off the *inferior* edge — 20 % of scalp vertices fell outside the masks
  and `T1.mgz`. Padding is now derived from the mesh bounding box on all six sides.

`mne.make_bem_model`, `mne.make_bem_solution` and `mne.viz.plot_bem` all work on the exported
subject directory now, and the `head→mri` transform round-trips onto `outer_skin.surf` to 1e-5 mm.

### Housekeeping
* `--regularize` exposes the regularizer the README already advertised but that was unreachable
  behind a hard-coded default. **Experimental**: its penalty is unweighted against the shape
  distance and can dominate and blow the fit up.
* `scipy` was imported but missing from `requirements.txt`, so a clean install could not run the
  pipeline at all. `scikit-learn`, `tqdm` and `nilearn` were listed or imported but unused and are
  gone. Added `pyproject.toml` so `pip install -e .` works.
* 69 tests (65 fast, 4 marked `slow`) and a GitHub Actions run. There were none.
* Removed unreachable code: `pca_warp()`, `error1by1()` and the `onebyone` branch,
  `load_elecs_txt()`, `transform_to_ctf.apply()`, and the commented-out regularizer variants.
  `ruff` is clean.

## Citing
If you find the headmodel individualization useful for your research, please consider citing our related [paper](https://direct.mit.edu/imag/article/doi/10.1162/IMAG.a.1073/134446).
```
@article{Harmening_2026,
      author  = {Harmening, Nils and
                 von Lühmann, Alexander and
                 Blankertz, Benjamin}
      title   = {Data-driven head model individualization from digitized electrode positions or photogrammetry improves M/EEG source localization accuracy}
      year    = {2026},
      journal = {Imaging Neuroscience}
      doi     = {10.1162/IMAG.a.1073},
      volume  = {4},
}
```
