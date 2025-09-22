# Enhancing Denoiser Models with FFT/DCT Video Fusion
My method provide a substantial improvement over the denoiser ([*DeepCAD-RT*](https://github.com/cabooster/DeepCAD-RT) in this example) prediction. As you can see, PSNR increases by $2dB$, but the most interesting result is the improvement of SSIM2D by $0.2$ points.
<p align="center">
  <img width="90%" src="assets/fft_vs_baseline.png"/>
</p>


## Test Datasets
### [Synthetic Calcium Imaging](https://zenodo.org/records/6254739)
This is the most relevant dataset in this study:
- **It is synthetic**, yet very much alike the real dataset provided by the affiliated research group.
- As such, **it has ground truths**. Therefore we can fairly assess the validity of the proposed solution.

<p align="center">
  <img width="45%" src="assets/fft_vs_deepcadrt.gif"/>
  <img width="70%" src="assets/synthetic.png"/>
</p>

### [Zebrafish Multiple Brain Regions](https://zenodo.org/records/6293696)
<p align="center">
  <img width="70%" src="assets/zebrafish.png"/>
</p>

### [Mouse Brain Neutrophils](https://zenodo.org/records/6296569)
<p align="center">
  <img width="70%" src="assets/neutrophils.png"/>
</p>

