# Questions for OABF
- Are dynamics slow? Take the video [`motion_corrected.mp4`](2-denoise/motion_corrected.mp4), is really that slow? Or are there any fast dynamics events? I thought the calcium dinamycs were in the order of 100-1000ms
- What are the values in the input `tiff`? (Photon count)
- What is the FPS? --> 7Hz (Astro, VPM) 30Hz(Res.Neur)
- How were the *denoised* version obtained? ---> DeepCAD
- What do "good" and "bad" refer to in the OneDrive hierarchy? ---> Consider also "bad" recording
- Hyperparams for DeepCAD, or code used

# Questions for us
- How do we quantitatively compare predictions? ---> PSNR/SSIM over averaged frames. DeepCAD for dynamics?
- How do we provide RT if using diffusion model? ---> distillation, LCM
- Does DeepCAD RT training trains DeepCAD then distillation ---> no it directly trains the pruned net

# Questions for me
- Light Sheet microscopy have allowed for faster and more gentle imaging than ever before (Light-sheets and smart microscopy, an exciting future is dawning)
- Confocal
- diverse *U2OS* cell data acquired with *iSIM*.

# Ideas
- DCT Video Fusion 🟢
- DiffDenoise 🟠
- Marigold 🔴
- Benchmark on segmentation -
- RGB2RAW - Useless
- FPS Augumentation with DeepCAD?
- CARE (sup)
- 3D-RCAN (sup)

# Video Fusion
- DCT vs FFT
- plot average PSNR vs GT