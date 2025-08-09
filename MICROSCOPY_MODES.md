# Microscopy modes — differences (widefield, 2-photon, iSIM, confocal, light-sheet)

I’ll keep this concise but practical — focusing on how they differ in *illumination strategy, optical sectioning (in-focus vs out-of-focus light), typical depth penetration, speed, and phototoxicity*.

* **Widefield (epi-widefield) microscopy**

  * Illumination: whole sample (or whole field) illuminated at once.
  * Detection: camera collects light from the whole depth — **no optical sectioning**. Out-of-focus light blurs the image.
  * Pros: simple, fast (camera captures full FOV in one shot), relatively low per-frame complexity.
  * Cons: poor contrast in thick samples due to background; needs deconvolution or denoising to improve optical sectioning. (Used in paper as one downstream task.)&#x20;

* **Confocal microscopy**

  * Illumination & detection: point-scan laser focused into sample; a **pinhole** in front of detector rejects out-of-focus fluorescence — gives optical sectioning.
  * Pros: optical sectioning, improved contrast; common for multi-color tissue imaging.
  * Cons: point-scanning can be slower over large volumes; can be more phototoxic because of scanning and higher laser intensities (but depends on settings). (Used in paper for mouse lung tissue.)&#x20;

* **Two-photon microscopy (2PM)**

  * Illumination: near-IR pulsed lasers (two photons absorbed simultaneously) — excitation probability is nonlinear, confined to the focal volume.
  * Pros: intrinsic optical sectioning without pinhole, deeper tissue penetration (IR scatters less), reduced out-of-focus photobleaching/phototoxicity, good for live/deep tissue imaging.
  * Cons: requires femtosecond pulsed laser, typically slower (point-scan) and more expensive. (Used in the paper for zebrafish experiments.)&#x20;

* **iSIM (instant Structured Illumination Microscopy)**

  * SIM in general uses patterned illumination and computational reconstruction to boost resolution (\~2× improvement over diffraction limit). iSIM (instant SIM) is a hardware/optical variant designed to produce super-resolution results with very short latency (real-time or “instant”) by performing some optical processing and faster reconstruction.
  * Pros: **enhanced lateral resolution** beyond conventional widefield, good for live imaging with better resolution than widefield but faster/less phototoxic than some other super-res methods.
  * Cons: still limited depth versus two-photon; complexity in patterning and reconstruction. The paper used iSIM data for backbone pre-training.&#x20;

* **Light-sheet microscopy (SPIM / LSFM)**

  * Illumination: the sample is illuminated by a thin sheet of light from the side (orthogonal to detection axis). Only a thin plane is excited at any time → **intrinsic optical sectioning**. Detection via a camera collects the plane.
  * Pros: **very fast volumetric imaging** (camera for whole plane), *low phototoxicity* because only the imaged plane is illuminated, excellent for large live specimens (embryos, organoids).
  * Cons: sample geometry/handling can be more complicated (sample mounting), shadowing/artifacts for opaque samples. The paper mentions light-sheet advantages in the intro (context).&#x20;