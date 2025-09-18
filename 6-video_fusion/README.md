# Improve DeepCAD With Fine Details With DCT-Fusion

## DCT
To preserve dynamics:
- $t_0 = 0$
- $δ_t \le 128$

## Wavelets
```python
import pywt
import numpy as np
'sym4', 'coif1'
def wavelet_fuse(vol1, vol2, wavelet='db2', level=2):
    """
    Fuse two volumes via an n‐level 3D discrete wavelet transform:
      - vol1: high‐detail averaged stack
      - vol2: dynamic but blurrier stack
    Strategy:
      1. Decompose each volume into a low‐pass approximation at level `level`
         and a set of detail sub‐bands.
      2. Take the approximation from vol1 (to keep fine structure).
      3. Take all detail bands from vol2 (to keep fast dynamics/texture).
      4. Reconstruct.
    """
    # 1) Full n‐level decomposition
    coeffs1 = pywt.dwtn(vol1, wavelet=wavelet, axes=(0,1,2))
    coeffs2 = pywt.dwtn(vol2, wavelet=wavelet, axes=(0,1,2))

    # 2) Build fused coeff dict
    fused = {}
    for key in coeffs1:
        if key == ('aaa',)*level:    # the deepest approximation band
            fused[key] = coeffs1[key]
        else:
            fused[key] = coeffs2[key]

    # 3) Inverse transform
    fused_vol = pywt.idwtn(fused, wavelet=wavelet, axes=(0,1,2))
    return fused_vol

# Example usage:
# fused = wavelet_fuse(vol_high_detail, vol_dynamic, wavelet='db2', level=2)


α = 0.7  # weight for vol1 in detail bands
fused[key] = α*coeffs1[key] + (1-α)*coeffs2[key]
```