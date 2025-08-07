# Simple UNet Training (2nd experiment)
Can we train a UNet to enrich the DeepCAD predictions with more fine details while retaining the cleanliness of the image, given a dataset made of pairs like the following?

<p align="center">
    <img src="asset/noisy_0.png" width="40%"></img>
    <img src="asset/cond_0.png" width="40%"></img>
</p>

### BiLoss
I had an idea. Why not combining a loss on the ground truth `gt` with a loss on the input `x`?
See `202508051043`, `202508051050`, `202508051059`, ...

The [`lf_hf_tv_loss`](loss.py) got me the best results.

## Experiments Overview
|             Trainings              |                           Loss                            | What's changed                                                                                                                                                                                                      |
| :--------------------------- | :------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `202508041803`,`202508041831` |               $L1_{GT}+SSIM_{GT}+Grad_{GT}$               | Samples averaged with a 192-large sliding window centered on the frame are used as ground truth. The resulting predictions may appear satisfactory, but the dynamics are smoothed out during the recording process. |
|                               |                                                           |                                                                                                                                                                                                                     |
|        `202508041853`         |               $L1_{GT}+SSIM_{GT}+Grad_{GT}$               | Averaged ground truths are replaced with vanilla noisy frames. This is okay because the noise is independent between frames, so the network won't learn to add noise to the input.                                  |
|        `202508041859`         |                        $MSE_{GT}$                         | L1 loss tends to foster overly sharp predictions. MSE produces better results.                                                                                                                                      |
| `202508041907`,`202508041920` |          $MSE_{GT}+L1_{GT}+SSIM_{GT}+Grad_{GT}$           | A more complex combined loss.                                                                                                                                                                                       |
|                               |                                                           |                                                                                                                                                                                                                     |
|        `202508051043`         | $LowFreq_{X}+0.5\times HighFreq_{GT}+1e-4\times TotalVar$ | The loss now takes into account the input itself, other than the ground truth.                                                                                                                                      |
| `202508051050`,`202508051102` |  $LowFreq_{X}+2\times HighFreq_{GT}+2e-4\times TotalVar$  | More sharpen results.                                                                                                                                                                                               |
|        `202508051059`         |             $MSE_X+SSIM_X+0.5\times L1_{GT}$              | Not satisfactory.                                                                                                                                                                                                   |
