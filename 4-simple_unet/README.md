# Simple UNet Training (2nd experiment)
Can we train a UNet to enrich the DeepCAD predictions with more fine details while retaining the cleanliness of the image, given a dataset made of pairs like the following?

<p align="center">
    <img src="asset/noisy_0.png" width="40%"></img>
    <img src="asset/cond_0.png" width="40%"></img>
</p>

## BiLoss
I had an idea. Why not combining a loss on the ground truth `gt` with a loss on the input `x`?
See `202508051043`, `202508051050`, `202508051059` and ``