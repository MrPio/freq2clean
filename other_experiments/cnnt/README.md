# [Convolutional neural network transformer (CNNT) for fluorescence microscopy image denoising with improved generalization and fast adaptation](https://www.nature.com/articles/s41598-024-68918-2)

This experiment aims to reproduce the CNNT training. The online repository lacks the pretrained checkpoints, thus, I am going to train the backbone from scratch first. Then, I am going to finetune it like specified by the authors, on my dataset.

However, three probles are found:
1. The backbone training requires 4 days on 2 A100 GPUs. That's a bit too much.
2. After training for 60 epochs out of 300, even though the loss seems to converge, the prediction over my dataset doensn't hold a candle against DeepCAD-RT's.
3. My dataset is unlabelled! My choice to use pseudo ground truths may be unbecoming.