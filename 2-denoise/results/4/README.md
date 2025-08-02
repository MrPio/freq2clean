# `resonant_neuro_202508022117`

A *ResidualUNet3D* is trained on `resonant_neuro` OABF dataset. This training is small on purpose. As you can see from the [`para.yaml`](../../pth/resonant_neuro_202508022117/para.yaml) file, I have set:

```yaml
n_epochs: 10
select_img_num: 1500
train_datasets_size: 1000
```

instead of the previous

```yaml
n_epochs: 20
select_img_num: 6000
train_datasets_size: 5000
```