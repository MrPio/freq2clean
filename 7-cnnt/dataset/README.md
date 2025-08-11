# Dataset download

Download the [`Denoising.zip`](https://zenodo.org/records/4624364#.YF4lBa9Kgal) dataset, and `unzip` it in this `dataset/` folder. The expected hierarchy is:

```sh
dataset
├── Denoising
│   ├── Actin
│   │   ├── Test
│   │   │   ├── GT
│   │   │   │   ├── 1_decon.tif
│   │   │   │   ├── ...
│   │   │   └── Raw
│   │   │       ├── 10.tif
│   │   │       ├── ...
│   │   └── Training
│   │       ├── GT
│   │       │   ├── ...
│   │       └── Raw
│   │           ├── ...
│   ├── ...
```

Finally, run [`dataset2hdf5.py`](../dataset2hdf5.py) to convert the dataset to HDF5 desired format.