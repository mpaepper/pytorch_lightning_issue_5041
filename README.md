# pytorch_lightning_issue_5041

Replication for [issue 5041 of PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning/issues/5041)

Running on 4 RTX 2080Ti.

Running on GPUs 0 and 1 works: `python index.py '0,1'`

Running on GPUs 0, 2 and 3 doesn't work / hangs: `python index.py '0,2,3'`
