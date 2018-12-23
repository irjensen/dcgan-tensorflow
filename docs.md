## train.py
`train.py` trains the model on a given set of images
### Flags
- `--checkpoint-dir` Directory to save checkpoint in. Directory will be created if it does not exist. Default: `./ckpt/`
- `--data-dir` Directory to set of training images. Default:`./data/`
- `--sample-dir` Directory to save sample images while training, if specified.
- `--epochs` Number of epoch to train for. Default:`5`
- `--crop-width` Horizontal width to center-crop training images before they are fed to the network
- `--crop-height` Vertical height to center-crop training images before they are fed to the network
- `--rescale-width` Horizontal size to rescale training images before they are fed to the network. Applied **after** cropping
- `--rescale-height` Vertical size to rescale training images before they are fed to the network. Applied **after** croppping
- `--batch-size` Batch size for training. Default:`64`
- `--learning-rate` Learning rate for training. Default:`2e-4`
- `--z-dim` Dimension of z vector used to generate images. Default: `100`
- `--constant-z` Use the same z for generation of samples during training

## generate.py
`generate.py` loads a saved model checkpoint from disk and generates random images
### Flags
- `--checkpoint` Directory to load checkpoint from. **Required**
- `--output-dir` Directory to save generated images. **Required**
- `--num_images` Number of images to generate. Default:`1`
- `--name` Name for the generated image(s). Do not include file extension. Default:`image`
- `--grid-size` Specify this to generate a square grid of images per file. Default:`1`
- `--seed` seed for the random generation of z
