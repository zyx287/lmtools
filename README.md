# lmtools
Tools and scripts for processing and visualizing light microscopy data


## Example Use
### As a python package
```python
from lmtools import load_nd2, generate_segmentation_mask, maskExtract
# Load an ND2 file
viewer = load_nd2("my_image.nd2")
```
### From the command line
```bash
lmtools load_nd2 my_image.nd2
lmtools generate_mask my_data.geojson output_dir 1024 768 --downsample_factor 2
lmtools extract_mask --file_path cellpose_output.npy --output_path extracted_mask.npy
```