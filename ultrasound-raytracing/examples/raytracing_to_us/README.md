### Install dependencies

```bash
pip install -r requirements.txt
```

### Produce Fake 3D Mesh

```Bash
python fake3d_us_generation.py \
    --image_dir ~/code/gan_training_data/tcia_us_dataset/ray_tracing_images/ \
    --mask_dir ~/code/gan_training_data/tcia_us_dataset/ray_tracing_masks/ \
    --output_dir ~/code/gan_training_data/tcia_us_dataset/ray_tracing_fake_3d/
```

### Produce Mesh

```
python mesh_generation.py --seg_dir ~/code/gan_training_data/tcia_us_dataset/ray_tracing_fake_3d/
```