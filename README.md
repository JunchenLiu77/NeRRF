# NeRRF

------

NeRRF: 3D Reconstruction and View Synthesis for Transparent and Specular Objects with Neural Refractive-Reflective Fields

### Setup

```
conda env create -f environment.yml
conda activate nerrf
```

### Dataset

Our blender synthetic dataset can be found here: [blender_dataset](https://drive.google.com/drive/folders/1us6geRhh0FwCoXy7VQAzixP2z1JF6QtS?usp=sharing)

### Usage

1. In the `NeRRF ` directory, ensure you have the following data:

    ```
    NeRRF
    |-- data
        |-- blender
            |-- depth
            |-- meta
            |-- specular
            |-- transparent
    ```

2. Run the training script

    ```
    # reconstruct the specular horse of the blender dataset
    sh scripts/train_horse_s.sh
    ```

    The scripts can be switched from the geometry reconstruction stage to the radiance estimating stage by changing the value of the `stage` variable. Set `stage` to `1` for geometry reconstruction.

3. Run the evaluation script

    ```
    # evaluate the specular horse of the blender dataset
    sh scripts/train_horse_s.sh
    ```