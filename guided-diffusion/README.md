## Imagenet128×128 (Guided‑Diffusion)

The `guided-diffusion` directory contains everything needed to reproduce the Imagenet128×128 (Guided‑Diffusion) main results reported in the paper.  
This implementation is adapted from the codebase at <https://github.com/thu-ml/DPM-Solver-v3>, which is distributed under the **MIT License**.

---

### Download Checkpoints
Download the pre‑trained Guided Diffusion checkpoints for both the model and the classifier:
```bash
mkdir -p ddpm_ckpt/imagenet128
wget -O ddpm_ckpt/imagenet128/128x128_diffusion.pt \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_diffusion.pt
wget -O ddpm_ckpt/imagenet128/128x128_classifier.pt \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_classifier.pt
```

---

### Install Dependencies
Install Python packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

### Sample Target  
Generate target image–noise pairs.  
This will create `samples/128x128_diffusion/unipc_200_scale{scale}/images/target_{0-15}.pt`.
```bash
./sample_target/imagenet128.sh
```

---

### Shape Optimization  
Learn shape parameters for **RBF-Solver**.  
This will create shape files named `shape_dir/imagenet128/scale{scale}/NFE={NFE},p={order}.npz`.  
*Note: The supplementary zip file already contains the `.npz` files.*
```bash
./shape_optim/imagenet128.sh
```

---

### Prepare FID Statistics
Download the FID statistics file:
```bash
mkdir -p fid_stats
wget -O fid_stats/VIRTUAL_imagenet128_labeled.npz \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/VIRTUAL_imagenet128_labeled.npz
```

---

### Sample with RBF-Solver and Evaluate FID
Generate samples using RBF-Solver.
```bash
./sample/rbf_imagenet128.sh
```

---

### Sample with Other Samplers and Evaluate FID
Generate comparison samples with baseline samplers.
```bash
./sample/others_imagenet128.sh
```


## Imagenet256×256 (Guided‑Diffusion)

The `guided-diffusion` directory contains everything needed to reproduce the Imagenet256×256 (Guided‑Diffusion) main results reported in the paper.  
This implementation is adapted from the codebase at <https://github.com/thu-ml/DPM-Solver-v3>, which is distributed under the **MIT License**.

---

### Download Checkpoints
Download the pre‑trained Guided Diffusion checkpoints for both the model and the classifier:
```bash
mkdir -p ddpm_ckpt/imagenet256
wget -O ddpm_ckpt/imagenet256/256x256_diffusion.pt \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt
wget -O ddpm_ckpt/imagenet256/256x256_classifier.pt \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt
```

---

### Install Dependencies
Install Python packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

### Sample Target  
Generate target image–noise pairs.  
This will create `samples/256x256_diffusion/unipc_200_scale{scale}/images/target_{0-15}.pt`.
```bash
./sample_target/imagenet256.sh
```

---

### Shape Optimization  
Learn shape parameters for **RBF-Solver**.  
This will create shape files named `shape_dir/imagenet256/scale{scale}/NFE={NFE},p={order}.npz`.  
*Note: The supplementary zip file already contains the `.npz` files.*
```bash
./shape_optim/imagenet256.sh
```

---

### Prepare FID Statistics
Download the FID statistics file:
```bash
mkdir -p fid_stats
wget -O fid_stats/VIRTUAL_imagenet256_labeled.npz \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```

---

### Sample with RBF-Solver and Evaluate FID
Generate samples using RBF-Solver.
```bash
./sample/rbf_imagenet256.sh
```

---

### Sample with Other Samplers and Evaluate FID
Generate comparison samples with baseline samplers.
```bash
./sample/others_imagenet256.sh
```
