## Imagenet64×64 (Improved-Diffusion)

The `ddpm_and_guided-diffusion` directory contains everything needed to reproduce the Imagenet64×64 (Improved‑Diffusion) main results reported in the paper.  
This implementation is adapted from the codebase at <https://github.com/LuChengTHU/dpm-solver>, which is distributed under the **MIT License**.

---

### Download Checkpoint
Download the pre-trained Improved-Diffusion checkpoint and place it under `ddpm_ckpt/imagenet64`:
```bash
mkdir -p ddpm_ckpt/imagenet64
wget -O ddpm_ckpt/imagenet64/imagenet64_uncond_100M_1500K.pt \
  https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt
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
This will create `experiments/imagenet64/unipc_order3_200/image_samples/images/target.pt`.
```bash
./sample_target.sh
```

---

### Shape Optimization  
Learn shape parameters for **RBF-Solver**.  
This will create shape files named `shape_dir/NFE={NFE},p={order}.npz`.  
*Note: The supplementary zip file already contains the `.npz` files.*
```bash
./shape_optim.sh
```

---

### Prepare FID Statistics  
Manually download **`fid_stats_imagenet64_train.npz`** from  
<https://drive.google.com/drive/folders/1_OpTXVPLffZM8BG-V3Ahsxk99aqxW7C3>  
and place it in `fid_stats/`.

---

### Sample with RBF-Solver and Evaluate FID  
Generate samples using RBF-Solver.
```bash
./sample_rbf.sh
```

---

### Sample with Other Samplers and Evaluate FID  
Generate comparison samples with baseline samplers.
```bash
./sample_others.sh
```
