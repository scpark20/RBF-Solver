## CIFAR-10 (Score-SDE)

The `score_sde` directory contains everything needed to reproduce the CIFAR-10 (Score-SDE) main results reported in the paper.  
This implementation is adapted from the codebase at <https://github.com/thu-ml/DPM-Solver-v3>, which is distributed under the **MIT License**.

---

### Download Checkpoint
Manually download **`checkpoint_8.pth`** from  
<https://drive.google.com/drive/folders/1F74y6G_AGqPw8DG5uhdO_Kf9DCX1jKfL>  
and place it in `checkpoints/cifar10_ddpmpp_deep_continuous/`.

---

### Install Dependencies
Install the Python packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

### Sample Target  
Generate target image–noise pairs.  
This will create `samples/checkpoint_8/UniPC_bh1_200/samples_0.pt`.
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

### Sample with RBF-Solver  
Generate samples using RBF-Solver.
```bash
./sample_rbf.sh
```

---

### Sample with Other Samplers  
Generate comparison samples with baseline samplers.
```bash
./sample_others.sh
```

---

### Prepare FID Statistics  
Manually download **`cifar10_stats.npz`** from  
<https://drive.google.com/drive/folders/1bofxWSwcoVGRqsUnAGUbco1z5lwP0Rb6>  
and place it in `assets/stats/`.

---

### Evaluate FID  
Compute the FID score using the generated images.
```bash
python compute_fid.py
```