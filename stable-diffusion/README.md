## Stable‑Diffusion v1.4

The `stable-diffusion` directory contains everything needed to reproduce the Stable Diffusion v1.4 main results reported in the paper.  
This implementation is adapted from the codebase at <https://github.com/thu-ml/DPM-Solver-v3>, which is distributed under the **MIT License**.

---

### Download Checkpoints
Download the pre‑trained Stable Diffusion checkpoints for both the model and the classifier:
```bash
mkdir -p models/ldm/stable-diffusion-v1
wget -O models/ldm/stable-diffusion-v1/sd-v1-4.ckpt \
  https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
```

---

### Download the MS‑COCO 2014 Annotation Dataset
Download the MS‑COCO 2014 annotation dataset and extract 10 000 prompts.  
This will create `prompt/prompt.txt`:
```bash
cd prompt
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
python extract_prompt.py
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
This will create `outputs/uni_pc_200_{scale}/{0-31}.pt`.
```bash
./sh/sample_target.sh
```

---

### Shape Optimization  
Learn shape parameters for **RBF‑Solver**.  
This will create shape files named `shape_dir/scale{scale}/NFE={NFE},p={order}.npz`.  
*Note: The supplementary zip file already contains the `.npz` files.*
```bash
./sh/shape_optim.sh
```

---

### Sample with RBF‑Solver  
Generate samples using RBF‑Solver:
```bash
./sh/sample_rbf.sh
```

---

### Sample with Other Samplers  
Generate comparison samples using baseline samplers:
```bash
./sh/sample_others.sh
```

---

### Extract CLIP Embeddings
Extract CLIP embeddings for all samples:
```bash
./sh/extract_clip.sh
```

---

### Show Results
Display RMSE and cosine‑similarity results:
```bash
./sh/show_results.sh
```
