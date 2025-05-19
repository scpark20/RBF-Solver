import os
import torch
import torch.nn.functional as F

def compare(ref_dir, dst_dir):
    mse_losses = []
    cos_sims = []
    for i in range(2500):
        ref_pt = os.path.join(ref_dir, f"{i}.pt")
        dst_pt = os.path.join(dst_dir, f"{i}.pt")
        if not (os.path.exists(ref_pt) and os.path.exists(dst_pt)):
            continue
        ref_data = torch.load(ref_pt)
        dst_data = torch.load(dst_pt)
        mse_loss = torch.mean(F.mse_loss(ref_data['sample_raw'], dst_data['sample_raw'], reduction='none'), dim=[1, 2, 3])
        mse_losses += mse_loss
        cos_sim = torch.cosine_similarity(ref_data['clip_features'], dst_data['clip_features'])
        cos_sims += cos_sim

    return torch.sqrt(torch.mean(torch.tensor(mse_losses))), torch.mean(torch.tensor(cos_sims))

model_names = ['dpm_solver++', 'uni_pc', 'rbf_solver']
orders = [2, 3]
nfes = [5, 6, 8, 10, 12, 15, 20]
scales = [1.5, 3.5, 5.5, 7.5, 9.5]

for scale in scales:
    ref_dir = f'outputs/uni_pc_2_200_{scale}'
    for order in orders:
        for nfe in nfes:
            for model_name in model_names:
                dst_dir = f"outputs/{model_name}_{order}_{nfe}_{scale}"
                if not os.path.exists(dst_dir):
                    continue
                rmse, cos_sim = compare(ref_dir, dst_dir)
                print(f"SCALE: {scale}, NFE: {nfe}, MODEL: {model_name}, RMSE: {rmse:0.4f}, COSSIM: {cos_sim:0.4f}")