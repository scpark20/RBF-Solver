"""Explicit, device-bound capacity records shared across NFEs."""
import json, os
import torch
from experiment import atomic_json

def environment(worker):
    p=torch.cuda.get_device_properties(worker.device)
    return dict(device=p.name,total_memory=p.total_memory,
                capability=list(torch.cuda.get_device_capability(worker.device)),
                torch=str(torch.__version__),cuda=torch.version.cuda,
                allocator=os.environ.get("PYTORCH_ALLOC_CONF",os.environ.get("PYTORCH_CUDA_ALLOC_CONF","")))

def reuse(worker,job):
    path=worker.out/f"capacity_gpu{worker.gpu}.json"
    if not path.exists():
        raise RuntimeError("Calibrate this device explicitly with calibrate_capacity.py before sampling; no per-NFE search")
    cache=json.loads(path.read_text())
    if cache.get("environment")!=environment(worker) or cache.get("plan_sha256")!=worker.hash:
        raise RuntimeError("Capacity belongs to another environment/protocol; run explicit calibration in the new environment")
    record=cache["methods"].get(job["method"])
    if not record:raise RuntimeError("No calibrated capacity for "+job["method"])
    batch=record["batch_size"]
    if not isinstance(batch,int) or not 1<=batch<=worker.p["num_samples"]:raise ValueError("Invalid capacity")
    worker.decode_limit=min(worker.decode_limit,record["decode_chunk"])
    worker.feature_limit=min(worker.feature_limit,record["fid_chunk"])
    atomic_json(worker.out/job["id"]/f"batch_reuse_gpu{worker.gpu}.json",
                dict(record,batch_size=batch,plan_sha256=worker.hash,environment=cache["environment"],
                     scope="One explicit device calibration reused across NFEs; actual batches count toward FID"))
    return batch

def recover(worker,job,failed_batch):
    path=worker.out/f"capacity_gpu{worker.gpu}.json"
    cache=json.loads(path.read_text())
    if cache.get("environment")!=environment(worker) or cache["plan_sha256"]!=worker.hash:
        raise RuntimeError("Incompatible recovery capacity")
    record=cache["methods"][job["method"]]
    record["failed"]=sorted(set(record["failed"])|{failed_batch})
    feasible=[b for b in record["passed"] if b<min(record["failed"])]
    if not feasible:raise RuntimeError("No smaller verified batch is available; preserve progress and recalibrate explicitly")
    record["batch_size"]=max(feasible)
    atomic_json(path,cache)
    return record["batch_size"]
