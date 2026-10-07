"""Explicit one-time sampling capacity measurement using the existing worker."""
import argparse, os, json, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu",type=int,required=True)
    p.add_argument("--kind",choices=["baseline","rbf","followup"],required=True)
    p.add_argument("--method",choices=["dpmpp","unipc","rbf"],required=True)
    p.add_argument("--cfg",type=float,required=True)
    p.add_argument("--nfe",type=int,required=True,help="A representative approved NFE; use the largest required history")
    p.add_argument("--run-baseline",action="store_true",help="After calibration explicitly sample this baseline condition, including FID")
    a=p.parse_args()
    if a.run_baseline and a.kind!="baseline":p.error("--run-baseline is only valid for baseline workers")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    from experiment import jobs,atomic_json
    from capacity import environment
    from benchmark import Worker,locked
    import torch
    if a.kind=="baseline":
        from launch import ensure_inputs
        from experiment import load_plan
        ensure_inputs(load_plan(True))
        worker=Worker(a.gpu)
    elif a.kind=="rbf":
        from rbf_pipeline import SamplingWorker
        worker=SamplingWorker(a.gpu)
    else:
        from rbf_followup import FollowupSamplingWorker
        worker=FollowupSamplingWorker(a.gpu)
    if a.gpu not in worker.p["gpus"]:raise ValueError("GPU not assigned to this plan")
    job=next((j for j in jobs(worker.p) if (j["method"],j["cfg"],j["nfe"])==(a.method,a.cfg,a.nfe)),None)
    if job is None:raise ValueError("Condition is not in the approved plan")
    output=worker.out
    if a.run_baseline and (output/job["id"]/"result.json").exists():
        raise ValueError("Completed condition exists; preserve it and use a new run directory")
    with locked(output/f"capacity_gpu{a.gpu}.lock"):
        record_path=output/f"capacity_gpu{a.gpu}.json"
        identity=environment(worker)
        cache=json.loads(record_path.read_text()) if record_path.exists() else {}
        if cache.get("environment")!=identity or cache.get("plan_sha256")!=worker.hash:
            cache=dict(plan_sha256=worker.hash,gpu=a.gpu,environment=identity,methods={})
        # Preserve old evidence. Calibration trials never overwrite condition results.
        dest=output/"capacity_calibration"/f"gpu{a.gpu}_{time.time_ns()}"
        (dest/job["id"]).mkdir(parents=True)
        worker.out=dest
        with torch.inference_mode():
            batch=worker.tune(job)
            z=worker.latent(job,0,batch)
            worker.postprocess(z,update=False)
            del z
        cache["methods"][a.method]=dict(batch_size=batch,
            passed=sorted({r["batch"] for r in worker.trials if r["passed"]}),
            failed=sorted({r["batch"] for r in worker.trials if not r["passed"]}),
            decode_chunk=worker.decode_limit,fid_chunk=worker.feature_limit,
            source=str(dest/job["id"]/f"batch_gpu{a.gpu}.json"))
        if record_path.exists():
            record_path.rename(dest/"previous_capacity.json")
        atomic_json(record_path,cache)
        print(json.dumps(dict(gpu=a.gpu,method=a.method,batch_size=batch,record=str(record_path))))
        worker.out=output
        if a.run_baseline:
            with torch.inference_mode():worker.run_job(job)
if __name__=="__main__":main()
