"""Inventory and verify external experiment assets without loading pickle files."""
import argparse, hashlib, json, os
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""):h.update(chunk)
    return h.hexdigest()

def resolve(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError("Path outside asset root: "+name)
    return path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["inventory","verify"])
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--reference-root",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    args=parser.parse_args()
    roots={"data":args.data_root.resolve(),"reference":args.reference_root.resolve()}
    if args.mode=="inventory":
        records=[]
        for section in ("weights","results"):
            for path in sorted((roots["data"]/section).rglob("*")):
                if not path.is_file():continue
                name=path.relative_to(roots["data"]).as_posix()
                if name.startswith("results/handoff/") or "/.cache/" in name:continue
                if path.is_symlink():
                    resolve(roots["data"],name)
                records.append(dict(root="data",path=name,size=path.stat().st_size,sha256=sha256(path)))
        for name in ("VIRTUAL_imagenet256_labeled.npz","cifar10_stats.npz"):
            path=resolve(roots["reference"],name)
            records.append(dict(root="reference",path=name,size=path.stat().st_size,sha256=sha256(path)))
        args.manifest.parent.mkdir(parents=True,exist_ok=True)
        args.manifest.write_text(json.dumps(dict(schema=1,files=records),indent=2)+"\n")
        print(json.dumps(dict(files=len(records),bytes=sum(r["size"] for r in records))))
    else:
        records=json.loads(args.manifest.read_text())["files"]
        errors=[]
        for record in records:
            path=resolve(roots[record["root"]],record["path"])
            if not path.is_file():errors.append(record["path"]+": missing");continue
            if path.stat().st_size!=record["size"] or sha256(path)!=record["sha256"]:
                errors.append(record["path"]+": checksum mismatch")
        if errors:raise SystemExit("\n".join(errors))
        print(json.dumps(dict(verified=len(records),bytes=sum(r["size"] for r in records))))
if __name__=="__main__":main()
