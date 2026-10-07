"""Create an isolated execution namespace; never rewrite existing artifact identities."""
import argparse,json,shutil,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",choices=["sit","1rf"],required=True)
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--gpus",type=int,nargs="+",required=True)
    p.add_argument("--apply",action="store_true",help="Default is a preview; apply only in a separate checkout")
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];root=repo/a.project
    base=Path("/data/RBF-Solver/results")/("1rf" if a.project=="1rf" else "")
    dest=a.results.resolve()
    if dest.exists() or not dest.is_relative_to(base) or dest==base:raise ValueError("Choose a new empty child of the project's result root")
    if not a.gpus or len(set(a.gpus))!=len(a.gpus) or min(a.gpus)<0:raise ValueError("Use unique visible GPU indices")
    path_keys={"output","training_output","input_bank","batch_sources","retrain_of"}
    changes=[]
    def move(v):
        if isinstance(v,str) and v.startswith(str(base)+"/"):return str(dest)+v[len(str(base)):]
        if isinstance(v,list):return [move(x) for x in v]
        return v
    for file in sorted(root.glob("*.json")):
        old=json.loads(file.read_text())
        if not isinstance(old,dict) or "output" not in old or "gpus" not in old:continue
        new={k:move(v) if k in path_keys else a.gpus if k=="gpus" else v for k,v in old.items()}
        changes.append((file,old,new))
    if not changes:raise ValueError("No experiment plans")
    print(json.dumps({str(f.relative_to(repo)):{k:{"before":o[k],"after":n[k]} for k in n if n[k]!=o[k]} for f,o,n in changes},indent=2))
    if a.apply:
        # Config backup is code, so it stays inside the allowed project directory.
        backup=root/".history"/("environment_"+str(time.time_ns()));backup.mkdir(parents=True)
        for file,old,new in changes:
            shutil.copy2(file,backup/file.name)
            file.write_text(json.dumps(new,indent=2,ensure_ascii=False)+"\n")
        dest.mkdir(parents=True)
        (dest/"configuration_changes.json").write_text(json.dumps(dict(project=a.project,gpus=a.gpus,scope="Only output references and GPU assignment changed"),indent=2)+"\n")
if __name__=="__main__":main()
