import json,os
from pathlib import Path
from torch.utils.data import DataLoader
from trajdata import UnifiedDataset
repo=Path("/ocean/projects/tra250008p/slin24/MetaLearning/adaptive-prediction")
data_root=Path("/ocean/projects/tra250008p/slin24/datasets/nuscenes")
cache_dir=repo.joinpath("data/trajdata_cache")
cache_dir.mkdir(parents=True,exist_ok=True)
splits=["nusc_trainval-train","nusc_trainval-train_val","nusc_trainval-val"]
info={}
for s in splits:
    ds=UnifiedDataset(desired_data=[s],data_dirs={"nusc_trainval":str(data_root)})
    dl=DataLoader(ds,batch_size=8,shuffle=True,collate_fn=ds.get_collate_fn(),num_workers=4)
    b=next(iter(dl))
    info[s]={"dataset_len":len(ds),"batch_agents":int(getattr(b,"num_agents",getattr(b,"agent_name",[]).__len__()))}
conf={"trajdata_cache_dir":str(cache_dir),"data_loc_dict":{"nusc_trainval":str(data_root)},"train_data":"nusc_trainval-train","eval_data":"nusc_trainval-train_val"}
out=repo.joinpath("config/datasets/nuscenes_trajdata.json")
out.write_text(json.dumps(conf,indent=2))
print(json.dumps({"cache_dir":str(cache_dir),"splits":info,"conf_path":str(out)},indent=2))
