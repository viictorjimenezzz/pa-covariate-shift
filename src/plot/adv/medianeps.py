import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
from secml.data.loader import CDataLoaderCIFAR10
from src.data.utils import carray2tensor

model_var = ["Standard", "BPDA", "Engstrom2019Robustness", "Wong2020Fast", "Addepalli2021Towards_RN18", "Wang2023Better_WRN-28-10"]
file_path = '/cluster/home/vjimenez/adv_pa_new/results/plots/adv/model_linfs.txt'
f = open(file_path, 'w')
for model in model_var:
    datapath = f"/cluster/home/vjimenez/adv_pa_new/data/adv/adv_datasets/model={model}_attack=FMN_steps=1000.pt"

    dset = CDataLoaderCIFAR10
    _, ts = dset().load(val_size=0)
    X, Y = ts.X / 255.0, ts.Y
    X = carray2tensor(X, torch.float32)
    dset_size = X.shape[0]

    linf_data = []
    for adversarial_ratio in torch.arange(0, 1.1, 0.1):
        adv_X = torch.load(datapath)

        if adversarial_ratio == 0.0:
            adv_X = X

        split = int(adversarial_ratio * dset_size)
        attack_norms = (adv_X - X).norm(p=float("inf"), dim=1)

        _, unpoison_ids = attack_norms.topk(dset_size - split)

        # remove poison for the largest 1 - adversarial_ratio attacked ones
        adv_X[unpoison_ids] = X[unpoison_ids]

        linf = torch.norm(adv_X - X, p=float("inf"), dim=1)
        #import ipdb; ipdb.set_trace()
        #print(linf.median().item()*255, linf.max().item()*255)
        linf_data.append((linf.median().item()*255, linf.max().item()*255))

    table_header = "\nAR\tMedian linf\tMax linf\n"
    table_rows = [f"{i/10:.1f}\t{median:.2f}\t{max_linf:.2f}" for i, (median, max_linf) in enumerate(linf_data)]
    table = table_header + "\n".join(table_rows)
    f.write(f"\n\nModel: {model}")
    f.write(table)
        





