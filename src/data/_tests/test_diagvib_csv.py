"""
TEST 1: Checking that datasets loaded for PA are sample-corresponding if they have
been generated like that.
"""
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components import MultienvDataset
from src.data._tests.utils import _load_plot_diagvib

"""
1. Train datasets in modelselection/_debug_train should NOT be sample-corresponding.
"""

cache_0 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper/train_env0.pkl"
ds_0, _plot_0 = _load_plot_diagvib(cache_0)

cache_1 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper/train_env1.pkl"
ds_1, _plot_1 = _load_plot_diagvib(cache_1)

pa_ds = MultienvDataset([ds_0, ds_1])
its = pa_ds.__getitems__([0,31,32,63])

import ipdb; ipdb.set_trace()

cache_0 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper_nonpaired/train_env0.pkl"
ds_0, _plot_0 = _load_plot_diagvib(cache_0)

cache_1 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper_nonpaired/train_env1.pkl"
ds_1, _plot_1 = _load_plot_diagvib(cache_1)

pa_ds = MultienvDataset([ds_0, ds_1])
its = pa_ds.__getitems__([0,31,32,63])
import ipdb; ipdb.set_trace()

"""
2. Validation datasets in modelselection/_debug (for example) should be sample-corresponding.
"""

cache_0 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper/val_env0.pkl"
ds_0, _plot_0 = _load_plot_diagvib(cache_0)

cache_1 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper/val_env1.pkl"
ds_1, _plot_1 = _load_plot_diagvib(cache_1)

pa_ds = MultienvDataset([ds_0, ds_1])
its = pa_ds.__getitems__([0,31,32,63])


import ipdb; ipdb.set_trace()

cache_0 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper_nonpaired/val_env0.pkl"
ds_0, _plot_0 = _load_plot_diagvib(cache_0)

cache_1 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper_nonpaired/val_env1.pkl"
ds_1, _plot_1 = _load_plot_diagvib(cache_1)

pa_ds = MultienvDataset([ds_0, ds_1])
its = pa_ds.__getitems__([0,31,32,63])

import ipdb; ipdb.set_trace()

"""
2. Test datasets in modelselection/_hue_test should be sample-corresponding, in the sense that the same samples are being tested
with just different levels of shift.
"""

cache_0 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper/test_env0.pkl"
ds_0, _plot_0 = _load_plot_diagvib(cache_0)

cache_1 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper/test_env1.pkl"
ds_1, _plot_1 = _load_plot_diagvib(cache_1)

cache_2 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper/test_env2.pkl"
ds_2, _plot_2 = _load_plot_diagvib(cache_2)

cache_3 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper/test_env3.pkl"
ds_3, _plot_3 = _load_plot_diagvib(cache_3)

cache_4 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper/test_env4.pkl"
ds_4, _plot_4 = _load_plot_diagvib(cache_4)

cache_5 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper/test_env5.pkl"
ds_5, _plot_5 = _load_plot_diagvib(cache_5)

pa_ds = MultienvDataset([ds_0, ds_1, ds_2, ds_3, ds_4, ds_5])
its = pa_ds.__getitems__([0,31,32,63])

import ipdb; ipdb.set_trace()

cache_0 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper_nonpaired/test_env0.pkl"
ds_0, _plot_0 = _load_plot_diagvib(cache_0)

cache_1 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper_nonpaired/test_env1.pkl"
ds_1, _plot_1 = _load_plot_diagvib(cache_1)

cache_2 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper_nonpaired/test_env2.pkl"
ds_2, _plot_2 = _load_plot_diagvib(cache_2)

cache_3 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper_nonpaired/test_env3.pkl"
ds_3, _plot_3 = _load_plot_diagvib(cache_3)

cache_4 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper_nonpaired/test_env4.pkl"
ds_4, _plot_4 = _load_plot_diagvib(cache_4)

cache_5 = r"/cluster/home/vjimenez/adv_pa_new/data/dg/dg_datasets/diagvib_datashift/_plot_paper_nonpaired/test_env5.pkl"
ds_5, _plot_5 = _load_plot_diagvib(cache_5)

pa_ds = MultienvDataset([ds_0, ds_1, ds_2, ds_3, ds_4, ds_5])
its = pa_ds.__getitems__([0,31,32,63])

import ipdb; ipdb.set_trace()



