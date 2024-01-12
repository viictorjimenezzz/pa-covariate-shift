import pandas as pd

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.plot.dg.dg import logpa
from src.plot.dg.utils import dg_pa_dataframe

entity='malvai'
project='cov_pa'
group = 'pa_lightningdebug'
df_dir = 'results/dataframes/dg/' + group + '/'
pic_dir = 'results/plots/dg/' + group + '/'

exp_names = ['erm_rebuttal', 'irm_rebuttal', 'lisa_rebuttalD']

df_list = []
for exp_name in exp_names:
    df = dg_pa_dataframe(
        project = entity + '/' + project,
        filters= {
                "group": group,
                "$and": [{"tags": exp_name}, {"tags": "adam"}]
            },
        dirname = df_dir,
        cache = True)
    df_list.append(df)
    
df = pd.concat(df_list, axis=0)
logpa(df, pic_dir, show_acc=False, picformat="pdf")

    


