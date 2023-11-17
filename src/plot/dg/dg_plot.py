import pandas as pd

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.plot.dg.dg import logpa
from src.plot.dg.utils import dg_pa_dataframe

#group = 'erm_irm_lisa_dg_replicate'
entity='malvai'
project='cov_pa'
group = 'new_code'
df_dir = 'results/dataframes/dg/' + group + '/'
pic_dir = 'results/plots/dg/' + group + '/'

exp_names = ['erm_new', 'irm_new', 'lisa_new']

df_list = []
for exp_name in exp_names:
    df = dg_pa_dataframe(
        project = entity + '/' + project,
        filters= {
                "group": group,
                "$and": [{"tags": exp_name}, {"tags": "adam"}]
            },
        dirname = df_dir,
        cache = False)
    df_list.append(df)
    #logpa(df)
    
df = pd.concat(df_list, axis=0)
print(df)
logpa(df, pic_dir)

    


