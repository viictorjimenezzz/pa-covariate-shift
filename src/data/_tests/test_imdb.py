import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.imdb_dataset import IMDB_PADataset

multienv_ds = IMDB_PADataset(perturbation='adversarial', intensity=10)
import ipdb; ipdb.set_trace()