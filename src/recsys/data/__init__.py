# Import datamodules to register datasets
from recsys.data.datamodules import movielens  # noqa: F401
from recsys.data.datamodules import kuairec as _kuairec_dm  # noqa: F401
from recsys.data.datamodules import kuairand as _kuairand_dm  # noqa: F401
