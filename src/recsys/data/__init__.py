# Import datamodules to register datasets
from recsys.data.datamodules import movielens  # noqa: F401
from recsys.data.datamodules import kuairec as _kuairec_dm  # noqa: F401
from recsys.data.datamodules import kuairand as _kuairand_dm  # noqa: F401
from recsys.data.datamodules import amazon as _amazon_dm  # noqa: F401
from recsys.data.datamodules import frappe as _frappe_dm  # noqa: F401
from recsys.data.datamodules import taobao_ad as _taobao_ad_dm  # noqa: F401
from recsys.data.datamodules import microvideo as _microvideo_dm  # noqa: F401
from recsys.data.datamodules import kuaivideo as _kuaivideo_dm  # noqa: F401
