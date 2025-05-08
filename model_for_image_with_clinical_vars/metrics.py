
from monai.metrics.utils import MetricReduction
from monai.metrics import ROCAUCMetric, DiceMetric, ConfusionMatrixMetric 


## Metric 
dice_metric    = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=False)
dice_metric1   = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=False)
confuse_metric = ConfusionMatrixMetric(include_background=True, metric_name=["f1 score", "accuracy", "sensitivity", "specificity",], compute_sample=False, reduction=MetricReduction.MEAN, get_not_nans=False)
auc_metric     = ROCAUCMetric() # Average Macro