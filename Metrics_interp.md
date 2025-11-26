### Interpretation of the metric for the baseline twin

The Seasonal Hourly Mean baseline achieved an MAE of 8.31 and an RMSE of 11.24, which correspond to approximately 4–6% of the parking lot capacity. These values indicate a reasonably accurate prediction performance for such a simple model.

The MAPE value (≈59%) is inflated due to very low occupancy values during early-night hours, where even small absolute errors result in large percentage deviations. For example, if the real occupation is 5 and the model predicts 12 it becomes -> error=12-5=7; MAPE =|7/5|=140%. Therefore, MAE and RMSE are more reliable indicators of performance for this task.