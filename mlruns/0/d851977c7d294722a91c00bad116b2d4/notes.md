I tried to get something similar to the implementation of `mlxtend`'s [`StackingCVRegressor`](http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.regressor/#stackingcvregressor) but with a CV-splitting strategy more appropriate for time-series data. I split the data in incrementing chunks of test years, with all prior data being included in the train years. The number of chunks were equal to the `cv` param and were roughly equal, with any remainder being included in the final chunk.

I tried some variations:
- Increasing the `cv` value to reduce chunk size
- Using KFold (with `shuffle=True`)
- Making the train sets equal size to the test sets rather than growing with each chunk

All the above had similar performance: they were all significantly worse than training sub-models on the full train set before feeding predictions into the meta-model.

Increasing the `cv` to 20 seemed to help slightly, which makes me think the problem is the number of seasons being predicted at a time. Perhaps models don't generalize well far into the future. The fix would be to have a fold per season, but this would blow out training time to hours for even a relatively small model, which wouldn't be worth the trouble.