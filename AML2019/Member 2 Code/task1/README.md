# task1

Brain Age Prediction

## Data imputation

Missing data was found in both the training data and the test data-set. After comparison, the median of each feature was used for replacement.

Data is then standardized.

## Feature selection

Two simple data transformation methods including `SelectKBest` and `PCA` were compared. In the round of comparison, `PCA` was out-performed by `SelectKBest`.

200 out of 832 in the original data was selected empirically.

## Model

Multiple regression were compared using cross-validation score on training data-set. `Ridge` was used as baseline, `SVR` performs badly on this task. In terms of the balance between run-time and accuracy, `LGBMRegressor` is probably the best one. Tree-base algorithm seems to have the best performance and the final prediction comes from the `LGBMRegressor`.

## Data transformation

Standardization of data was performed after imputation and low variance ones were filtered out. Test data set were transformed using the same pipeline fitted by training data.

