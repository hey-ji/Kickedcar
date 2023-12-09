library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(vcd)
library(discrim)

# Read in the data
training <-vroom("./training.csv", na=c("","NULL","NA")) %>%
  mutate(IsBadBuy = factor(IsBadBuy))
test <- vroom("./test.csv", na=c("","NULL","NA"))

# Recipe
my_recipe <- recipe(IsBadBuy ~ ., data = training) %>%
  step_novel(all_nominal_predictors(), -all_outcomes()) %>%
  step_unknown(all_nominal_predictors()) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = .7) %>%
  step_zv() %>%
  step_normalize(all_numeric_predictors())

prep <- prep(my_recipe)
bake(prep, new_data=training)

# Penalized Logistic Regression -------------------------------------------
penalized_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(penalized_mod)

# Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

# Split data for CV
folds <- vfold_cv(training, v = 5, repeats=1)

#  Run the CV
CV_results <- wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL, roc_auc, f_meas, sens, recall, spec,22

# Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")


# Finalize the Workflow & fit it
final_wf <-
  wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=training)

# Predict
penalized_predictions <- final_wf %>%
  predict(new_data = test, type="prob") %>%
  bind_cols(test) %>%
  rename(IsBadBuy=.pred_1) %>%
  select(RefId, IsBadBuy)

vroom_write(x = penalized_predictions, file = "PenalizedLogisticRegression.csv", delim = ",")

