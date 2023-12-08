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
my_recipe <- recipe(IsBadBuy~., data= training) %>%
  update_role(RefId, new_role = 'ID') %>% #make it into ID so it's not a predictor
  update_role_requirements("ID", bake = FALSE) %>% #
  step_mutate(IsBadBuy = factor(IsBadBuy), skip = TRUE) %>% # It converts the variable IsBadBuy to a factor
  step_mutate(IsOnlineSale = factor(IsOnlineSale)) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_rm(contains('MMR')) %>%
  step_rm(BYRNO, WheelTypeID, BYRNO, WheelTypeID, VehYear, VNST, VNZIP1, PurchDate, # these variables don't seem very informative, or are repetitive
          AUCGUART, PRIMEUNIT, # these variables have a lot of missing values
          Model, SubModel, Trim) %>%
  step_corr(all_numeric_predictors(), threshold = .7) %>%
  step_other(all_nominal_predictors(), threshold = .0001) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors())

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

