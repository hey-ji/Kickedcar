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
my_recipe <- recipe(IsBadBuy ~ ., data = train) %>%
  step_novel(all_nominal_predictors(), -all_outcomes()) %>%
  step_unknown(all_nominal_predictors()) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = .7) %>%
  step_zv() %>%
  step_normalize(all_numeric_predictors())

prep <- prep(my_recipe)
bake(prep, new_data=training)

# Random Forest -----------------------------------------------------------

# model for random forest
forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 100) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Create a workflow with model & recipe
wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(forest_mod) %>%
  fit(data = training)

# Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range = c(1,ncol(training)-1)),
                            min_n(),
                            levels = 5)

# Set up K-fold CV
folds <- vfold_cv(training, v = 10, repeats=1)

# run the CV
CV_results <- wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

# Find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf <-
  wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=training)

final_wf %>%
  predict(new_data = test)

# Formatting for submission
rf_predictions <- final_wf %>%
  predict(new_data = test, type="prob") %>%
  bind_cols(test) %>%
  rename(IsBadBuy=.pred_1) %>%
  select(RefId, IsBadBuy)

vroom_write(x = rf_predictions, file = "RandomForest.csv", delim = ",")
