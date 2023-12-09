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
  select(ID, IsBadBuy)
vroom_write(x = rf_predictions, file = "RandomForest.csv", delim = ",")
