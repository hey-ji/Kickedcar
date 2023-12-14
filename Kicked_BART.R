library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(vcd)
library(discrim)
install.packages("dbarts")
library(dbarts)

# Read in the data

setwd("/Users/student/Desktop/STAT348/Kickedcar")
training <-vroom("/Users/student/Desktop/STAT348/Kickedcar/training.csv", na=c("","NULL","NA")) %>%
  mutate(IsBadBuy = factor(IsBadBuy))
test <- vroom("/Users/student/Desktop/STAT348/Kickedcar/test.csv", na=c("","NULL","NA"))

# training <-vroom("./training.csv", na=c("","NULL","NA")) %>%
#   mutate(IsBadBuy = factor(IsBadBuy))
# test <- vroom("./test.csv", na=c("","NULL","NA"))

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

# BART --------------------------------------------------------------------

# Bart model
bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)

# Set up a tuning grid and folds
bart_tuneGrid <- grid_regular(trees(),
                              levels=10)
folds <- vfold_cv(training, v = 5, repeats=1)

# Tune it
bart_cv <- wf %>%
  tune_grid(resamples=folds,
            grid=bart_tuneGrid,
            metrics=metric_set(accuracy))

# Find the best Fit
bestTune <- bart_cv %>%
  select_best("accuracy")

bestTune
# Finalize workflow and predict
final_wf <-
  wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=training)

final_wf %>%
  predict(new_data = test)

# Predict
bart_predictions <- final_wf %>%
  predict(new_data = test, type="prob") %>%
  bind_cols(test) %>%
  rename(IsBadBuy=.pred_1) %>%
  select(RefId, IsBadBuy)

vroom_write(x = bart_predictions, file = "Bart.csv", delim = ",")