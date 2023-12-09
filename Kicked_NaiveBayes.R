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

# Naive Bayes -------------------------------------------------------------

# create a workflow with model & recipe
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

# set up tuning grid and folds

folds <- vfold_cv(training, v = 5, repeats=1)

nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness())

## Tune smoothness and Laplace here
nb_cv <- wf %>%
  tune_grid(resamples=folds,
            grid=nb_tuning_grid,
            metrics=metric_set(accuracy))

## Find the best Fit
bestTune <- nb_cv %>%
  select_best("accuracy")

# Finalize workflow and predict
final_wf <-
  wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=training)

final_wf %>%
  predict(new_data = test)

## Predict
bayes_predictions <- predict(final_wf,
                             new_data=test,
                             type="class") %>%
  bind_cols(test) %>%
  rename(IsBadBuy=.pred_1) %>%
  select(ID, IsBadBuy)

vroom_write(x = bayes_predictions, file = "NaiveBayes.csv", delim = ",")

