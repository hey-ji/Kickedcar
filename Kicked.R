library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(vcd)
library(discrim)

# random forest(stacking), naive bayes(stacking), BART(accuracy)

# Recipe
# my_recipe <- recipe(IsBadBuy ~ ., data = train) %>%
#   step_novel(all_nominal_predictors(), -all_outcomes()) %>%
#   step_unknown(all_nominal_predictors()) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
#   step_impute_mean(all_numeric_predictors()) %>%
#   step_corr(all_numeric_predictors(), threshold = .7) %>%
#   step_zv() %>%
#   step_normalize(all_numeric_predictors())

# Read in the data
setwd("/Users/student/Desktop/STAT348/Kickedcar")
kicked_training <-vroom("/Users/student/Desktop/STAT348/Kickedcar/training.csv", na=c("","NULL","NA")) %>%
    mutate(IsBadBuy = factor(IsBadBuy))
kicked_test <- vroom("/Users/student/Desktop/STAT348/Kickedcar/test.csv", na=c("","NULL","NA"))

# kicked_training <-vroom("./training.csv", na=c("","NULL","NA")) %>%
#   mutate(IsBadBuy = factor(IsBadBuy))
# view(kicked_training)
# kicked_test <- vroom("./test.csv", na=c("","NULL","NA"))

my_recipe <- recipe(IsBadBuy~., data= kicked_training) %>%
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
bake(prep, new_data=kicked_training)


# Penalized Logistic Regression -------------------------------------------
penalized_logistic_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
   set_engine("glmnet")
 
 kicked_workflow <- workflow() %>%
   add_recipe(my_recipe) %>%
   add_model(penalized_logistic_mod)

# Grid of values to tune over
 tuning_grid <- grid_regular(penalty(),
                             mixture(),
                             levels = 5) ## L^2 total tuning possibilities
 
# Split data for CV
 folds <- vfold_cv(kicked_training, v = 5, repeats=1)

#  Run the CV
 CV_results <- kicked_workflow %>%
   tune_grid(resamples=folds,
             grid=tuning_grid,
             metrics=metric_set(roc_auc)) #Or leave metrics NULL, roc_auc, f_meas, sens, recall, spec,22

# Find Best Tuning Parameters
 bestTune <- CV_results %>%
   select_best("roc_auc")
 bestTune

 # Finalize the Workflow & fit it
 final_wf <-
   kicked_workflow %>%
   finalize_workflow(bestTune) %>%
   fit(data=kicked_training)
 
# Predict
 penalized_predictions <- final_wf %>%
   predict(new_data = kicked_test, type="prob") %>%
   bind_cols(kicked_test) %>%
   rename(IsBadBuy=.pred_1) %>%
   select(RefId, IsBadBuy)

  vroom_write(x = penalized_predictions, file = "PenalizedLogisticRegression.csv", delim = ",")


# Random Forest -----------------------------------------------------------

# model for random forest
forest_mod <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 100) %>%
    set_engine("ranger") %>%
    set_mode("classification")
  
# Create a workflow with model & recipe
rf_workflow <- workflow() %>%
    add_recipe(my_recipe) %>%
    add_model(forest_mod) %>%
    fit(data = kicked_training)
  
# Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range = c(1,ncol(kicked_training)-1)),
                              min_n(),
                              levels = 5)
  
# Set up K-fold CV
folds <- vfold_cv(kicked_training, v = 10, repeats=1)
  
# run the CV
  CV_results <- rf_workflow %>%
    tune_grid(resamples=folds,
              grid=tuning_grid,
              metrics=metric_set(roc_auc))
  
# Find best tuning parameters
  bestTune <- CV_results %>%
    select_best("roc_auc")
  
# Finalize workflow and predict
  final_wf <-
    rf_workflow %>%
    finalize_workflow(bestTune) %>%
    fit(data=kicked_training)
  
  final_wf %>%
    predict(new_data = kicked_test)
 
# Formatting for submission
  rf_predictions <- final_wf %>%
    predict(new_data = kicked_test, type="prob") %>%
    bind_cols(kicked_test) %>%
    rename(IsBadBuy=.pred_1) %>%
    select(RefId, IsBadBuy)
  vroom_write(x = rf_predictions, file = "/Users/student/Desktop/STAT348/STAT348/RandomForest.csv", delim = ",")
  

# BART --------------------------------------------------------------------

# Bart model
  bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
    set_engine("dbarts") %>% # might need to install
    set_mode("classification")
  
  bart_wf <- workflow() %>%
    add_recipe(my_recipe) %>%
    add_model(bart_model)
  
# Set up a tuning grid and folds
  bart_tuneGrid <- grid_regular(trees(),
                                levels=10)
  folds <- vfold_cv(kicked_training, v = 5, repeats=1)
  
# Tune it
  bart_cv <- bart_wf %>%
    tune_grid(resamples=folds,
              grid=bart_tuneGrid,
              metrics=metric_set(accuracy))
  
# Find the best Fit
  bestTune <- bart_cv %>%
    select_best("accuracy")
  
# Finalize workflow and predict
  final_wf <-
    bart_wf %>%
    finalize_workflow(bestTune) %>%
    fit(data=kicked_training)
  
  final_wf %>%
    predict(new_data = kicked_test)
  
# Predict
  boost_predictions <- predict(final_wf,
                               new_data=kicked_test,
                               type="class") %>%
    bind_cols(kicked_test) %>%
    rename(IsBadBuy=.pred_1) %>%
    select(RefId, IsBadBuy)
  
  vroom_write(x = boost_predictions, file = "Bart.csv", delim = ",")
  
# Naive Bayes -------------------------------------------------------------

# create a workflow with model & recipe
  nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
    set_mode("classification") %>%
    set_engine("naivebayes") # install discrim library for the naivebayes eng
  
  nb_wf <- workflow() %>%
    add_recipe(my_recipe) %>%
    add_model(nb_model)
  
# set up tuning grid and folds
  
  folds <- vfold_cv(kicked_training, v = 5, repeats=1)
  
  nb_tuning_grid <- grid_regular(Laplace(),
                                 smoothness())
  
## Tune smoothness and Laplace here
  nb_cv <- nb_wf %>%
    tune_grid(resamples=folds,
              grid=nb_tuning_grid,
              metrics=metric_set(accuracy))
  
## Find the best Fit
  bestTune <- nb_cv %>%
    select_best("accuracy")
  
# Finalize workflow and predict
  final_wf <-
    nb_wf %>%
    finalize_workflow(bestTune) %>%
    fit(data=kicked_training)
  
  final_wf %>%
    predict(new_data = kicked_test)
  
## Predict
  bayes_predictions <- predict(final_wf,
                             new_data=kicked_test,
                             type="class") %>%
    bind_cols(kicked_test) %>%
    rename(IsBadBuy=.pred_1) %>%
    select(RefId, IsBadBuy)
  
vroom_write(x = bayes_predictions, file = "NaiveBayes.csv", delim = ",")
  
