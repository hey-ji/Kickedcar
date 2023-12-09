library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(vcd)
library(discrim)
install.packages("dbarts")
library(dbarts)

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

# Finalize workflow and predict
final_wf <-
  wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=training)

final_wf %>%
  predict(new_data = test)

# Predict
boost_predictions <- predict(final_wf,
                             new_data=test,
                             type="class") %>%
  bind_cols(test) %>%
  rename(IsBadBuy=.pred_1) %>%
  select(ID, IsBadBuy)

vroom_write(x = boost_predictions, file = "Bart.csv", delim = ",")