library(tidyverse)
library(data.table)
library(randomForest)
library(MASS)
library(pROC)
library(xgboost)

data <- fread("Data/train.csv")
test <- fread("Data/test.csv")

# Data description --------------------------------------------------------

# Na in each column
table(colSums(is.na(data)))

# Balance of the data
data %>%
  group_by(target) %>% 
  summarise(n = n())

# mean, sd, min, max by group.
data_means <- aggregate(data[, 3:ncol(data)], list(data$target), mean)
data_means %>% 
  melt(id.vars = "Group.1") %>% 
  ggplot()+
  geom_density(aes( x = value, fill = as.factor(Group.1)), alpha = 0.5)
  
data_sd <- aggregate(data[, 3:ncol(data)], list(data$target), sd)
data_sd %>% 
  melt(id.vars = "Group.1") %>% 
  ggplot()+
  geom_density(aes( x = value, fill = as.factor(Group.1)), alpha = 0.5)

data_max <- aggregate(data[, 3:ncol(data)], list(data$target), max)
data_max %>% 
  melt(id.vars = "Group.1") %>% 
  ggplot()+
  geom_density(aes( x = value, fill = as.factor(Group.1)), alpha = 0.5)

data_min <- aggregate(data[, 3:ncol(data)], list(data$target), min)
data_min %>% 
  melt(id.vars = "Group.1") %>% 
  ggplot()+
  geom_density(aes( x = value, fill = as.factor(Group.1)), alpha = 0.5)

data_corr <- round(cor(data[, -c(1,2)]), 2)
data_corr[lower.tri(data_corr, diag = T)] <-  NA
data_corr <- as.numeric(data_corr)
hist(data_corr, breaks = "FD", probability = T)



# ks-test -----------------------------------------------------------------


ks_tests <- data %>% 
  dplyr::select(-id) %>% 
  gather(key = variable, value = value,  -target) %>% 
  group_by(target, variable) %>% 
  summarise(value = list(value)) %>% 
  spread(target, value) %>% 
  group_by(variable) %>% 
  mutate(p_value = ks.test(unlist(`0`), unlist(`1`))$p.value,
         t_value = ks.test(unlist(`0`), unlist(`1`))$statistic) %>% 
  filter(p_value < 0.05)
  
variables_ks <- ks_tests$variable



# Mean --------------------------------------------------------------------

# It does not help for prediction purposes
means <- data %>% 
  dplyr::select(-id) %>% 
  gather(key = variable, value = value,  -target) %>% 
  group_by(target, variable) %>% 
  summarise(value = list(value)) %>% 
  spread(target, value) %>% 
  group_by(variable) %>% 
  mutate(p_value = t.test(unlist(`0`), unlist(`1`))$p.value,
         t_value = t.test(unlist(`0`), unlist(`1`))$statistic)%>% 
  filter(p_value < 0.05)


variables_means <- means$variable 

final_variables <- intersect(variables_ks, variables_means)
# Random Forest -----------------------------------------------------------

randomForest(y = as.factor(data$target), x = data[, 3:ncol(data)], xtest = test[, 2:ncol(test)], importance = T)

rf <- randomForest(y = as.factor(data$target),
                   x = data[, ..variables_ks], 
                   xtest = test[, ..variables_ks], 
                   importance = T,
                   ntree = 500
                   )

plot(rf$y,  rf$votes[,2])
rf_roc <- roc(rf$y, rf$votes[,2])
plot(rf_roc)
auc(rf_roc)


# XGBOOST -----------------------------------------------------------------

variables <-c("target", variables_ks) 
train <- as.matrix(data[, ..variables])
xg_data <- xgb.DMatrix(data = train[, -1], label = train[, 1])

xg_fit <- xgb.cv(data = xg_data, 
        nrounds = 200,
        max_depth = 2,
        nfold = 10,
        objective = "binary:logistic",
        metrics = "auc")

xg_fit <- xgboost(data = xg_data, 
                 nrounds = 200,
                 max_depth = 2,
                 objective = "binary:logistic",
                 eval_metric = "auc")

test_results <- rf$test$predicted
plot(rf)


test_results <- predict(xg_fit, as.matrix(test[, ..final_variables]))

submission <- data.frame(id = 250:19999, target = test_results)

write.csv(submission, "submission_7.csv", row.names = F)
