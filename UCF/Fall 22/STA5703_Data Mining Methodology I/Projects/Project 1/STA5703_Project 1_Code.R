# Loading Libraries
## basic
library(caret)
library(dplyr)
library(tidyr)
library(fastDummies)
library(xlsx)
library(broom)

## visualization
library(ggplot2)

## feature engineering
library(olsrr)          # to check multicollinearity
library(FactoMineR)     # PCA
library(factoextra)
library(psych)          # Varimax Rotation

## regressor
library(randomForest)
library(xgboost)


# Options
options(max.print=100000)
options(scipen=100)

# Importing Dataset
getwd()
project1_data <- read.csv('./UCF/Fall 22/STA5703_Data Mining Methodology I/Projects/Project 1/train.csv')

# Descriptive Analysis
head(project1_data)
dim(project1_data)
str(project1_data)
summary(project1_data)



################################## Data Pre-processing ##################################

processed_data <- project1_data

# Data Cleaning
colnames(project1_data) # the superconductivity data is a clean data

## checking for missing data
colSums(is.na(project1_data))/nrow(project1_data)


# Data Wrangling
## splitting data into training and testing
set.seed(906)
train_index <- createDataPartition(processed_data$critical_temp, p=0.7, list=FALSE, times=1)
train_data <- processed_data[train_index, ]
test_data <- processed_data[-train_index, ]

## histogram with density plot for 'Critical Temperature (K)'
ggplot(train_data, aes(x=critical_temp)) + 
  geom_histogram(aes(y=..density..), colour='black', fill='indianred3', binwidth=5) +
  labs(y = 'Density', x = 'Critical Temperature (K)')

mean(train_data$critical_temp)



################################## Feature Engineering ##################################

# Testing Multicollinearity
model_vif <- lm(critical_temp ~ ., data=train_data)
df_vif <- ols_vif_tol(model_vif)
df_vif <- df_vif[df_vif$VIF < 500, ]

ggplot(df_vif[1:40, 1:3], aes(Variables, VIF)) +
  geom_point() +
  ylim(0, 500) +
  labs(x='Features', title='Variable Inflation Factor (VIF)') +
  scale_x_discrete(guide = guide_axis(angle = 90)) +
  geom_hline(yintercept=10, linetype=2, color='indianred3')

# PCA with Varimax Rotation
## function to calculate Kaiser-Meyer-Olkin (KMO) and Bartlett's scores 
KMO_Bartlett <- function(data) {
  KMO_result <- KMO(data[ , 1:81]) # Kaiser-Meyer-Olkin factor adequacy
  KMO <- KMO_result$MSA
  
  Bartlett_result <- cortest.bartlett(data[ , 1:81]) # Bartlett's test
  Bartlett <- Bartlett_result$p.value
  
  return(list(KMO, Bartlett))
}

## function to visualize screeplot using 'FactoMineR'
scree_plot <- function(data, no_comp) {
  FactoMineR_PCA <- PCA(data, ncp=no_comp)
  sp <- fviz_screeplot(FactoMineR_PCA, choice='eigenvalue', ncp=no_comp, xlab='Principal Components', 
                       geom='line', linecolor='black', pointsize=2.5, ggtheme=theme_gray()) +
    geom_hline(yintercept=1, linetype=1, color='indianred3')
  
  return(sp)
}

## function to determine eigenvalue and variance of PCs along with PCs
eig_var_PCA <- function(data, no_comp) {
  FactoMineR_PCA <- PCA(data, ncp=no_comp)
  PCA <- as.data.frame(FactoMineR_PCA$ind$coord)
  FactoMineR_PCA_result <- as.data.frame(FactoMineR_PCA$eig)
  
  return(list(PCA, FactoMineR_PCA_result))
} 

## function to determine rotated loadings using 'Psych'
varimax_PCA <- function(data, no_comp) {
  psych_PCA <- principal(data, nfactors=no_comp, rotate='varimax', scores=TRUE)
  psych_PCA_result <- round(psych_PCA$loadings, 2)
  
  return(psych_PCA_result)
}

## train_data
### KMO and Barlett's test
KMO_Bartlett_train <- KMO_Bartlett(data=train_data)
KMO_Bartlett_train[1] 
KMO_Bartlett_train[2]

### train data scaling for PCA 
scaled_train_data <- scale(train_data[ , 1:81], center=TRUE)

### screeplot
scree_plot(data=scaled_train_data, no_comp=20)

### eigenvalue and variance of PCs
PCA_train <- eig_var_PCA(data=scaled_train_data, no_comp=20)
PCA_train_data <- PCA_train[[1]]
write.table(PCA_train[[2]], file='./UCF/Fall 22/STA5703_Data Mining Methodology I/Projects/Project 1/Result/Eigenvalue+Variance_PCA.csv', sep=',' ,col.names=TRUE, row.names=FALSE)

### varimax rotation 
varimax_PCA_train <- varimax_PCA(data=scaled_train_data, no_comp=13)
write.table(varimax_PCA_train, file='./UCF/Fall 22/STA5703_Data Mining Methodology I/Projects/Project 1/Result/Rotated Loadings_PCA.csv', sep=',' ,col.names=TRUE)


## test data
### test data scaling for PCA 
scaled_test_data <- scale(test_data[ , 1:81], center=TRUE)

### PCs
PCA_test <- eig_var_PCA(data=scaled_test_data, no_comp=13)
PCA_test_data <- PCA_test[[1]]


## full data
## full data scaling for PCA
scaled_data <- scale(processed_data[ , 1:81], center=TRUE)

## PCs
PCA_all_data <- eig_var_PCA(data=scaled_data, no_comp=13)
PCA_all_data <- PCA_all_data[[1]]


# final train and test data
PCA_train_data <- PCA_train_data[ , 1:13]
PCA_train_data[['critical_temp']] <- train_data[ , 82]
PCA_test_data[['critical_temp']] <- test_data[ , 82]
PCA_all_data[['critical_temp']] <- processed_data[ , 82]



#################################### Modeling #####################################

# Identifying Factors using MLR
mlr_fit <- lm(critical_temp ~ ., data=PCA_train_data)
summary(mlr_fit)

tidy_mlr_fit <- tidy(mlr_fit)
write.csv(tidy_mlr_fit, './UCF/Fall 22/STA5703_Data Mining Methodology I/Projects/Project 1/Result/mlr result.csv')

# Predictive Modeling
## linear regression
lr_fit <- lm(critical_temp ~ ., data=PCA_train_data)
summary(lr_fit)

lr_pred <- predict(lr_fit, newdata=PCA_test_data)
summary(lr_pred)

actuals_preds_lr <- data.frame(cbind(actuals=PCA_test_data$critical_temp, predicteds=lr_pred)) 

### RMSE
lr_rmse <- sqrt(mean((actuals_preds_lr$actuals - actuals_preds_lr$predicteds)^2))

### predicted vs observed plot
ggplot(actuals_preds_lr, aes(x=actuals, y=predicteds)) +
  geom_point() +
  geom_abline(intercept=0, slope=1, color='indianred3', size=1.5) +
  xlim(0, 150) +
  ylim(-25, 125) +
  labs(x='Observed Critical Temperature (K)', y='Predicted Critical Temperature (K)', title='Multiple Linear Regression Model')


## random forest regressor
rf_fit <- randomForest(critical_temp ~ ., data=PCA_train_data)
summary(rf_fit)

rf_pred <- predict(rf_fit, newdata=PCA_test_data)
summary(rf_pred)

actuals_preds_rf <- data.frame(cbind(actuals=PCA_test_data$critical_temp, predicteds=rf_pred)) 

### RMSE
rf_rmse <- sqrt(mean((actuals_preds_rf$actuals - actuals_preds_rf$predicteds)^2))

### predicted vs observed plot
ggplot(actuals_preds_rf, aes(x=actuals, y=predicteds)) +
  geom_point() +
  geom_abline(intercept=0, slope=1, color='indianred3', size=1.5) +
  xlim(0, 150) +
  ylim(-20, 130) +
  labs(x='Observed Critical Temperature (K)', y='Predicted Critical Temperature (K)', title='Random Forest Model')


## extreme gradient boosting regressor
xgb_fit <- randomForest(critical_temp ~ ., data=PCA_train_data)
summary(xgb_fit)

xgb_pred <- predict(xgb_fit, newdata=PCA_test_data)
summary(xgb_pred)

actuals_preds_xgb <- data.frame(cbind(actuals=PCA_test_data$critical_temp, predicteds=xgb_pred)) 

### RMSE
xgb_rmse <- sqrt(mean((actuals_preds_xgb$actuals - actuals_preds_xgb$predicteds)^2))

### predicted vs observed plot
ggplot(actuals_preds_xgb, aes(x=actuals, y=predicteds)) +
  geom_point() +
  geom_abline(intercept=0, slope=1, color='indianred3', size=1.5) +
  xlim(0, 150) +
  ylim(-20, 130) +
  labs(x='Observed Critical Temperature (K)', y='Predicted Critical Temperature (K)', title='Extreme Gradient Boosting Model')



rmse <- data.frame(model=c('MLR', 'RF', 'XGBoost'), rmse_scores=c(lr_rmse, rf_rmse, xgb_rmse))
ggplot(rmse, aes(x=model, y=rmse_scores)) +
  geom_bar(stat='identity', fill='indianred3', width=0.5) +
  labs(x='Type of Regressor', y='RMSE Score')












