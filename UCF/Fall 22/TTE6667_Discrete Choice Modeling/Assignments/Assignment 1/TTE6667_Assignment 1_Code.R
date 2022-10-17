# Loading Libraries
## basic
library(caret)
library(dplyr)
library(tidyr)
library(fastDummies)
library(xlsx)

# Importing Dataset
getwd()
assignment1_data <- read.csv('./UCF/Fall 22/TTE6667_Discrete Choice Modeling/Assignments/Assignment 1/vehicle.csv')

# descriptive analysis
head(assignment1_data)
dim(assignment1_data)
str(assignment1_data)
summary(assignment1_data)



################################## Data Pre-processing ##################################

# Data Cleaning
colnames(assignment1_data)

## omitting irrelevant variables
processed_data <- select(assignment1_data, -HOUSEID, -VEHID)

## independent variables
### number of non-adults (NUMNADLT) ---> new feature
processed_data$NUMNADLT <- processed_data$HHSIZE - processed_data$NUMADLT

### household income (HHFAMINC)
processed_data$HHFAMINC <- as.numeric(as.character(processed_data$HHFAMINC))
processed_data <- mutate(processed_data,
                         HHLWINC = ifelse(HHFAMINC <= 10, 'Low.Income', NA),
                         HHMEDINC = ifelse(HHFAMINC > 10 & HHFAMINC <= 17, 'Medium.Income', NA),
                         HHHGHINC = ifelse(HHFAMINC > 17, 'High.Income', NA))

temp_HHFAMINC <- select(processed_data, HHLWINC:HHHGHINC)
temp_HHFAMINC$id <- 1:length(temp_HHFAMINC$HHLWINC)
temp_HHFAMINC$HHFAMINC <- temp_HHFAMINC %>% gather(HHFAMINC, value, -id) %>% na.omit() %>%
  select(-value) %>% arrange(id) %>% select(-id)

temp_HHFAMINC$HHFAMINC <- as.character(temp_HHFAMINC$HHFAMINC$HHFAMINC)

processed_data$HHFAMINC <- as.factor(temp_HHFAMINC$HHFAMINC)
processed_data <- select(processed_data, -(HHLWINC:HHHGHINC))
rm(temp_HHFAMINC)

### vehicle hybrid or not (HYBRID)
processed_data <- processed_data %>% mutate(HYBRID = recode(HYBRID,
                                                            '1' = 'Hybrid', '2' = 'Non-Hybrid'))
processed_data$HYBRID <- as.factor(processed_data$HYBRID)

### location (URBRUR)
processed_data <- processed_data %>% mutate(URBRUR = recode(URBRUR,
                                                            '1' = 'Urban', '2' = 'Rural'))
processed_data$URBRUR <- as.factor(processed_data$URBRUR)

### vehicle type (VEHTYPE)
processed_data <- processed_data %>% mutate(VEHTYPE = recode(VEHTYPE,
                                                            '1' = 'Car', '2' = 'Other',
                                                            '3' =  'Other', '4' = 'Other', 
                                                            '5' = 'Other', '6' = 'Other',
                                                            '97' = 'Other'))
processed_data$VEHTYPE <- as.factor(processed_data$VEHTYPE)


## dummy formation
processed_data <- dummy_cols(processed_data, select_columns = c('HHFAMINC', 'URBRUR', 'HYBRID', 'VEHTYPE'), 
                             remove_selected_columns = TRUE)
columns_to_remove <- c('HHFAMINC_HHLWINC', 'URBRUR_Rural', 'HYBRID_Non-Hybrid', 'VEHTYPE_Other')
processed_data[ , columns_to_remove] <- NULL


training_data <- processed_data



##################################### Model Training #####################################

# Modeling
## linear regression model
### 3(a).
model_3a <- lm(ANNMILES ~ HHSIZE+HHFAMINC_HHHGHINC+HHFAMINC_HHMEDINC+URBRUR_Urban, 
              data = training_data)

summary(model_3a)

### 3(b).
model_3b <- lm(ANNMILES ~ HHFAMINC_HHHGHINC+HHFAMINC_HHMEDINC+URBRUR_Urban+NUMADLT+NUMNADLT, 
              data = training_data)

summary(model_3b)

### 3(c). F-test
f_test <- anova(model_3a, model_3b)
f_test

### 3(d).
model_3d <- lm(ANNMILES ~ HHVEHCNT++VEHAGE+GSYRGAL+GSTOTCST+WRKCOUNT+HHFAMINC_HHHGHINC+
                HHFAMINC_HHMEDINC+URBRUR_Urban+HYBRID_Hybrid+VEHTYPE_Car, 
              data = training_data)

summary(model_3d)

### 4(a). Hybrid
training_data_hybrid <- training_data[training_data$HYBRID_Hybrid == 1, ]

model_4a <- lm(ANNMILES ~ HHVEHCNT++VEHAGE+GSYRGAL+GSTOTCST+WRKCOUNT+HHFAMINC_HHHGHINC+
                HHFAMINC_HHMEDINC+URBRUR_Urban+HYBRID_Hybrid+VEHTYPE_Car, 
              data = training_data_hybrid)
summary(model_4a)

### 4(b). Non-Hybrid
training_data_nonHybrid <- training_data[training_data$HYBRID_Hybrid == 0, ]

model_4b <- lm(ANNMILES ~ HHVEHCNT++VEHAGE+GSYRGAL+GSTOTCST+WRKCOUNT+HHFAMINC_HHHGHINC+
                HHFAMINC_HHMEDINC+URBRUR_Urban+HYBRID_Hybrid+VEHTYPE_Car, 
              data = training_data_nonHybrid)
summary(model_4b)

### 4(c). 
model_4c <- lm(ANNMILES ~ (HHVEHCNT+VEHAGE+GSYRGAL+GSTOTCST+WRKCOUNT+HHFAMINC_HHHGHINC+
                 HHFAMINC_HHMEDINC+URBRUR_Urban+VEHTYPE_Car)*HYBRID_Hybrid, 
               data = training_data)
summary(model_4c)
