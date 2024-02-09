# iFood-Analysis

# load packages
install.packages("tidyverse")
library(tidyverse)
library(tidyverse)
library(readr)
library(reshape2)
library(ggplot2)
library(mice)

marketing_data <- read.csv(file ='/Users/crystalluo/Desktop/DS Capstone/ml_project1_data.csv')
summary(marketing_data)
dim(marketing_data)
head(marketing_data)
str(marketing_data)
names(marketing_data)

table(marketing_data$Education)
table(marketing_data$Marital_Status)

#data cleaning
# cleaning Marital Status:
# Recategorize "Alone", "Absurd" and "YOLO" as "Single" (or remove them, depending on your preference)
marketing_data$Marital_Status[marketing_data$Marital_Status %in% c("Alone", "Absurd", "YOLO")] <- "Single"
marketing_data$Marital_Status <- as.factor(marketing_data$Marital_Status)
table(marketing_data$Marital_Status)

# Merge "2n Cycle" and "Master" into "Master"
marketing_data$Education[marketing_data$Education == "2n Cycle"] <- "Master"
# Rename "Graduation" to "Undergraduate"
marketing_data$Education[marketing_data$Education == "Graduation"] <- "Undergraduate"
marketing_data$Education <- as.factor(marketing_data$Education)
table(marketing_data$Education)

# Convert Dt_Customer from character to Date type
marketing_data$Dt_Customer <- as.Date(marketing_data$Dt_Customer, format = "%Y-%m-%d")
# Calculating the number of days of enrollment 
days_since_today <- as.numeric(Sys.Date() - marketing_data$Dt_Customer)

#Making the response variable a binary outcome variable
marketing_data$Response <- as.factor(marketing_data$Response)

# Check for NA values across all variables in the dataframe
na_counts <- colSums(is.na(marketing_data))
# Print the counts of NA values for each column
print(na_counts)

# Setting up MICE parameters
mice_params <- mice::mice(marketing_data, m = 5, maxit = 50, method = 'pmm', seed = 500)
# Performing the imputation
imputed_data <- mice::complete(mice_params, 1)
na_counts_updated <- colSums(is.na(imputed_data))

# Basic data summaries
summary(imputed_data)  
dim(imputed_data)
names(imputed_data)

# EDA for numeric features against the response
numeric_features <- imputed_data %>%
  dplyr::select(where(is.numeric), Response) %>%
  pivot_longer(-Response, names_to = "metric", values_to = "value") 

ggplot(numeric_features, aes(value, color = as.factor(Response))) +
  geom_density(alpha = 0.6) +
  facet_wrap(vars(metric), scales = "free", ncol = 3) +
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        )

# EDA for nominal features against the response
nominal_features <- imputed_data %>%
  dplyr::select(where(is.character), where(is.factor), -Response) %>%
  names()

imputed_data %>% 
  dplyr::select(all_of(nominal_features), Response) %>% 
  pivot_longer(cols = nominal_features, names_to = "type", values_to = "value") %>% 
  group_by(type, value, Response) %>% 
  summarize(n = n(), .groups = "drop") %>% 
  mutate(frac = n / sum(n)) %>% 
  ggplot(aes(frac, value, fill = as.factor(Response))) +
  geom_col() +
  scale_x_continuous(labels = scales::percent) +
  facet_wrap(~ type, scales = "free_y", ncol = 3) +
  theme(legend.position = "top") +
  labs(title = "Impact on Response - Categorical Features", x = "Fraction of Observations", y = NULL)

# For continuous variables
ggplot(imputed_data, aes(x = Response, y = Income)) +
  geom_boxplot() +
  labs(title = "Boxplot of ContinuousVariable by Response")

# For categorical variables
ggplot(imputed_data, aes(x = CategoricalVariable, fill = as.factor(Response))) +
  geom_bar(position = "fill") +
  labs(title = "Bar Plot of CategoricalVariable by Response")



