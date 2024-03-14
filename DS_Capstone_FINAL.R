#iFOOD MARKETING ANALYSIS 
#SPRING 2024 DATA SCIENCE CAPSTONE
# --- Package Management ---
install.packages(c("tidyverse", "caret", "mice", "car", "lmtest", "pROC", "e1071", "rpart","rpart.plot"))
install.packages("randomForest")
library(tidyverse)
library(caret)
library(mice)
library(car)
library(lmtest)
library(pROC)
library(e1071)
library(rpart)
library(rpart.plot)
library(randomForest)

# --- Data Loading and Initial Exploration ---
marketing_data <- read.csv(file ='/Users/crystalluo/Desktop/DS Capstone/ml_project1_data.csv')
glimpse(marketing_data)
summary(marketing_data)
table(marketing_data$Response)

# --- Data Cleaning ---
# Recategorize 'Marital_Status'
marketing_data$Marital_Status[marketing_data$Marital_Status %in% c("Alone", "Absurd", "YOLO")] <- "Single"
marketing_data$Marital_Status <- as.factor(marketing_data$Marital_Status)

# Merge 'Education' categories and rename
marketing_data$Education[marketing_data$Education == "2n Cycle"] <- "Master"
marketing_data$Education[marketing_data$Education == "Graduation"] <- "Undergraduate"
marketing_data$Education <- as.factor(marketing_data$Education)

# Remove 'Z_CostContact' and 'Z_Revenue'
marketing_data <- select(marketing_data, -Z_CostContact, -Z_Revenue)

# Convert 'Dt_Customer' to Date type
marketing_data$Dt_Customer <- as.Date(marketing_data$Dt_Customer, format = "%Y-%m-%d")

#making the response variable as categorical
marketing_data$Response <- as.factor(marketing_data$Response)

# --- Handling Missing Values ---
# Impute missing values with MICE
set.seed(500) # For reproducibility
imputed_data <- mice(marketing_data, m = 5, maxit = 50, method = 'pmm', seed = 500) %>% complete()

# --- Feature Engineering ---
# Standardize numeric features
numeric_columns <- sapply(imputed_data, is.numeric)
imputed_data[numeric_columns] <- scale(imputed_data[numeric_columns])

# --- Exploratory Data Analysis (EDA) ---
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

# EDA for categorical features against the response
nominal_features <- imputed_data %>%
  dplyr::select(where(is.character), where(is.factor), -Response) %>%
  names()

imputed_data %>% 
  dplyr::select(all_of(nominal_features), Response) %>% 
  pivot_longer(cols = nominal_features, names_to = "type", values_to = "value") %>% 
  group_by(type, value, Response) %>% 
  summarize(n = n(), .groups = "drop") %>% 
  mutate(frac = n / sum(n)) %>% 
  ggplot(aes(frac, value, fill = Response)) +
  geom_col() +
  scale_x_continuous(labels = scales::percent) +
  facet_wrap(~ type, scales = "free_y", ncol = 3) +
  theme(legend.position = "top") +
  labs(title = "Categorical Features", x = "Fraction of Observations", y = NULL)

imputed_data %>% 
  dplyr::select(all_of(nominal_features), Response) %>% 
  pivot_longer(cols = nominal_features, names_to = "type", values_to = "value") %>% 
  group_by(type, value, Response) %>% 
  summarize(n = n(), .groups = "drop") %>% 
  mutate(frac = n / sum(n)) %>% 
  ggplot(aes(frac, value, fill = Response)) +
  geom_col() +
  scale_x_continuous(labels = scales::percent) +
  scale_fill_manual(values = c("skyblue", "pink")) + # Light blue and pink colors
  facet_wrap(~ type, scales = "free_y", ncol = 3) +
  theme(legend.position = "top") +
  labs(title = "Impact on Response - Categorical Features", x = "Fraction of Observations", y = NULL)

# --- Model Training ---
# Split data
set.seed(123)
index <- createDataPartition(imputed_data$Response, p = 0.8, list = FALSE)
train_data <- imputed_data[index, ]
test_data <- imputed_data[-index, ]

# --- Model Evaluation ---
# Logistic Regression Model
model_logit <- glm(Response ~ ., data = train_data, family = binomial(link="logit"))
vif_results <- vif(model_logit)
summary(vif_results)

# Perform stepwise model selection based on AIC
AIC_model1 <- step(model_logit, direction = "both", trace = FALSE, k = 2)

# Perform stepwise model selection based on BIC
BIC_model1 <- step(model_logit, direction = "both", trace = FALSE, k = log(nrow(train_data)))

# Decision Tree Model
model_dt <- rpart(Response ~ ., data = train_data, method = "class")
summary(model_dt)
rpart.plot(model_dt, type = 4, extra = 102, main = "Decision Tree Diagram")

# SVM Model
model_svm <- svm(Response ~ ., data = train_data, probability = TRUE)

#Random Forest
model_rf <- randomForest(Response ~ ., data = train_data, method = "class")
# Get feature importance
feature_importance <- importance(model_rf)
# View the feature importance scores
print(feature_importance)
# For a more visual representation, you can plot the feature importances
importance_data <- as.data.frame(importance(model_rf))
ggplot(importance_data, aes(x = reorder(row.names(importance_data), MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +  # Flips the axes
  theme_minimal() +
  labs(x = "Features", y = "Importance",
       title = "Random Forest Feature Importance",
       subtitle = "Using Mean Decrease in Gini") +
  theme(plot.title = element_text(hjust = 0.5)) 

# --- Model Predictions and ROC Curve Analysis ---
# Logistic Regression Predictions
predictions_logit <- predict(model_logit, newdata = test_data, type = "response")

#AIC and BIC Predictions
predictions_AIC <- predict(AIC_model1, newdata = test_data, type = "response")
predictions_BIC <- predict(BIC_model1, newdata = test_data, type = "response")

# Decision Tree Predictions
predictions_dt <- predict(model_dt, newdata = test_data, type = "prob")[,2]

# SVM Predictions
predictions_svm <- attr(predict(model_svm, newdata = test_data, probability = TRUE), "probabilities")[,2]

#Random Forest Predictions
predictions_rf <- predict(model_rf, newdata = test_data, type = "prob")
positive_class_probs <- predictions_rf[, 2]


# Compute ROC Curves and AUC Values
roc_logit <- roc(response = test_data$Response, predictor = predictions_logit)
roc_AIC <- roc(response = test_data$Response, predictor = predictions_AIC)
roc_BIC <- roc(response = test_data$Response, predictor = predictions_BIC)
roc_dt <- roc(response = test_data$Response, predictor = predictions_dt)
roc_svm <- roc(response = test_data$Response, predictor = predictions_svm)
roc_rf <- roc(response = test_data$Response, predictor = positive_class_probs)

# Plot ROC Curves for All Models
plot(roc_logit, col = "red", main = "ROC Curves for Different Models")
plot(roc_AIC, col = "blue", add = TRUE)
plot(roc_BIC, col = "green", add = TRUE)
plot(roc_dt, col = "purple", add = TRUE)
plot(roc_svm, col = "brown", add = TRUE)
plot(roc_rf,col = "orange", add = TRUE)
legend("bottomright", 
       legend = c("Logit Model", "AIC Model", "BIC Model","Decision Tree", "SVM","Random Forest"), 
       col = c("red", "blue", "green", "purple", "brown","orange"), 
       lty = 1)

# Print AUC Values
print(paste("AUC for Logit Model:", auc(roc_logit)))
print(paste("AUC for AIC Model:", auc(roc_AIC)))
print(paste("AUC for BIC Model:", auc(roc_BIC)))
print(paste("AUC for Decision Tree:", auc(roc_dt)))
print(paste("AUC for SVM:", auc(roc_svm)))
print(paste("AUC for Random Forest:", auc(roc_rf)))


# Assuming predictions_logit, predictions_AIC, predictions_BIC, predictions_dt, and predictions_svm are probabilities of the positive class
threshold <- 0.5
predicted_classes_logit <- factor(ifelse(predictions_logit > 0.5, "1", "0"), levels = c("0", "1"))
predicted_classes_AIC <- factor(ifelse(predictions_AIC > 0.5, "1", "0"), levels = c("0", "1"))
predicted_classes_BIC <- factor(ifelse(predictions_BIC > 0.5, "1", "0"), levels = c("0", "1"))
predicted_classes_dt <- factor(ifelse(predictions_dt > 0.5, "1", "0"), levels = c("0", "1"))
predicted_classes_svm <- factor(ifelse(predictions_svm > 0.5, "1", "0"), levels = c("0", "1"))
predicted_classes_rf <- factor(ifelse(positive_class_probs > 0.5, "1", "0"))

library(caret)

# Ensure  test_data$Response is factor variable with 0 and 1
test_data$Response <- as.factor(test_data$Response)
typeof(test_data$Response)
#Logit function
actual <- factor(test_data$Response, levels = c("0", "1"))

#Calculating Accuracy
accuracy_logit <- mean(predicted_classes_logit == actual)
print(paste("Accuracy:", accuracy_logit))
cm_logit <- confusionMatrix(predicted_classes_logit, actual)

#Precision, Recall, and F1 Score
precision_logit <- cm_logit$byClass['Pos Pred Value']
recall_logit <- cm_logit$byClass['Sensitivity']
f1_logit <- 2 * ((precision_logit * recall_logit) / (precision_logit + recall_logit))

print(paste("Accuracy for Logistic Regression:", accuracy_logit))
print(paste("Precision for Logistic Regression:", precision_logit))
print(paste("Recall for Logistic Regression:", recall_logit))
print(paste("F1 Score for Logistic Regression:", f1_logit))

# For AIC model (if it's logistic regression based, with probabilities)
accuracy_AIC <- mean(predicted_classes_AIC == actual)
cm_AIC <- confusionMatrix(predicted_classes_AIC, actual)

precision_AIC <- cm_AIC$byClass['Pos Pred Value']
recall_AIC <- cm_AIC$byClass['Sensitivity']
f1_AIC <- 2 * ((precision_AIC * recall_AIC) / (precision_AIC + recall_AIC))

print(paste("Accuracy for AIC Model:", accuracy_AIC))
print(paste("Precision for AIC Model:", precision_AIC))
print(paste("Recall for AIC Model:", recall_AIC))
print(paste("F1 Score for AIC Model:", f1_AIC))

# For BIC model (if it's logistic regression based, with probabilities)
accuracy_BIC <- mean(predicted_classes_BIC == actual)
cm_BIC <- confusionMatrix(predicted_classes_BIC, actual)

precision_BIC <- cm_BIC$byClass['Pos Pred Value']
recall_BIC <- cm_BIC$byClass['Sensitivity']
f1_BIC <- 2 * ((precision_BIC * recall_BIC) / (precision_BIC + recall_BIC))

print(paste("Accuracy for BIC Model:", accuracy_BIC))
print(paste("Precision for BIC Model:", precision_BIC))
print(paste("Recall for BIC Model:", recall_BIC))
print(paste("F1 Score for BIC Model:", f1_BIC))

#For Decision Trees
accuracy_dt <- mean(predicted_classes_dt == actual)
cm_dt <- confusionMatrix(predicted_classes_dt, actual)

precision_dt <- cm_dt$byClass['Pos Pred Value']
recall_dt <- cm_dt$byClass['Sensitivity']
f1_dt <- 2 * ((precision_dt * recall_dt) / (precision_dt + recall_dt))

print(paste("Accuracy for Decision Tree:", accuracy_dt))
print(paste("Precision for Decision Tree:", precision_dt))
print(paste("Recall for Decision Tree:", recall_dt))
print(paste("F1 Score for Decision Tree:", f1_dt))

#For Support Vector Machines
accuracy_svm <- mean(predicted_classes_svm == actual)
cm_svm <- confusionMatrix(predicted_classes_svm, actual)

precision_svm <- cm_svm$byClass['Pos Pred Value']
recall_svm <- cm_svm$byClass['Sensitivity']
f1_svm <- 2 * ((precision_svm * recall_svm) / (precision_svm + recall_svm))

print(paste("Accuracy for SVM:", accuracy_svm))
print(paste("Precision for SVM:", precision_svm))
print(paste("Recall for SVM:", recall_svm))
print(paste("F1 Score for SVM:", f1_svm))

#for Random Forest
#For Decision Trees
accuracy_rf <- mean(predicted_classes_rf == actual)
cm_rf <- confusionMatrix(predicted_classes_rf, actual)

precision_rf <- cm_rf$byClass['Pos Pred Value']
recall_rf <- cm_rf$byClass['Sensitivity']
f1_rf <- 2 * ((precision_rf * recall_rf) / (precision_rf + recall_rf))

print(paste("Accuracy for Random Forest:", accuracy_rf))
print(paste("Precision for Random Forest:", precision_rf))
print(paste("Recall for Random Forest:", recall_rf))
print(paste("F1 Score for Random Forest:", f1_rf))






