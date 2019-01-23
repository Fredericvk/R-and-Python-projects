################################
#Census Full Analysis
################################

# 1.Importing the data
library(readr)
library(readxl)
census <- read_csv("census_income.csv")

# 2. Exploring the data
View(census)
summary(census)
str(census)

# 3. Massaging the data
#Giving a column name to X15
colnames(census)[15] <- "income"

#making the income column binary
census$income <- gsub("<=50K", 0, census$income)
census$income <- gsub(">50K", 1, census$income)
census$income <- as.numeric(census$income)

#making sex binary and numeric
census$sex <- gsub("Male", 1, census$sex)
census$sex <- gsub("Female", 0, census$sex)
census$sex <- as.numeric(census$sex)

#Checking the data
summary(census)

#Checking normality
census <- as.data.frame(census)
for (i in 1:ncol(census)){
  try(hist(census[,i]))
}

# 4. Build a tree
library(rpart)
library(rpart.plot)
census_tree <- rpart(income~ age + education_num + sex + hours_per_week, data = census, method = "class", cp = 0.012)
rpart.plot(census_tree, type = 1, extra = 1, box.palette = c("pink","green"))
plotcp(census_tree)
summary(census_tree)

#5. Build a logistic regression
census_log <- glm(income~ age + education_num + sex + hours_per_week, data = census, family = "binomial")
summary(census_log)

#6. Prediction
#LOGISTIC: predicting probability of 1
predict_logit <- predict(census_log, census, type="response")
print(predict_logit)

#predicting probability of 1 TREE
predict_tree <- predict(census_tree, census, type="prob")
print(predict_tree)

#Storing Model Performance Scores
library(ROCR)
pred_val_tree <- prediction(predict_tree[,2], census$income)
pred_val_logit <- prediction(predict_logit, census$income) 

#we need performance
perf_tree <- performance(pred_val_tree,"tpr","fpr")
perf_logit <- performance(pred_val_logit,"tpr","fpr")

#Plotting Lift Curve
plot(perf_tree,col="black",lty=3, lwd=3)
plot(perf_logit,col="blue",lty=3, lwd=3, add=TRUE)
plot(performance(pred_val_tree, measure="lift", x.measure="rpp"), colorize=TRUE)