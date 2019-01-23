########################################################
#ALLOY
########################################################
setwd(dir = "desktop/R")
#install.packages("sqldf")
library(readxl)
library(sqldf)
library(ggplot2)
library(plotly)

#IMPORT DIFFERENT SHEETS
weekly_visits <- read_excel("Web Analytics Case Student Spreadsheet.xls", sheet = "Weekly Visits", skip = 4)
financials <- read_excel("Web Analytics Case Student Spreadsheet.xls", sheet = "Financials", skip = 4)
lbssold <- read_excel("Web Analytics Case Student Spreadsheet.xls", sheet = "Lbs. Sold", skip = 4)
daily_visits <- read_excel("Web Analytics Case Student Spreadsheet.xls", sheet = "Daily Visits", skip = 4)
demographics <- read_excel("Web Analytics Case Student Spreadsheet.xls", sheet = "Demographics", skip = 5)

#DESCRIPTIVE STATISTICS
#1
xform <- list(categoryorder = "array",
              categoryarray = weekly_visits$`Week (2008-2009)`)

xform2 <- list(categoryorder = "array",
              categoryarray = financials$`Week (2008-2009)`)
# Visits over time
plot_ly(data = weekly_visits, x = weekly_visits$`Week (2008-2009)`, y = weekly_visits$Visits, type ='scatter', mode = 'lines') %>%
  layout(title = "Visits Over Time", xaxis = xform)

# Unique visits over time
plot_ly(data = weekly_visits, x = weekly_visits$`Week (2008-2009)`, y = weekly_visits$`Unique Visits`, type ='bar') %>%
  layout(title = "Unique Visits Over Time", xaxis = xform)

# Revenue over time
plot_ly(data = financials, x = financials$`Week (2008-2009)`, y = financials$Revenue, type ='bar') %>%
  layout(title = "Revenues Over Time", xaxis = xform2)

# Profit over time
plot_ly(data = financials, x = financials$`Week (2008-2009)`, y = financials$Profit, type ='bar') %>%
  layout(title = "Profits Over Time", xaxis = xform2)

# Pounds sold over time
plot_ly(data = financials, x = financials$`Week (2008-2009)`, y = financials$`Lbs. Sold`, type ='bar')

########################################################
#2
########################################################

lbs_metrics_label <- c('Mean',
                       'Median',
                       'Standard Deviation',
                       'Min.',
                       'Max.'
)

#Initial - period 1


visits_period1 <- c(mean(weekly_visits$Visits[1:14]), 
                    median(weekly_visits$Visits[1:14]),
                    sd(weekly_visits$Visits[1:14]),
                    min(weekly_visits$Visits[1:14]),
                    max(weekly_visits$Visits[1:14])
)

unique_visits_period1 <- c(mean(weekly_visits$`Unique Visits`[1:14]), 
                          median(weekly_visits$`Unique Visits`[1:14]),
                          sd(weekly_visits$`Unique Visits`[1:14]),
                          min(weekly_visits$`Unique Visits`[1:14]),
                          max(weekly_visits$`Unique Visits`[1:14])
)

revenue_period1 <- c(mean(financials$Revenue[1:14]), 
                    median(financials$Revenue[1:14]),
                    sd(financials$Revenue[1:14]),
                    min(financials$Revenue[1:14]),
                    max(financials$Revenue[1:14])
)

profit_period1 <- c(mean(financials$Profit[1:14]), 
                     median(financials$Profit[1:14]),
                     sd(financials$Profit[1:14]),
                     min(financials$Profit[1:14]),
                     max(financials$Profit[1:14])
)

lbssold_period1 <- c(mean(financials$`Lbs. Sold`[1:14]), 
                     median(financials$`Lbs. Sold`[1:14]),
                     sd(financials$`Lbs. Sold`[1:14]),
                     min(financials$`Lbs. Sold`[1:14]),
                     max(financials$`Lbs. Sold`[1:14])
)


summary_period1 <- data.frame("label" = lbs_metrics_label,
                              "visits" = visits_period1, 
                              "Unique Visits" = unique_visits_period1,
                              "Revenue" = revenue_period1,
                              "Profit" = profit_period1,
                              "lbs. Sold" = lbssold_period1
                              )

#pre-promotion - period 2

visits_period2 <- c(mean(weekly_visits$Visits[15:35]), 
                    median(weekly_visits$Visits[15:35]),
                    sd(weekly_visits$Visits[15:35]),
                    min(weekly_visits$Visits[15:35]),
                    max(weekly_visits$Visits[15:35])
)

unique_visits_period2 <- c(mean(weekly_visits$`Unique Visits`[15:35]), 
                           median(weekly_visits$`Unique Visits`[15:35]),
                           sd(weekly_visits$`Unique Visits`[15:35]),
                           min(weekly_visits$`Unique Visits`[15:35]),
                           max(weekly_visits$`Unique Visits`[15:35])
)

revenue_period2 <- c(mean(financials$Revenue[15:35]), 
                     median(financials$Revenue[15:35]),
                     sd(financials$Revenue[15:35]),
                     min(financials$Revenue[15:35]),
                     max(financials$Revenue[15:35])
)

profit_period2 <- c(mean(financials$Profit[15:35]), 
                    median(financials$Profit[15:35]),
                    sd(financials$Profit[15:35]),
                    min(financials$Profit[15:35]),
                    max(financials$Profit[15:35])
)

lbssold_period2 <- c(mean(financials$`Lbs. Sold`[15:35]), 
                     median(financials$`Lbs. Sold`[15:35]),
                     sd(financials$`Lbs. Sold`[15:35]),
                     min(financials$`Lbs. Sold`[15:35]),
                     max(financials$`Lbs. Sold`[15:35])
)


summary_period2 <- data.frame("label" = lbs_metrics_label,
                              "visits" = visits_period2, 
                              "Unique Visits" = unique_visits_period2,
                              "Revenue" = revenue_period2,
                              "Profit" = profit_period2,
                              "lbs. Sold" = lbssold_period2
)

#promotion - period 3

visits_period3 <- c(mean(weekly_visits$Visits[36:52]), 
                    median(weekly_visits$Visits[36:52]),
                    sd(weekly_visits$Visits[36:52]),
                    min(weekly_visits$Visits[36:52]),
                    max(weekly_visits$Visits[36:52])
)

unique_visits_period3 <- c(mean(weekly_visits$`Unique Visits`[36:52]), 
                           median(weekly_visits$`Unique Visits`[36:52]),
                           sd(weekly_visits$`Unique Visits`[36:52]),
                           min(weekly_visits$`Unique Visits`[36:52]),
                           max(weekly_visits$`Unique Visits`[36:52])
)

revenue_period3 <- c(mean(financials$Revenue[36:52]), 
                     median(financials$Revenue[36:52]),
                     sd(financials$Revenue[36:52]),
                     min(financials$Revenue[36:52]),
                     max(financials$Revenue[36:52])
)

profit_period3 <- c(mean(financials$Profit[36:52]), 
                    median(financials$Profit[36:52]),
                    sd(financials$Profit[36:52]),
                    min(financials$Profit[36:52]),
                    max(financials$Profit[36:52])
)

lbssold_period3 <- c(mean(financials$`Lbs. Sold`[36:52]), 
                     median(financials$`Lbs. Sold`[36:52]),
                     sd(financials$`Lbs. Sold`[36:52]),
                     min(financials$`Lbs. Sold`[36:52]),
                     max(financials$`Lbs. Sold`[36:52])
)


summary_period3 <- data.frame("label" = lbs_metrics_label,
                              "visits" = visits_period3, 
                              "Unique Visits" = unique_visits_period3,
                              "Revenue" = revenue_period3,
                              "Profit" = profit_period3,
                              "lbs. Sold" = lbssold_period3
)

#post - promotion - period 4

visits_period4 <- c(mean(weekly_visits$Visits[53:66]), 
                    median(weekly_visits$Visits[53:66]),
                    sd(weekly_visits$Visits[53:66]),
                    min(weekly_visits$Visits[53:66]),
                    max(weekly_visits$Visits[53:66])
)

unique_visits_period4 <- c(mean(weekly_visits$`Unique Visits`[53:66]), 
                           median(weekly_visits$`Unique Visits`[53:66]),
                           sd(weekly_visits$`Unique Visits`[53:66]),
                           min(weekly_visits$`Unique Visits`[53:66]),
                           max(weekly_visits$`Unique Visits`[53:66])
)

revenue_period4 <- c(mean(financials$Revenue[53:66]), 
                     median(financials$Revenue[53:66]),
                     sd(financials$Revenue[53:66]),
                     min(financials$Revenue[53:66]),
                     max(financials$Revenue[53:66])
)

profit_period4 <- c(mean(financials$Profit[53:66]), 
                    median(financials$Profit[53:66]),
                    sd(financials$Profit[53:66]),
                    min(financials$Profit[53:66]),
                    max(financials$Profit[53:66])
)

lbssold_period4 <- c(mean(financials$`Lbs. Sold`[53:66]), 
                     median(financials$`Lbs. Sold`[53:66]),
                     sd(financials$`Lbs. Sold`[53:66]),
                     min(financials$`Lbs. Sold`[53:66]),
                     max(financials$`Lbs. Sold`[53:66])
)

summary_period4 <- data.frame("label" = lbs_metrics_label,
                              "visits" = visits_period4, 
                              "Unique Visits" = unique_visits_period4,
                              "Revenue" = revenue_period4,
                              "Profit" = profit_period4,
                              "lbs. Sold" = lbssold_period4
)

########################################################
#3 - means
########################################################

means <- data.frame("label" = c("initial", "pre-promotion", "promotion", "post-promotion"),
            "visits" = c(visits_period1[1], visits_period2[1], visits_period3[1], visits_period4[1]), 
           "Unique Visits" = c(unique_visits_period1[1], unique_visits_period2[1], unique_visits_period3[1], unique_visits_period4[1]),
           "Revenue" = c(revenue_period1[1], revenue_period2[1], revenue_period3[1], revenue_period4[1]),
           "Profit" = c(profit_period1[1], profit_period2[1], profit_period3[1], profit_period4[1]),
           "lbs. Sold" = c(lbssold_period1[1], lbssold_period2[1], lbssold_period3[1], lbssold_period4[1])
)


#plot means weekly visits

xform3 <- list(categoryorder = "array",
              categoryarray = c("initial", "pre-promotion", "promotion", "post-promotion"))

plot_ly(data = weekly_visits,
        x = c("initial", "pre-promotion", "promotion", "post-promotion"), 
        y = c(visits_period1[1], visits_period2[1], visits_period3[1], visits_period4[1])
)  %>%
  layout(title = "Mean Weekly Visits", xaxis = xform3)

#plot means unique visits

plot_ly(data = weekly_visits,
        x = c("initial", "pre-promotion", "promotion", "post-promotion"), 
        y = c(unique_visits_period1[1], unique_visits_period2[1], unique_visits_period3[1], unique_visits_period4[1]),
        type = 'bar'
) %>%
  layout(title = "Mean Weekly Visits", xaxis = xform3)

#plot means revenues

plot_ly(data = weekly_visits,
        x = c("initial", "pre-promotion", "promotion", "post-promotion"), 
        y = c(revenue_period1[1], revenue_period2[1], revenue_period3[1], revenue_period4[1]),
        type = 'bar'
)%>%
  layout(title = "Mean Revenues", xaxis = xform3)

#plot means profits

plot_ly(data = weekly_visits,
        x = c("initial", "pre-promotion", "promotion", "post-promotion"), 
        y = c(profit_period1[1], profit_period2[1], profit_period3[1], profit_period4[1]),
        type = 'bar'
) %>%
  layout(title = "Mean Profits" , xaxis = xform3)

#plot means lbs sold

plot_ly(data = weekly_visits,
        x = c("initial", "pre-promotion", "promotion", "post-promotion"), 
        y = c(lbssold_period1[1], lbssold_period2[1], lbssold_period3[1], lbssold_period4[1]),
        type = 'bar'
) %>%
  layout(title = "Mean Lbs. Sold", xaxis = xform3)

########################################################
#5 - Relationship revenue <-> pound sold
########################################################

plot(x = financials$Revenue, y = financials$`Lbs. Sold`, type = 'p', xlab = 'Lbs. Sold', ylab = 'Revenues', main = 'More quantity sold means more revenues')
cor(financials$Revenue, financials$`Lbs. Sold`)
data.frame('correlation' = c("Lbs. Sold"), 'revenues' = c(cor(financials$Revenue, financials$`Lbs. Sold`)))

########################################################
#6 - Relationship revenue <-> visits
########################################################

plot(x = weekly_visits$Visits, y = financials$Revenue, type = 'p', xlab = 'Weekly Visits', ylab = 'Revenue', main = 'Weekly Visits - Revenues')
cor(financials$Revenue, weekly_visits$Visits)

########################################################
#8 - normalities
########################################################

#a

summary_lbssold <- data.frame("label" = lbs_metrics_label,
                              "lbssold" = c(mean(lbssold$`Lbs. Sold`), 
                                            median(lbssold$`Lbs. Sold`),
                                            sd(lbssold$`Lbs. Sold`),
                                            min(lbssold$`Lbs. Sold`),
                                            max(lbssold$`Lbs. Sold`)
                                            )
                              )

summary_lbssold

#B
as.Date(lbssold$Week, format = '%d-%m-%y')
plot(x = lbssold$Week, y = lbssold$`Lbs. Sold`, type = 'h')

#D
lbssold$'z-score' <- (lbssold$`Lbs. Sold` - mean(lbssold$`Lbs. Sold`))/sd(lbssold$`Lbs. Sold`)
lbssold$'score' <- NULL

for (i in 1:length(lbssold$`z-score`)){
  if (lbssold$`z-score`[i] < 1 & lbssold$`z-score`[i]>-1){
    lbssold$'score'[i] <- 1
  }
  else if (lbssold$`z-score`[i] < -1 & lbssold$`z-score`[i]>-2 | lbssold$`z-score`[i] > 1 & lbssold$`z-score`[i]< 2){
    lbssold$'score'[i] <- 2
  }
  else if (lbssold$`z-score`[i] < -2 & lbssold$`z-score`[i]>-3 | lbssold$`z-score`[i] > 2 & lbssold$`z-score`[i]< 3){
    lbssold$'score'[i] <- 3
  }
  else {
    lbssold$'score'[i] <- 0
  }
}

#Calculating theoretical number of observations into vector
theoretical_no_obs1 <- length(lbssold$`Lbs. Sold`)*.68
theoretical_no_obs2 <- length(lbssold$`Lbs. Sold`)*.95
theoretical_no_obs3 <- length(lbssold$`Lbs. Sold`)*.99
theoretical_no_vec <- c(theoretical_no_obs1,theoretical_no_obs2,theoretical_no_obs3)

#Calculating actual number of observations into vector
actual_no_obs1 <- length(lbssold$score[which(lbssold$score == 1)])
actual_no_obs2 <- length(lbssold$score[which(lbssold$score == 1)]) + length(lbssold$score[which(lbssold$score == 2)])
actual_no_obs3 <- length(lbssold$score[which(lbssold$score == 1)]) + length(lbssold$score[which(lbssold$score == 2)]) + length(lbssold$score[which(lbssold$score == 3)])
actual_no_vec <- c(actual_no_obs1, actual_no_obs2, actual_no_obs3)

#Creating interval labels vector
int1 <- 'mean ± 1 std. dev.'
int2 <- 'mean ± 2 std. dev.'
int3 <- 'mean ± 3 std. dev.'
int_vec <- c(int1,int2,int3)

#Creating interval labels vector
perc1 <- .68
perc2 <- .95
perc3 <- .99
perc_vec <- c(perc1,perc2,perc3)

#Arranging theoretical and actual number of observations into dataframe
normality_df <- data.frame("Interval" = int_vec,
                           "Theoretical % of Data" = perc_vec,
                           "Theoretical No. Obs." = theoretical_no_vec,
                           "Actual No. Obs." = actual_no_vec)

#E

#Initialize new column
lbssold$'score2' <- NULL

#Fill it up with values
for (i in 1:length(lbssold$`z-score`)){
  if (lbssold$`z-score`[i] < 0 & lbssold$`z-score`[i]>-1){
    lbssold$'score2'[i] <- -1
  }
  else if (lbssold$`z-score`[i] < -1 & lbssold$`z-score`[i]>-2){
    lbssold$'score2'[i] <- -2
  }
  else if (lbssold$`z-score`[i] < -2 & lbssold$`z-score`[i]>-3){
    lbssold$'score2'[i] <- -3
  }
  else if (lbssold$`z-score`[i] > 0 & lbssold$`z-score`[i]<1){
    lbssold$'score2'[i] <- 1
  }
  else if (lbssold$`z-score`[i] > 1 & lbssold$`z-score`[i]<2){
    lbssold$'score2'[i] <- 2
  }
  else if (lbssold$`z-score`[i] > 2 & lbssold$`z-score`[i]<3){
    lbssold$'score2'[i] <- 3
  }
  else {
    lbssold$'score2'[i] <- 0
  }
}

#Creating interval labels vector
int_1a <- 'mean + 1 std. dev.'
int_1b <- 'mean - 1 std. dev.'
int_2a <- '1 std. dev. to 2 std. dev.'
int_2b <- '-1 std. dev. to -2 std. dev.'
int_3a <- '2 std. dev. to 3 std. dev.'
int_3b <- '-2 std. dev. to -3 std. dev'
int_vec2 <- c(int_1a, int_1b, int_2a, int_2b, int_3a, int_3b)

#Creating interval labels vector
perc_1a <- .34
perc_1b <- .34
perc_2a <- .135
perc_2b <- .135
perc_3a <- .02
perc_3b <- .02
perc_vec2 <- c(perc_1a, perc_1b, perc_2a, perc_2b,
               perc_3a, perc_3b)

#Calculating theoretical number of observations into vector
theoretical_no_obs_1a <- length(lbssold$`Lbs. Sold`)*perc_1a
theoretical_no_obs_1b <- length(lbssold$`Lbs. Sold`)*perc_1b
theoretical_no_obs_2a <- length(lbssold$`Lbs. Sold`)*perc_2a
theoretical_no_obs_2b <- length(lbssold$`Lbs. Sold`)*perc_2b
theoretical_no_obs_3a <- length(lbssold$`Lbs. Sold`)*perc_3a
theoretical_no_obs_3b <- length(lbssold$`Lbs. Sold`)*perc_3b
theoretical_no_vec2 <-c(theoretical_no_obs_1a,theoretical_no_obs_1b,
                        theoretical_no_obs_2a, theoretical_no_obs_2b,
                        theoretical_no_obs_3a,theoretical_no_obs_3b)

#Calculating actual number of observations into vector
actual_no_obs_1a <- length(lbssold$score2[which(lbssold$'score2' == 1)])
actual_no_obs_1b <- length(lbssold$score2[which(lbssold$'score2' == 2)])
actual_no_obs_2a <- length(lbssold$score2[which(lbssold$'score2' == 3)])
actual_no_obs_2b <- length(lbssold$score2[which(lbssold$'score2' == -1)])
actual_no_obs_3a <- length(lbssold$score2[which(lbssold$'score2' == -2)])
actual_no_obs_3b <- length(lbssold$score2[which(lbssold$'score2' == -3)])
actual_no_vec2 <- c(actual_no_obs_1a, actual_no_obs_1b,
                    actual_no_obs_2a, actual_no_obs_2b,
                    actual_no_obs_3a, actual_no_obs_3b)

#Arranging theoretical and actual number of observations into dataframe
normality_df2<-data.frame("Interval"=int_vec2,
                          "Theoretical % of Data" = perc_vec2,
                          "Theoretical No. Obs." = theoretical_no_vec2,
                          "Actual No. Obs." = actual_no_vec2)

normality_df2

#distribution graph
xform4 <- list(categoryorder = "array",
               categoryarray = c("-3", "-2", "-1", "1", "2", "3"))

plot_ly(data = weekly_visits,
        x = c("-3", "-2", "-1", "1", "2", "3"), 
        y = c(actual_no_obs_3b, actual_no_obs_3a,
              actual_no_obs_2b, actual_no_obs_1a,
              actual_no_obs_1b, actual_no_obs_2a),
        type = 'bar'
)%>%
  layout(title = "Distribution Lbs. Sold", xaxis = xform4)

#G
#install.packages('moments')
library(moments)
skewness_ps <- skewness(lbssold$`Lbs. Sold`)
kurtosis_ps <- (lbssold$`Lbs. Sold`)

########################################################
#10 - demograpics visual representation
########################################################


visibility <- data.frame("Traffic sources" = c("Referring sites", "Search Engines", "Direct Traffic", "Other"),
                         "Visits" = c(38754, 20964, 9709, 4))

plot_ly(data = visibility) %>%
  add_pie(labels = visibility[,1],
          values = visibility[,2])


visibility2 <- data.frame("Traffic sources" = c("googleads.g.doubleclick.net", "pagead2.googlesyndication.com", "sedoparking.com",
                                                "globalspec.com", "searchportal.information.com", "freepatentsonline.com", "thomasnet.com",
                                                "mu.com", "mail.google.com","psicofxp.com"),
                         "Visits" = c(15626, 8.044, 3138, 693, 582, 389, 379, 344, 337, 310))

plot_ly(data = visibility,
        x = visibility2[,1], 
        y = visibility2[,2],
        type = 'bar'
) %>%
  layout(title = "Top Referring Websites")
