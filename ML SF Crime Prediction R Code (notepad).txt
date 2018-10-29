#DATA IS TO BE DOWNLOADED FROM:
#https://www.kaggle.com/c/sf-crime/data

rm(list=ls())
library(lightgbm)
library(caret)
library(lubridate)
library(stringr)
library(Matrix)
library(dummies)
#===================================================================================
#LOAD DATA
traindata <- read.csv("C:/Users/Chris/Desktop/Work/Kaggle/Kaggle SF Crime/train.csv", header = TRUE)
testdata <- read.csv("C:/Users/Chris/Desktop/Work/Kaggle/Kaggle SF Crime/test.csv", header = TRUE)

#===================================================================================
#COMBINE TRAIN AND TEST AND REMOVE UNUSED COLUMNS THAT ARE ONLY UNIQUE TO ONE DATASET
train_and_test_data <- rbind(traindata[,-c(2,3,6,7)],testdata[,-c(1,5)])

#===================================================================================
#RENAMING COLUMNS AND FORMATING DATE/TIMES
colnames(train_and_test_data) <- str_replace(colnames(train_and_test_data), "X","Longitude")
colnames(train_and_test_data) <- str_replace(colnames(train_and_test_data), "Y","Latitude")
DateTimes <- as.POSIXct(train_and_test_data$Dates, format="%Y-%m-%d %H:%M:%S", tz = "GMT")
Dates <- as.Date(train_and_test_data$Dates, format = "%Y-%m-%d")
Minutes <- minute(DateTimes)
Hours <- hour(DateTimes)
Days <- day(DateTimes)
Years <- year(DateTimes)
Months <- as.numeric(factor(month(DateTimes)))

#===================================================================================
#TAKING THE AVERAGE LATITUDES BY DISTRICT
Y_mean_BAYVIEW <- mean(train_and_test_data[which(train_and_test_data$PdDistrict=="BAYVIEW"),]$Latitude)
Y_mean_CENTRAL <- mean(train_and_test_data[which(train_and_test_data$PdDistrict=="CENTRAL"),]$Latitude)
Y_mean_INGLESIDE <- mean(train_and_test_data[which(train_and_test_data$PdDistrict=="INGLESIDE"),]$Latitude)
Y_mean_MISSION <- mean(train_and_test_data[which(train_and_test_data$PdDistrict=="MISSION"),]$Latitude)
Y_mean_NORTHERN <- mean(train_and_test_data[which(train_and_test_data$PdDistrict=="NORTHERN"),]$Latitude)
Y_mean_PARK <- mean(train_and_test_data[which(train_and_test_data$PdDistrict=="PARK"),]$Latitude)
Y_mean_RICHMOND <- mean(train_and_test_data[which(train_and_test_data$PdDistrict=="RICHMOND"),]$Latitude)
Y_mean_SOUTHERN <- mean(train_and_test_data[which(train_and_test_data$PdDistrict=="SOUTHERN"),]$Latitude)
Y_mean_TARAVAL <- mean(train_and_test_data[which(train_and_test_data$PdDistrict=="TARAVAL"),]$Latitude)
Y_mean_TENDERLOIN <- mean(train_and_test_data[which(train_and_test_data$PdDistrict=="TENDERLOIN"),]$Latitude)

#===================================================================================
# TESTING FOR OUTLIERS in LONGITUDE, LATITUDE, AND "NA" VALUES
# San Fransico is approximately X = -122.431297 AND Y = 37.773972
row_NA <- which(is.na(traindata))
mean_X <- mean(traindata$X)
mean_Y <- mean(traindata$Y)

unique_X <- unique(round(traindata$X)) # -120, -122, -123
unique_Y <- unique(round(traindata$Y)) # 38, 90

outlier_Y <- which(round(traindata$Y) == 90) # Selects each row with the obvious outlier of Y ~ 90

#===================================================================================
# TAKING THE AVERAGE OF ALL Y VALUES WITH THE CORRESPONDING DISTRICT

Y_mean_BAYVIEW <- mean(traindata[which(traindata$PdDistrict=="BAYVIEW"),]$Y)
Y_mean_CENTRAL <- mean(traindata[which(traindata$PdDistrict=="CENTRAL"),]$Y)
Y_mean_INGLESIDE <- mean(traindata[which(traindata$PdDistrict=="INGLESIDE"),]$Y)
Y_mean_MISSION <- mean(traindata[which(traindata$PdDistrict=="MISSION"),]$Y)
Y_mean_NORTHERN <- mean(traindata[which(traindata$PdDistrict=="NORTHERN"),]$Y)
Y_mean_PARK <- mean(traindata[which(traindata$PdDistrict=="PARK"),]$Y)
Y_mean_RICHMOND <- mean(traindata[which(traindata$PdDistrict=="RICHMOND"),]$Y)
Y_mean_SOUTHERN <- mean(traindata[which(traindata$PdDistrict=="SOUTHERN"),]$Y)
Y_mean_TARAVAL <- mean(traindata[which(traindata$PdDistrict=="TARAVAL"),]$Y)
Y_mean_TENDERLOIN <- mean(traindata[which(traindata$PdDistrict=="TENDERLOIN"),]$Y)

#===================================================================================
# REPLACING ALL OUTLIER Y VALUES WITH THE AVERAGE Y VALUE OF THE CORRESPONDING DISTRICT

for (i in 1:length(outlier_Y)) {
  pd <- traindata[outlier_Y[i],5]
  pd <- as.integer(pd)
  traindata[outlier_Y[i],9] <- switch(pd, 
                                      "1"=Y_mean_BAYVIEW,
                                      "2"=Y_mean_CENTRAL,
                                      "3"=Y_mean_INGLESIDE,
                                      "4"=Y_mean_MISSION,
                                      "5"=Y_mean_NORTHERN,
                                      "6"=Y_mean_PARK,
                                      "7"=Y_mean_RICHMOND,
                                      "8"=Y_mean_SOUTHERN,
                                      "9"=Y_mean_TARAVAL,
                                      "10"=Y_mean_TENDERLOIN)
}
#===================================================================================
#EXTRACTING DAY OF WEEK AND ASSIGNING UNIQUE INTEGER ID TO EACH DISTRICT
train_and_test_data["DayOfWeek"] <- wday(ymd(Dates))
train_and_test_data["PdDistrict"] <- as.integer(factor(train_and_test_data$PdDistrict))

#===================================================================================
#K MEANS CLUSTER CATEGORIES OF CRIME BY LOCATION OF LONGITUDE AND LATITUDE TO 39 UNIQUE CLUSTERS AS A NEW FEATURE
clust <- train_and_test_data %>% select(Longitude, Latitude)
k <- kmeans(clust, length(unique(traindata$Category)), iter.max = 300)
train_and_test_data$Cluster <- as.factor(k$cluster)

#===================================================================================
#CONVERTING LONGITUDE AND LATITUDES TO X,Y,Z COORDINATES (PLAYING WITH DATA HOPING TO MINIMIZE OVERFITTING)
#ALSO ASSUME EARTH IS PERFECT SPHERE SO RADIUS IS ASSUMED TO BE 1
radius <- 1
latitude <- train_and_test_data$Latitude
longitude <- train_and_test_data$Longitude
train_and_test_data["x"] <- radius*cos(latitude)*cos(longitude)
train_and_test_data["y"] <- radius*cos(latitude)*sin(longitude)
train_and_test_data["z"] <- radius*sin(latitude)

#===================================================================================
#ASSIGNING IF CRIME OCCURS DURING A PEAK TIME. (BETWEEN 1AM - 8AM LOW CRIME)
train_and_test_data$Peaktimes <-  ifelse(Hours >= 1 & Hours <= 8, 0, 1)

#===================================================================================
#ASSIGN THE PROPER SEASON TO THE CRIME DEPENDING ON THE MONTH
Seasons <- matrix(0,nrow = nrow(train_and_test_data),ncol = 1)

for (i in 1:length(Months)) {
  pd <- Months[i]
  Seasons[i,1] <- switch(pd, 
                         "1"="Winter",
                         "2"="Winter",
                         "3"="Spring",
                         "4"="Spring",
                         "5"="Spring",
                         "6"="Summer",
                         "7"="Summer",
                         "8"="Summer",
                         "9"="Fall",
                         "10"="Fall",
                         "11"="Fall",
                         "12"="Winter")
}
Seasons <- data.frame(Seasons)
Dummy_Seasons <- dummy.data.frame(Seasons)
Seasons <- Dummy_Seasons
colnames(Seasons) <- c("Fall", "Spring", "Summer", "Winter" )

#===================================================================================
#COMBINING ALL DATA THAT WILL BE USED AS FEATURE MATRIX TO TRAIN
features <- cbind(Minutes, Hours, Days, Months, Years,Seasons, train_and_test_data[,-c(1)])

#===================================================================================
#GENERIC LAYOUT OF SEPERATING PREDICTERS AND TARGET VALUES
train_matrix <- features[1:nrow(traindata),]
train_matrix["Category"] <- traindata$Category
train_matrix$Category <- as.numeric(as.factor(train_matrix$Category)) - 1
test_matrix <- features[(nrow(traindata)+1):nrow(train_and_test_data),]

data <- data.matrix(train_matrix)
data <- data.frame(data)
rownames(test_matrix) <- NULL

#===================================================================================
#rm(list=setdiff(ls(), c('test_matrix','train_matrix')))
#SPLIT DATA BY 80/20 SPLIT RANDOMLY FOR CROSS VALIDATION TO EVALUATE MODEL
split <- createDataPartition(y=data$Category,p=.80,list=FALSE)
train <- as.matrix(data[split,])
test <- as.matrix(data[-split,])
testing <- data[-split,]

dtrain <- lgb.Dataset(data = train[, 1:ncol(data)-1], label = train[, ncol(data)])
dtest <- lgb.Dataset.create.valid(dtrain, data = test[, 1:ncol(data)-1], label = test[, ncol(data)])
valids <- list(test = dtest)

params <- list(objective = "multiclass", metric = "multi_logloss", num_class = length(unique(data$Category)))

#===================================================================================
#TRAIN
system.time(
  model <- lgb.train(params,
                     dtrain,
                     max_depth = -1, #no limit
                     nrounds = 2000,
                     valids,
                     num_threads = 4,
                     num_leaves = 800, #(default 127)
                     min_data = 100,
                     learning_rate = .02,
                     early_stopping_rounds = 5
  ))

#===================================================================================
data_t <- data.matrix(test_matrix)

#===================================================================================
#FITTING TEST DATA INTO TRAIN MODEL TO PREDICT NEW TARGET VALUES
system.time(
  predictions <- predict(model, data_t)
)

#===================================================================================
#CHANGING FORMAT OF PREDICTIONS TO MATCH THE DESIRED FORMAT
probability_matrix <-  matrix(predictions, byrow=T, ncol= length(unique(data$Category)))
#predictions_index <- apply(probability_matrix, 1, which.max)
df_probability_matrix <-  data.frame(probability_matrix)

#===================================================================================
#PUTTING BACK IN THE INDEX TO DATA SINCE IT WAS REMOVED FOR THE TRAIN
index_start <- 0
index_end <- nrow(testdata)-1
index <- c(index_start:index_end)

#===================================================================================
#REASSIGNING COLUMN NAMES TO DATA ON PREDICTED MATRIX
df_probability_matrix <-  cbind(index,df_probability_matrix)
colnames(df_probability_matrix) <- c("Id","ARSON", "ASSAULT", "BAD CHECKS","BRIBERY","BURGLARY","DISORDERLY CONDUCT",
                                     "DRIVING UNDER THE INFLUENCE", "DRUG/NARCOTIC",
                                     "DRUNKENNESS","EMBEZZLEMENT","EXTORTION", "FAMILY OFFENSES","FORGERY/COUNTERFEITING",
                                     "FRAUD","GAMBLING","KIDNAPPING","LARCENY/THEFT","LIQUOR LAWS","LOITERING",
                                     "MISSING PERSON","NON-CRIMINAL","OTHER OFFENSES","PORNOGRAPHY/OBSCENE MAT",
                                     "PROSTITUTION","RECOVERED VEHICLE","ROBBERY","RUNAWAY","SECONDARY CODES","SEX OFFENSES FORCIBLE",
                                     "SEX OFFENSES NON FORCIBLE","STOLEN PROPERTY","SUICIDE","SUSPICIOUS OCC","TREA","TRESPASS","VANDALISM",
                                     "VEHICLE THEFT", "WARRANTS","WEAPON LAWS")

#===================================================================================
#COPY PROBABILITY MATRIX TO CSV FILE
write.csv(df_probability_matrix, file = "SF_Prediction.csv", row.names = FALSE)
