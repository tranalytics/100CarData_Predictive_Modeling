#set working directory
setwd("C:/Users/Alex/Desktop/Nauto/Data")
#install and load libraries
library(caret)

library(dplyr)
#load files
event_video<-read.delim('100CarEventVideoReducedData_v1_5.txt',header=FALSE,sep='\t')
base_video<-read.delim('100CarBaselineVideoReducedData_v1_1.txt',header=FALSE,sep='\t')
base_video_m<-base_video[c(1:5,8:13,17:26)]
headers<-c('webfileid','vehicle_id','subject_id','age','gender','behavior_1','behavior_2',
           'behavior_3','distraction_1','distraction_2','distraction_3','surface','flow','lanes','density',
           'control','junction','alightment','locality','lighting','weather')
names(base_video_m)<-headers
base_video_m$outcome<-'normal'
ev_c<-c(1,2,5,6,7,8,16,17,18,34,38,42,49,50,51,52,53,54,55,56,57,58)
event_video_m<-event_video[ev_c]

headers2<-c('webfileid','vehicle_id','outcome','subject_id','age','gender','behavior_1','behavior_2',
            'behavior_3','distraction_1','distraction_2','distraction_3','surface','flow','lanes','density',
            'control','junction','alightment','locality','lighting','weather')
names(event_video_m)<-headers2
event_video_m$outcome<-'event'

video_data<-rbind(base_video_m,event_video_m)

#convert outcome to factor
video_data$outcome<-factor(video_data$outcome)
#remove ids, behaviors, and distractions
video_data1<-video_data[,-c(1:3,6:11)]
#Some genders are blank, check to see how many
nrow(video_data1[which(video_data1$gender==' '),])
#check to see how many are blank and are an event                              
nrow(video_data1[which(video_data1$gender==' '& video_data1$outcome=='event'),])
#remove those values
video_reduced<-droplevels(video_data1[-which(video_data1$gender==' '),])
#Remove entries with 'no analyzed data'
for (i in 2:12){
  video_reduced2<-droplevels(video_reduced[-which(video_reduced[,i]=='No analyzed data'),])
  print(unique(video_reduced[,i]))}

#created a function for to see the levels
seelevels<-function(df=video_reduced2){
  
  for (i in 1:ncol(df)){
    print (names(df)[i])
    print(unique(df[,i]))
  }}
seelevels()
#Noticed that different factors were created due to capitalization errors. lowercase everything just incase
for (i in 1:ncol(video_reduced2)){
  levels(video_reduced2[[i]])<-tolower(levels(video_reduced2[[i]]))
}
#density has an error because of a period & phrase
levels(video_reduced2$density)<-gsub('\\.','',levels(video_reduced2$density), perl = TRUE)
levels(video_reduced2$density)<-gsub('unstable flow - temporary restrictions substantially slow driver','unstable flow, temporary restrictions, substantially slow driver',levels(video_reduced2$density), perl = TRUE)
#change age to numeric and outcome to factor
video_reduced2$age<-as.numeric(as.character(video_reduced2$age))
video_reduced2$outcome<-as.factor(video_reduced2$outcome)
#Check for near zero variance

nzv <- nearZeroVar(video_reduced2[,-(13)],saveMetrics = TRUE)
sum(nzv$nzv) #no zero variance vectors

#stratify data
#setseed
set.seed(33)
df1<-video_reduced2
split1 <- createDataPartition(df1$outcome, p = .8, list = FALSE)
df1train <- df1[ split1,]
df1test  <- df1[-split1,]
o_df1train<-df1[split1,13]
o_df1test<-df1[-split1,13]
typeof(o_df1test)
#create cv folds
myfolds1<-createFolds(df1train$outcome,5)
#check for stratification
a <- myfolds1$Fold1
table(df1train$outcome[a]) / length(a) 
#set controls, each model will have same data, control and seed therefore becomparable
mycontrol<-trainControl(summaryFunction = twoClassSummary,
                        classProbs = TRUE, 
                        verboseIter = TRUE,
                        savePredictions = TRUE,
                        index=myfolds1)

names(df1)
#random forest
modelo_rf<-train(outcome~.,
                 data=df1train,
                 tuneGrid=data.frame(mtry=c(2,5,7,10,15)),      
                 method='ranger',
                 metric='ROC',
                 trControl = mycontrol)
#decision tree
modelo_tree<-train(outcome~.,
                   data=df1train,
                   method='rpart',
                   metric='ROC',
                   trControl = mycontrol)

#tried no upsampling with a couple different models
#bayesglm model
modelo_bayesglm<-train(outcome~.,
                       data=df1train,
                       metric='ROC',
                       method='bayesglm',
                       trControl=mycontrol)
#treebag
modelo_treebag<-train(outcome~.,
                          data=df1train,
                      method = "treebag",
                      nbagg = 50,
                      trControl=mycontrol)
#resample models and compare
model_list<-list(model1=modelo_tree,model2=modelo_rf,model3=modelo_bayesglm,model4=modelo_treebag)
resamples<-resamples(model_list)
summary(resamples)

confusionMatrix(tree.pred,df1test$outcome)
confusionMatrix(rf.pred,df1test$outcome)
confusionMatrix(predict(modelo_bayesglm,df1test),df1test$outcome)
confusionMatrix(predict(modelo_bagEarthGCV,df1test),df1test$outcome)
confusionMatrix(predict(modelo_treebag,df1test),df1test$outcome)

#using upsampling
?upSample
up_traindf1 <- upSample(x = df1train[,-13],
                     y = df1train$outcome)     
table(up_traindf1$Class)
str(up_traindf1)
upmodelo_rf1<-train(Class~.,
                 data=up_traindf1,
                 tuneGrid=data.frame(mtry=c(2,5,7,10,15)),      
                 method='ranger',
                 metric='ROC',
                 trControl = mycontrol)
upmodelo_tree1<-train(Class~.,
                      data=up_traindf1,
                      method='rpart',
                      metric='ROC',
                      trControl = mycontrol)
upmodelo_bayesglm1<-train(Class~.,
                          data=up_traindf1,
                          metric='ROC',
                          method='bayesglm',
                          trControl=mycontrol)
#treebag
upmodelo_treebag1<-train(Class~.,
                         data=up_traindf1,
                         method = "treebag",
                         nbagg = 50,
                         trControl=mycontrol)

#resample models and compare
up_models<-list(m1=upmodelo_rf1,m2=upmodelo_tree1,m3=upmodelo_bayesglm1,m4=upmodelo_bagEarthGCV1, m5=upmodelo_treebag1)
resamplesup1<-resamples(up_models)
summary(resamplesup1)

confusionMatrix(predict(upmodelo_rf,df1test),df1test$outcome)

model_list2<-list(model1=modelo_rf,model2=upmodelo_rf)
resamples2<-resamples(model_list2)
summary(resamples2)
#turn age and weather into ranks
df2<-df1%>%mutate(age_rank=ntile(df1$age,3),
weather_cat= ifelse(weather %in% c('clear','cloudy'),'clear','wet'))
#remove age and weather, convert to factor
df2<-df2[,-c(1,12)]
df2$age_rank<-factor(df2$age_rank)
df2$weather_cat<-factor(df2$weather_cat)

#set up df2
df2train <- df2[ split1,]
df2test  <- df2[-split1,]
#no upsample rf df2
modelo_rf2<-train(outcome~.,
                 data=df2train,
                 tuneGrid=data.frame(mtry=c(2,5,7,10,15)),      
                 method='ranger',
                 metric='ROC',
                 trControl = mycontrol)
#no up sample decision tree df2
#decision tree
modelo_tree2<-train(outcome~.,
                   data=df2train,
                   method='rpart',
                   metric='ROC',
                   trControl = mycontrol)
#bayesglm model df2
modelo_bayesglm2<-train(outcome~.,
                       data=df2train,
                       metric='ROC',
                       method='bayesglm',
                       trControl=mycontrol)
#treebag df2
modelo_treebag2<-train(outcome~.,
                      data=df2train,
                      method = "treebag",
                      nbagg = 50,
                      trControl=mycontrol)
model_list2<-list(model1=modelo_tree2,model2=modelo_rf2,model3=modelo_bayesglm2,model4=modelo_treebag2)
resamples2<-resamples(model_list2)
summary(resamples2)

#upsample df2
up_traindf2 <- upSample(x = df2train[,-11],
                        y = df2train$outcome)     
upmodelo_rf2<-train(Class~.,
                   data=up_traindf2,
                   tuneGrid=data.frame(mtry=c(2,5,7,10,15)),      
                   method='ranger',
                   metric='ROC',
                   trControl = mycontrol)

upmodelo_tree2<-train(Class~.,
                   data=up_traindf2,
                   method='rpart',
                   metric='ROC',
                   trControl = mycontrol)
upmodelo_bayesglm2<-train(Class~.,
                       data=up_traindf2,
                       metric='ROC',
                       method='bayesglm',
                       trControl=mycontrol)
#treebag
upmodelo_treebag2<-train(Class~.,
                      data=up_traindf2,
                      method = "treebag",
                      nbagg = 50,
                      metric='ROC',
                      trControl=mycontrol)
up_models2<-list(m1=upmodelo_rf2,m2=upmodelo_tree2,m3=upmodelo_bayesglm2, m4=upmodelo_treebag2)
resamplesup2<-resamples(up_models2)
summary(resamplesup2)
#DUMMY TRY2
dmy <- dummyVars(" ~ .-outcome", data = df1)
dmydf1 <- data.frame(predict(dmy, newdata = df1))
#combine response variable back in
dmydf1<-cbind(dmydf1,df1$outcome)
names(dmydf1)[78]<-'outcome'

dmydf1train<-dmydf1[split1,]
dmydf1test<-dmydf1[-split1,]
##rename outcome
names(dmydf1)[78]<-'outcome'
dmydf1train<-dmydf1[split1,]
dmydf1test<-dmydf1[-split1,]
dmymodelo_rf1<-train(outcome~.,
                    data=dmydf1,
                    tuneGrid=data.frame(mtry=c(2,5,7,10,15)),      
                    method='ranger',
                    metric='ROC',
                    trControl = mycontrol)

#tree where all values are numeric except for outcome
dmymodelo_tree1<-train(outcome~.,
                    data=dmydf1,
                    method='rpart',
                    metric='ROC',
                    trControl = mycontrol)
plot (dmymodelo_rf1)
confusionMatrix(predict(dmymodelo_rf1,dmydf1test),dmydf1test$outcome)

#upsample dmydf1
dmyup_traindf1 <- upSample(x = dmydf1train[,-78],
                        y = dmydf1train$outcome)     
dmyupmodelo_rf1<-train(Class~.,
                    data=dmyup_traindf1,
                    tuneGrid=data.frame(mtry=c(2,5,7,10,15)),      
                    method='ranger',
                    metric='ROC',
                    trControl = mycontrol)
dmyupmodelo_tree1<-train(Class~.,
                      data=dmyup_traindf1,
                      method='rpart',
                      metric='ROC',
                      trControl = mycontrol)
dmyupmodelo_bayesglm1<-train(Class~.,
                          data=dmyup_traindf1,
                          metric='ROC',
                          method='bayesglm',
                          trControl=mycontrol)
#treebag
dmyupmodelo_treebag1<-train(Class~.,
                         data=dmyup_traindf1,
                         method = "treebag",
                         nbagg = 50,
                         trControl=mycontrol)
plot(dmyupmodelo_rf1)
