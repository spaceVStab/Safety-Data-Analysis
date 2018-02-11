library(data.table)
library(dplyr)
library(ggplot2)
library(caret)
library(stringr)
library(reshape2)

churn = read.csv("Customer Churn Data.csv", header = TRUE, stringsAsFactors = FALSE)

churn$churn=str_trim(churn$churn)
churn$international_plan=str_trim(churn$international_plan)
churn$voice_mail_plan=str_trim(churn$voice_mail_plan)

churn$international_plan=ifelse(churn$international_plan=="yes",1,0)
churn$voice_mail_plan=ifelse(churn$voice_mail_plan=="yes",1,0)
churn$churn=ifelse(churn$churn=="True", 1, 0)

#churn$churn=as.factor(churn$churn)
#churn$international_plan=as.factor(churn$international_plan)
#churn$voice_mail_plan=as.factor(churn$voice_mail_plan)

mean_customer_service_calls=churn %>% group_by(churn) %>% summarise(avg=mean(number_customer_service_calls))

# churn$churn=ifelse(churn$churn=="True", 1, 0)
total_churn_customers_by_state=churn %>% group_by(state) %>% summarise(total=sum(churn))
state=as.data.frame(table(churn$state))

colnames(state)=c("state", "total_customers")
colnames(total_churn_customers_by_state)=c("state", "total_churn")
churn_rate_by_state=left_join(total_churn_customers_by_state, state, by="state")

churn_rate_by_state$rate=churn_rate_by_state$total_churn*100/churn_rate_by_state$total_customers

ggplot(churn_rate_by_state, aes(x=reorder(state, -rate), y=rate))+geom_point()+xlab("State")+ylab("Churn Percentage")

# Violin Plots

#pdf("plots.pdf")
dodge=position_dodge(width = 0.4)
ggplot(churn, aes(x=factor(churn), y=number_vmail_messages,fill = factor(churn)))+geom_violin(position = dodge)+geom_boxplot(width=.01, outlier.colour="white", position = dodge)+labs(x="churn",y="number_vmail_messages")

ggplot(churn, aes(x=factor(churn), y=total_day_calls, fill=factor(churn)))+geom_violin(position = dodge)+geom_boxplot(width=.1, outlier.colour="white", position = dodge)+labs(x="churn",y="total_day_calls")
ggplot(churn, aes(x=factor(churn), y=total_day_minutes, fill=factor(churn)))+geom_violin(position = dodge)+geom_boxplot(width=.1, outlier.colour="white", position = dodge)+labs(x="churn",y="total_day_minutes")
ggplot(churn, aes(x=factor(churn), y=total_day_charge, fill=factor(churn)))+geom_violin(position = dodge)+geom_boxplot(width=.1, outlier.colour="white", position = dodge)+labs(x="churn",y="total_day_charge")


ggplot(churn, aes(x=factor(churn), y=total_eve_calls, fill=factor(churn)))+geom_violin(position = dodge)+geom_boxplot(width=.1, outlier.colour="white", position = dodge)+labs(x="churn",y="total_eve_calls")
ggplot(churn, aes(x=factor(churn), y=total_eve_minutes, fill=factor(churn)))+geom_violin(position = dodge)+geom_boxplot(width=.1, outlier.colour="white", position = dodge)+labs(x="churn",y="total_eve_minutes")
ggplot(churn, aes(x=factor(churn), y=total_eve_charge, fill=factor(churn)))+geom_violin(position = dodge)+geom_boxplot(width=.1, outlier.colour="white", position = dodge)+labs(x="churn",y="total_eve_charge")

ggplot(churn, aes(x=factor(churn), y=total_night_calls, fill=factor(churn)))+geom_violin(position = dodge)+geom_boxplot(width=.1, outlier.colour="white", position = dodge)+labs(x="churn",y="total_night_calls")
ggplot(churn, aes(x=factor(churn), y=total_night_minutes, fill=factor(churn)))+geom_violin(position = dodge)+geom_boxplot(width=.1, outlier.colour="white", position = dodge)+labs(x="churn",y="total_night_minutes")
ggplot(churn, aes(x=factor(churn), y=total_night_charge, fill=factor(churn)))+geom_violin(position = dodge)+geom_boxplot(width=.1, outlier.colour="white", position = dodge)+labs(x="churn",y="total_night_charge")

ggplot(churn, aes(x=factor(churn), y=total_intl_calls, fill=factor(churn)))+geom_violin(position = dodge)+geom_boxplot(width=.1, outlier.colour="white", position = dodge)+labs(x="churn",y="total_intl_calls")
ggplot(churn, aes(x=factor(churn), y=total_intl_minutes, fill=factor(churn)))+geom_violin(position = dodge)+geom_boxplot(width=.1, outlier.colour="white", position = dodge)+labs(x="churn",y="total_intl_minutes")
ggplot(churn, aes(x=factor(churn), y=total_intl_charge, fill=factor(churn)))+geom_violin(position = dodge)+geom_boxplot(width=.1, outlier.colour="white", position = dodge)+labs(x="churn",y="total_intl_charge")

ggplot(churn, aes(x=factor(churn), y=number_customer_service_calls, fill=factor(churn)))+geom_violin(position = dodge)+geom_boxplot(width=.05, outlier.colour="white", position = dodge)+labs(x="churn",y="number_customer_service_calls")
#dev.off()


# Heat Maps

# Before standardization
cormat = round(cor(churn[,-c(1,2,5,22)]),2)
melted_cormat = melt(cormat)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + geom_tile()

get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}


reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}

cormat <- reorder_cormat(cormat)
upper_tri <- get_upper_tri(cormat)
melted_cormat <- melt(upper_tri, na.rm = TRUE)

ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+xlab("")+ylab("")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0.5, limit = c(0,1), space = "Lab", 
                       name="Correlation Score") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 8, hjust = 1))+
  coord_fixed()

ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))

# After standardization
preObj = preProcess(churn[,-c(1,2,5,22)], method=c("center", "scale"))
churn_standardized = predict(preObj, churn[,-c(1,2,5,22)])
cormat_standardized = round(cor(churn_standardized),2)
melted_cormat_standardized = melt(cormat_standardized)
ggplot(data = melted_cormat_standardized, aes(x=Var1, y=Var2, fill=value)) + geom_tile()

# No change in maps


# Training and Testing sets

set.seed(123)
trainIndex <- createDataPartition(churn$churn, p = .8, list = FALSE)
churnTrain=churn[ trainIndex,]
churnTest=churn[-trainIndex,]

churnTrain=churnTrain[,-c(1,2,5)]
preProcValues <- preProcess(x = churnTrain,method = c("center", "scale"))
ctrl <- trainControl(method="repeatedcv",repeats = 3)
knnFit <- train(factor(churn) ~ ., data = churnTrain, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
plot(knnFit)

library(e1071)
#Full Data set can be used for cross validation
knn.cross <- tune.knn(x = scale(churnTrain[,-19]), y = factor(churnTrain[,19]), k = 1:20,tunecontrol=tune.control(sampling = "cross"), cross=10)
#Summarize the resampling results set
summary(knn.cross)


metric=read.csv("churnModelAccuracy.csv", header = TRUE)
ggplot(data=metric, aes(x=Model, y=Test_Acc))+geom_boxplot()+expand_limits(y=0.8)










# New variables

churn$total_minutes=churn$total_day_minutes+churn$total_eve_minutes+churn$total_night_minutes
churn$total_charge=churn$total_day_charge+churn$total_eve_charge+churn$total_night_charge
churn$day_rate=churn$total_day_charge/churn$total_day_minutes
churn$eve_rate=churn$total_eve_charge/churn$total_eve_minutes
churn$night_rate=churn$total_night_charge/churn$total_night_minutes
churn$init_rate=churn$total_intl_charge/churn$total_intl_minutes

churn$day_rate[is.na(churn$day_rate)]=mean(churn$day_rate, na.rm = TRUE)
churn$night_rate[is.na(churn$night_rate)]=mean(churn$night_rate, na.rm = TRUE)
churn$eve_rate[is.na(churn$eve_rate)]=mean(churn$eve_rate, na.rm = TRUE)
churn$init_rate[is.na(churn$init_rate)]=mean(churn$init_rate, na.rm = TRUE)

