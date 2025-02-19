#install.packages("glmnet")
#install.packages("survival")
#install.packages('survminer')



library(glmnet)
library(survival)
library(survminer)
inputFile="expTime.txt"        
setwd("F:\\图神经网络\\model")      
rt=read.table(inputFile, header=T, sep="\t", check.names=F, row.names=1)    


multiCox=coxph(Surv(futime, fustat) ~ ., data = rt)
multiCox=step(multiCox, direction="both")
multiCoxSum=summary(multiCox)


outMultiTab=data.frame()
outMultiTab=cbind(
		          coef=multiCoxSum$coefficients[,"coef"],
		          HR=multiCoxSum$conf.int[,"exp(coef)"],
		          HR.95L=multiCoxSum$conf.int[,"lower .95"],
		          HR.95H=multiCoxSum$conf.int[,"upper .95"],
		          pvalue=multiCoxSum$coefficients[,"Pr(>|z|)"])
outMultiTab=cbind(id=row.names(outMultiTab),outMultiTab)
write.table(outMultiTab, file="multiCox.txt", sep="\t", row.names=F, quote=F)


score=predict(multiCox, type="risk", newdata=rt)
coxGene=rownames(multiCoxSum$coefficients)
coxGene=gsub("`", "", coxGene)
outCol=c("futime", "fustat", coxGene)
risk=as.vector(ifelse(score>median(score), "high", "low"))
outTab=cbind(rt[,outCol], riskScore=as.vector(score), risk)
write.table(cbind(id=rownames(outTab),outTab), file="risk.txt", sep="\t", quote=F, row.names=F)



