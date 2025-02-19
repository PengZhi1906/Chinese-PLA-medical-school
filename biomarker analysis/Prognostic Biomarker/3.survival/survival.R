# 加载必要的库
library(survival)
library(survminer)

# 设置工作目录
setwd("F:\\图神经网络\\4.survival")

# 定义生存分析函数
bioSurvival <- function(inputFile=null, outFile=null) {
  # 读取数据文件
  rt <- read.table(inputFile, header=T, sep="\t", check.names=F)
  
  # 计算风险组之间的生存差异
  diff <- survdiff(Surv(futime, fustat) ~ risk, data=rt)
  pValue <- 1 - pchisq(diff$chisq, df=1)
  
  # 计算生存曲线
  fit <- survfit(Surv(futime, fustat) ~ risk, data = rt)
  
  # 计算5年生存率
  surv5yr <- summary(fit, times=5)$surv
  
  # 生成生存曲线图，并添加5年生存率的表格
  surPlot <- ggsurvplot(fit, 
                        data=rt,
                        conf.int=TRUE,
                        pval=formatC(pValue, format="e", digits=3), # 使用科学计数法展示p值
                        pval.size=6,
                        surv.median.line = "hv",
                        legend.title="Risk",
                        legend.labs=c("High risk", "Low risk"),
                        xlab="Time(years)",
                        break.time.by = 1,
                        palette=c("#E41A1C", "#377EB8"),
                        xlim=c(0, 6), # 限制x轴范围到5年
                        risk.table=TRUE,
                        risk.table.col = "strata",
                        risk.table.height=0.25,
                        risk.table.y.text=TRUE,
                        risk.table.text=paste0("5-year survival: ", round(surv5yr, 3))) # 只显示5年生存率
  
  # 保存生存曲线图
  pdf(file=outFile, onefile=FALSE, width=6.5, height=5.5)
  print(surPlot)
  dev.off()
}

# 调用函数进行生存分析
bioSurvival(inputFile="risk.txt", outFile="survival12.pdf")