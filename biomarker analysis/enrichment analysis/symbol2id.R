#if (!requireNamespace("BiocManager", quietly = TRUE))
#    install.packages("BiocManager")
#BiocManager::install("org.Hs.eg.db")


library("org.Hs.eg.db")         #引用包
inputFile="intersect.txt"       #输入文件
setwd("C:\\Users\\15.symbo2id")      #设置工作目录
rt=read.table(inputFile,sep="\t",check.names=F,header=F)    #读取输入文件
genes=as.vector(rt[,1])         #获取基因列表
entrezIDs <- mget(genes, org.Hs.egSYMBOL2EG, ifnotfound=NA)    #找出基因对应的id
entrezIDs <- as.character(entrezIDs)

#输出基因id的结果文件
out=cbind(rt,entrezID=entrezIDs)
colnames(out)[1]="Gene"
write.table(out,file="id.txt",sep="\t",quote=F,row.names=F)



