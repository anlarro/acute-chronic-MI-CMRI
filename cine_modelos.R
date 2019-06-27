setwd("D:ANDRES/Doctorado/Proyectos/Mis proyectos/Textura_CMRI/NI_acute_chronic_CINEcomparison/R")

#Librerías a utilizar
library(caret)
library(e1071)
library(MASS)
library(randomForest)
library(doParallel)
library(klaR)
library(rgl)
library(ROCR)
source("myFS.R")

modelos<-c("parRF","svmRadial","svmPoly")
parRFGrid<- expand.grid(.mtry = seq(2,10,by=2))
svmRadialGrid<-expand.grid(.sigma = 10^(seq(-1, 0)),.C = 2^(seq(-1, 2)))
svmPolyGrid<-expand.grid(.degree=3,.scale=10^(seq(-3, -1)),.C=2^(seq(-2, 0)))
grillas<-list(parRFGrid,svmRadialGrid,svmPolyGrid)

#########################33
#Cargamos datos
cine=read.csv('cine.csv')
Area_cine<-cine[,1]
cine<-cine[,-1]
cine<-cine[,!apply(is.na(cine),2,any)]

input=cbind(cine[,ncol(cine)],cine[,-ncol(cine)])  #cine
names(input)[1]<-"CLASS"

set.seed(11)
folds<-unname(createMultiFolds(input[,1],k=5,times = 10)) #training indices for k-fold generalization performance

#Ranking SVM
rankSVMRFE<-lapply(folds,svmRFE.wrap,input,k=5,halve.above=100)
top.featuresSVMRFE=WriteFeatures(rankSVMRFE,input,save=F)

#Ranking Fisher
#rankFisher<-lapply(folds,fisher.wrap,input)
#top.featuresFisher=WriteFeatures(rankFisher,input,save=F)


#Estimate Generalization Performance
sweepModelosSVMRFE<-list()
rocsModelosSVMRFE<-list()
for (n in 1:length(modelos)){
  ptm<-Sys.time()
  cl <- makeCluster(8) #use 6 cores, ie for an 8-core machine
  clusterEvalQ(cl, {library(caret); library(pROC)})
  sweepModelosSVMRFE[[n]]<-parLapply(cl,seq(1,ncol(input)-1,by=1),FeatSweepROC.wrap,rankSVMRFE,input,modelos[n],grillas[[n]],k=5)
  stopCluster(cl)
  rocsModelosSVMRFE[[n]]<-sapply(sweepModelosSVMRFE[[n]], function(y) mean(sapply(y, function(x) x$ROC)))
  print(paste('SVM-RFE',n))
  print(Sys.time()-ptm)
}


#Plot profile SVM-RFE
tiff("profileCINE.tiff", width = 3200, height = 3200, units = 'px', res = 600, compression="lzw")
plot(seq(1,length(rocsModelosSVMRFE[[3]]),by=1),rocsModelosSVMRFE[[3]][seq(1,length(rocsModelosSVMRFE[[3]]),by=1)],type="l",pch=16,xlim=c(0,length(rocsModelosSVMRFE[[3]])),ylim=c(0.55, 1),yaxt="n",xaxt="n",xlab="Number of features",ylab="AUC (Repeated cross-validation)",main="Cine MRI",lwd=2,lty=1,col="blue")
grid()
points(x=which.max(rocsModelosSVMRFE[[3]]),y=max(rocsModelosSVMRFE[[3]]),pch=16,col="blue")
points(x=which.max(rocsModelosSVMRFE[[3]]),y=max(rocsModelosSVMRFE[[3]]),pch=8,col="blue")

lines(1:length(rocsModelosSVMRFE[[1]]),rocsModelosSVMRFE[[1]],type="l",pch=15,col="red",lty=5,lwd=2)
points(x=which.max(rocsModelosSVMRFE[[1]]),y=max(rocsModelosSVMRFE[[1]]),pch=16,col="red")
points(x=which.max(rocsModelosSVMRFE[[1]]),y=max(rocsModelosSVMRFE[[1]]),pch=8,col="red")

lines(1:length(rocsModelosSVMRFE[[2]]),rocsModelosSVMRFE[[2]],type="l",pch=17,col="black",lty=3,lwd=2)
points(x=which.max(rocsModelosSVMRFE[[2]]),y=max(rocsModelosSVMRFE[[2]]),pch=16,col="black")
points(x=which.max(rocsModelosSVMRFE[[2]]),y=max(rocsModelosSVMRFE[[2]]),pch=8,col="black")

ylabel <- seq(0.6, 1,0.1)
axis(2, at = ylabel,las=1) #las=1 para que muestre texto horizontal
xlabel <- seq(0,250,50)
axis(1, at = xlabel)
axis(1,at=10,labels=F)
mtext("10",side=1,line=0.5,at=10,cex=0.8)
abline(v=10,lty=3,col="gray")
legend("bottomright",c(modelos[3],modelos[1],modelos[2]),lty=c(1,5,3),lwd=c(2,2,2),col=c("blue","red","black"))
dev.off()


#Best models
modelosTrained<-list()
for (i in 1:length(modelos)){
  compute<-function(x){
    set.seed(11)
    model = train(input[x$train.data.ids, 1+x$feature.ids[1:which.max(rocsModelosSVMRFE[[i]])],drop=F],
                  input[x$train.data.ids, 1],
                  method = modelos[i],
                  metric = "ROC",
                  tuneGrid=grillas[[i]],
                  preProc = c("center", "scale"),
                  trControl = trainControl(summaryFunction = twoClassSummary,allowParallel=F,classProbs = TRUE,method='cv',number=5))
    predicciones<-predict(model,input[x$test.data.ids,1+x$feature.ids[1:which.max(rocsModelosSVMRFE[[i]])],drop=F],type='prob')[,1]
    labels<-input[x$test.data.ids, 1]
    return(list(predicciones=predicciones, labels=labels))
  }
  modelosTrained[[i]]<-lapply(rankSVMRFE,compute)
}

################################
#Unique ROC plot
tiff("rocCINE.tiff", width = 3200, height = 3200, units = 'px', res = 600, compression="lzw")

#parRF
pred<-prediction(lapply(modelosTrained[[1]],function (x) x$predicciones),lapply(modelosTrained[[1]],function (x) x$labels),label.ordering=c("CI","AI"))

auc1_mean<-mean(unlist(performance(pred,"auc")@y.values))
auc1_sd<-sd(unlist(performance(pred,"auc")@y.values))
CUTOFF<-0.32
sensibilidad<-vector()
especificidad<-vector()
perf <- performance(pred, 'sens', 'spec')
for (k in 1:length(modelosTrained[[1]])){
  ix <- which.min(abs(perf@alpha.values[[k]] - CUTOFF)) #good enough in our case
  sensibilidad<-c(sensibilidad,perf@y.values[[k]][ix]) #note the order of arguments to `perfomance` and of x and y in `perf`
  especificidad<-c(especificidad,perf@x.values[[k]][ix])
}

mean(sensibilidad)*mean(especificidad)

ppv<-vector()
npv<-vector()
perf <- performance(pred, 'ppv', 'npv')
for (k in 1:length(modelosTrained[[1]])){
  ix <- which.min(abs(perf@alpha.values[[k]] - CUTOFF)) #good enough in our case
  ppv<-c(ppv,perf@y.values[[k]][ix]) #note the order of arguments to `perfomance` and of x and y in `perf`
  npv<-c(npv,perf@x.values[[k]][ix])
}
perf<-performance(pred,"tpr","fpr")
plot(perf,col="red",lwd=2,lty=5,avg="threshold",ylab="Sensitivity",xlab="1-Specificity",main="Cine MRI") #curva ROC
points(x=1-(mean(especificidad)),y=mean(sensibilidad)-0.005,col="red",pch=19)
#points(x=1-(mean(especificidad)),y=mean(sensibilidad)-0.005,col="blue",pch=8)

#svmRadial
pred<-prediction(lapply(modelosTrained[[2]],function (x) x$predicciones),lapply(modelosTrained[[2]],function (x) x$labels),label.ordering=c("CI","AI"))

auc2_mean<-mean(unlist(performance(pred,"auc")@y.values))
auc2_sd<-sd(unlist(performance(pred,"auc")@y.values))
CUTOFF<-0.31
sensibilidad<-vector()
especificidad<-vector()
perf <- performance(pred, 'sens', 'spec')
for (k in 1:length(modelosTrained[[2]])){
  ix <- which.min(abs(perf@alpha.values[[k]] - CUTOFF)) #good enough in our case
  sensibilidad<-c(sensibilidad,perf@y.values[[k]][ix]) #note the order of arguments to `perfomance` and of x and y in `perf`
  especificidad<-c(especificidad,perf@x.values[[k]][ix])
}
mean(sensibilidad)*mean(especificidad)

ppv<-vector()
npv<-vector()
perf <- performance(pred, 'ppv', 'npv')
for (k in 1:length(modelosTrained[[2]])){
  ix <- which.min(abs(perf@alpha.values[[k]] - CUTOFF)) #good enough in our case
  ppv<-c(ppv,perf@y.values[[k]][ix]) #note the order of arguments to `perfomance` and of x and y in `perf`
  npv<-c(npv,perf@x.values[[k]][ix])
}
perf<-performance(pred,"tpr","fpr")
plot(perf,col="black",lwd=2,lty=3,avg="threshold",add=T) #curva ROC
points(x=1-(mean(especificidad)),y=mean(sensibilidad)-0.005,col="black",pch=19)
#points(x=1-(mean(especificidad)),y=mean(sensibilidad)-0.005,col="red",pch=8)

#svmPoly
pred<-prediction(lapply(modelosTrained[[3]],function (x) x$predicciones),lapply(modelosTrained[[3]],function (x) x$labels),label.ordering=c("CI","AI"))

auc3_mean<-mean(unlist(performance(pred,"auc")@y.values))
auc3_sd<-sd(unlist(performance(pred,"auc")@y.values))
CUTOFF<-0.35
sensibilidad<-vector()
especificidad<-vector()
perf <- performance(pred, 'sens', 'spec')
for (k in 1:length(modelosTrained[[3]])){
  ix <- which.min(abs(perf@alpha.values[[k]] - CUTOFF)) #good enough in our case
  sensibilidad<-c(sensibilidad,perf@y.values[[k]][ix]) #note the order of arguments to `perfomance` and of x and y in `perf`
  especificidad<-c(especificidad,perf@x.values[[k]][ix])
}
mean(sensibilidad)*mean(especificidad)

ppv<-vector()
npv<-vector()
perf <- performance(pred, 'ppv', 'npv')
for (k in 1:length(modelosTrained[[3]])){
  ix <- which.min(abs(perf@alpha.values[[k]] - CUTOFF)) #good enough in our case
  ppv<-c(ppv,perf@y.values[[k]][ix]) #note the order of arguments to `perfomance` and of x and y in `perf`
  npv<-c(npv,perf@x.values[[k]][ix])
}
perf<-performance(pred,"tpr","fpr")
plot(perf,col="blue",lwd=2,lty=1,avg="threshold",add=T) #curva ROC
points(x=1-(mean(especificidad)),y=mean(sensibilidad)-0.005,col="blue",pch=19)
#points(x=1-(mean(especificidad)),y=mean(sensibilidad)-0.005,col="black",pch=8)

abline(0,1)
ley<-c(paste0(modelos[3], "     AUC = ",sprintf("%.2f",round(auc3_mean,2))," ± ",sprintf("%.2f",round(auc3_sd,2))),paste0(modelos[1],"        AUC = ",sprintf("%.2f",round(auc1_mean,2))," ± ",sprintf("%.2f",round(auc1_sd,2))),paste0(modelos[2], " AUC = ",sprintf("%.2f",round(auc2_mean,2))," ± ",sprintf("%.2f",round(auc2_sd,2))))
legend("bottomright",ley,lty=c(1,5,3),lwd=c(2,2,2),col=c("blue","red","black"))
dev.off()

#######################################################
#Scatter Plot
tiff("scatterCINE.tiff", width = 3200, height = 3200, units = 'px', res = 600, compression="lzw")
plot(input[,1+top.featuresSVMRFE$FeatureID[2]],input[,1+top.featuresSVMRFE$FeatureID[3]],lwd=2,pch=c(1,4)[as.numeric(input$CLASS)],col=c("blue", "red")[as.numeric(input$CLASS)],xlab=names(input)[1+top.featuresSVMRFE$FeatureID[2]],ylab=names(input)[1+top.featuresSVMRFE$FeatureID[3]])
legend("topleft",c("CI", "NI"),pch=c(1,4),col=c("blue","red"))
dev.off()

plot(input[,1+top.featuresSVMRFE$FeatureID[2]],input[,1+top.featuresSVMRFE$FeatureID[3]],lwd=2,pch=c(1,4)[as.numeric(input$CLASS)],col=c("blue", "red")[as.numeric(input$CLASS)],xlab=names(input)[1+top.featuresSVMRFE$FeatureID[2]],ylab=names(input)[1+top.featuresSVMRFE$FeatureID[3]])

plot3d(input[,1+top.featuresSVMRFE$FeatureID[2]],input[,1+top.featuresSVMRFE$FeatureID[3]],input[,1+top.featuresSVMRFE$FeatureID[4]], col=c("blue", "red")[as.numeric(input$CLASS)],type='s',size=1) #magenta(miocardio),amarillo(cicatriz)

plot3d(input[,1+rankSVMRFE_full[2]],input[,1+rankSVMRFE_full[3]],input[,1+rankSVMRFE_full[4]], col=c("blue", "red")[as.numeric(input$CLASS)],type='s',size=1) #magenta(miocardio),amarillo(cicatriz)

plot3d(input[,1+rankFisher_full[2]],input[,1+rankFisher_full[3]],input[,1+rankFisher_full[4]], col=c("blue", "red")[as.numeric(input$CLASS)],type='s',size=1) #magenta(miocardio),amarillo(cicatriz)

###############################333

