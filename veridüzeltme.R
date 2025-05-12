# Paketlerin y??klenmesi (sadece ilk kez ??al????t??r??l??r)
install.packages(c("TTR", "quantmod", "caret", "randomForest", "FSelector", "Boruta"))
library(Boruta)

# Gerekli k??t??phaneler
library(TTR)
library(quantmod)
library(caret)
library(randomForest)
library(FSelector)
library(readr)

# Veri setinin okunmas?? ve tarih s??tununun silinmesi
btc <- read.csv("C:/Users/yildi/OneDrive/Desktop/projeCrypto/btc_verisi.csv")
btc$Date <- NULL

# Teknik indikat??rlerin eklenmesi
btc$RSI <- RSI(btc$Close, n = 14)

macd_result <- MACD(btc$Close, nFast = 12, nSlow = 26, nSig = 9, maType = EMA)
btc$MACD <- macd_result[, 1]
btc$Signal <- macd_result[, 2]

btc$SMA20 <- SMA(btc$Close, n = 20)
btc$EMA20 <- EMA(btc$Close, n = 20)

# NA olu??an ilk 34 sat??r?? temizle
btc <- btc[-c(1:34), ]

# Sadece say??sal verileri al
btc_numeric <- btc[sapply(btc, is.numeric)]

# Hedef de??i??keni olu??tur (??rnek: fiyat ertesi g??n artt?? m???)
btc_numeric$Target <- ifelse(dplyr::lead(btc_numeric$Close, 1) > btc_numeric$Close, 1, 0)
btc_numeric <- na.omit(btc_numeric)  # lead nedeniyle olu??an NA'lar?? sil

# High ve Low s??tunlar??n?? ????kar (istenmiyor)
btc_numeric$High <- NULL
btc_numeric$low <- NULL

# Normalize et
preprocess_minmax <- preProcess(btc_numeric[, -ncol(btc_numeric)], method = c("range"))
btc_scaled <- predict(preprocess_minmax, btc_numeric[, -ncol(btc_numeric)])
btc_scaled$Target <- as.factor(btc_numeric$Target)  # Hedefi ekle

# ---- 1. ??zellik Se??imi: Boruta ----
# Boruta y??ntemi ile ??zellik se??imi
boruta_result <- Boruta(Target ~ ., data = btc_scaled, doTrace = 2)
print(boruta_result)

# E??er karars??z ??zellikler varsa bunlar?? da dahil et
final_vars_boruta <- getSelectedAttributes(boruta_result, withTentative = TRUE)

if (length(final_vars_boruta) == 0) {
  stop("Boruta ile hi??bir ??zellik se??ilemedi.")
} else {
  btc_boruta <- btc_scaled[, c(final_vars_boruta, "Target"), drop = FALSE]
  btc_boruta <- as.data.frame(btc_boruta)
}

str(btc_boruta)

# ---- 2. ??zellik Se??imi: Recursive Feature Elimination (RFE) ----
# ??zellikleri se??mek i??in RFE (Recursive Feature Elimination) uygulamas??
X <- btc_scaled[, -ncol(btc_scaled)]  # Target d??????ndaki ??zellikler
y <- btc_scaled$Target  # Hedef de??i??ken

# RFE kontrol??
control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)

# RFE uygula
rfe_result <- rfe(X, y, sizes = c(3, 5, 7), rfeControl = control)

# En iyi ??zellikleri al
selected_features_rfe <- predictors(rfe_result)
btc_rfe <- btc_scaled[, c(selected_features_rfe, "Target")]

# ---- 3. ??zellik Se??imi: Random Forest Feature Importance ----
# Random Forest ile ??zellik se??imi
rf_model <- randomForest(x = X, y = y, importance = TRUE)

# ??zelliklerin ??nem s??ras??n?? al
importance_vals <- importance(rf_model)
selected_features_rf <- rownames(importance_vals)[order(importance_vals[, 1], decreasing = TRUE)][1:3]
btc_rf <- btc_scaled[, c(selected_features_rf, "Target")]

# Set seed for reproducibility
set.seed(42)

# E??itim ve test verisine b??lme i??lemi
trainIndex <- createDataPartition(btc_scaled$Target, p = .7, list = FALSE)
train_data <- btc_scaled[trainIndex, ]
test_data <- btc_scaled[-trainIndex, ]

# ---- 4. Modelleme ve ??zellik Setlerini Kullanma ----

# Boruta ??zellik seti ile 4 model

# (1) Do??rusal Regresyon: Boruta ??zellik seti
lm_model_boruta <- train(Target ~ ., data = btc_boruta[trainIndex, ], method = "glm")
lm_pred_boruta <- predict(lm_model_boruta, btc_boruta[-trainIndex, ])

# (2) KNN Modeli: Boruta ??zellik seti
knn_model_boruta <- train(Target ~ ., data = btc_boruta[trainIndex, ], method = "knn", tuneLength = 10)
knn_pred_boruta <- predict(knn_model_boruta, btc_boruta[-trainIndex, ])

# (3) Karar A??a??lar?? (Random Forest): Boruta ??zellik seti
rf_model_boruta <- randomForest(Target ~ ., data = btc_boruta[trainIndex, ])
rf_pred_boruta <- predict(rf_model_boruta, btc_boruta[-trainIndex, ])

# (4) Destek Vekt??r Makinesi (SVM): Boruta ??zellik seti
svm_model_boruta <- train(Target ~ ., data = btc_boruta[trainIndex, ], method = "svmRadial")
svm_pred_boruta <- predict(svm_model_boruta, btc_boruta[-trainIndex, ])


# RFE ??zellik seti ile 4 model

# (1) Do??rusal Regresyon: RFE ??zellik seti
lm_model_rfe <- train(Target ~ ., data = btc_rfe[trainIndex, ], method = "glm")
lm_pred_rfe <- predict(lm_model_rfe, btc_rfe[-trainIndex, ])

# (2) KNN Modeli: RFE ??zellik seti
knn_model_rfe <- train(Target ~ ., data = btc_rfe[trainIndex, ], method = "knn", tuneLength = 10)
knn_pred_rfe <- predict(knn_model_rfe, btc_rfe[-trainIndex, ])

# (3) Karar A??a??lar?? (Random Forest): RFE ??zellik seti
rf_model_rfe <- randomForest(Target ~ ., data = btc_rfe[trainIndex, ])
rf_pred_rfe <- predict(rf_model_rfe, btc_rfe[-trainIndex, ])

# (4) Destek Vekt??r Makinesi (SVM): RFE ??zellik seti
svm_model_rfe <- train(Target ~ ., data = btc_rfe[trainIndex, ], method = "svmRadial")
svm_pred_rfe <- predict(svm_model_rfe, btc_rfe[-trainIndex, ])


# Random Forest ??zellik seti ile 4 model

# (1) Do??rusal Regresyon: Random Forest ??zellik seti
lm_model_rf <- train(Target ~ ., data = btc_rf[trainIndex, ], method = "glm")
lm_pred_rf <- predict(lm_model_rf, btc_rf[-trainIndex, ])

# (2) KNN Modeli: Random Forest ??zellik seti
knn_model_rf <- train(Target ~ ., data = btc_rf[trainIndex, ], method = "knn", tuneLength = 10)
knn_pred_rf <- predict(knn_model_rf, btc_rf[-trainIndex, ])

# (3) Karar A??a??lar?? (Random Forest): Random Forest ??zellik seti
rf_model_rf <- randomForest(Target ~ ., data = btc_rf[trainIndex, ])
rf_pred_rf <- predict(rf_model_rf, btc_rf[-trainIndex, ])

# (4) Destek Vekt??r Makinesi (SVM): Random Forest ??zellik seti
svm_model_rf <- train(Target ~ ., data = btc_rf[trainIndex, ], method = "svmRadial")
svm_pred_rf <- predict(svm_model_rf, btc_rf[-trainIndex, ])


# ---- 5. Kar??????kl??k Matrisi ve De??erlendirme ----

# Kar??????kl??k Matrisi ve Performans De??erlendirmesi

# Boruta ??zellik Seti:
print("Boruta ??zellik Seti: Do??rusal Regresyon Sonu??lar??")
lm_cm_boruta <- confusionMatrix(lm_pred_boruta, btc_boruta[-trainIndex, ]$Target)
print(lm_cm_boruta)

print("Boruta ??zellik Seti: KNN Sonu??lar??")
knn_cm_boruta <- confusionMatrix(knn_pred_boruta, btc_boruta[-trainIndex, ]$Target)
print(knn_cm_boruta)

print("Boruta ??zellik Seti: Random Forest Sonu??lar??")
rf_cm_boruta <- confusionMatrix(rf_pred_boruta, btc_boruta[-trainIndex, ]$Target)
print(rf_cm_boruta)

print("Boruta ??zellik Seti: SVM Sonu??lar??")
svm_cm_boruta <- confusionMatrix(svm_pred_boruta, btc_boruta[-trainIndex, ]$Target)
print(svm_cm_boruta)

# RFE ??zellik Seti:
print("RFE ??zellik Seti: Do??rusal Regresyon Sonu??lar??")
lm_cm_rfe <- confusionMatrix(lm_pred_rfe, btc_rfe[-trainIndex, ]$Target)
print(lm_cm_rfe)

print("RFE ??zellik Seti: KNN Sonu??lar??")
knn_cm_rfe <- confusionMatrix(knn_pred_rfe, btc_rfe[-trainIndex, ]$Target)
print(knn_cm_rfe)

print("RFE ??zellik Seti: Random Forest Sonu??lar??")
rf_cm_rfe <- confusionMatrix(rf_pred_rfe, btc_rfe[-trainIndex, ]$Target)
print(rf_cm_rfe)

print("RFE ??zellik Seti: SVM Sonu??lar??")
svm_cm_rfe <- confusionMatrix(svm_pred_rfe, btc_rfe[-trainIndex, ]$Target)
print(svm_cm_rfe)

# Random Forest ??zellik Seti:
print("Random Forest ??zellik Seti: Do??rusal Regresyon Sonu??lar??")
lm_cm_rf <- confusionMatrix(lm_pred_rf, btc_rf[-trainIndex, ]$Target)
print(lm_cm_rf)

print("Random Forest ??zellik Seti: KNN Sonu??lar??")
knn_cm_rf <- confusionMatrix(knn_pred_rf, btc_rf[-trainIndex, ]$Target)
print(knn_cm_rf)

print("Random Forest ??zellik Seti: Random Forest Sonu??lar??")
rf_cm_rf <- confusionMatrix(rf_pred_rf, btc_rf[-trainIndex, ]$Target)
print(rf_cm_rf)

print("Random Forest ??zellik Seti: SVM Sonu??lar??")
svm_cm_rf <- confusionMatrix(svm_pred_rf, btc_rf[-trainIndex, ]$Target)
print(svm_cm_rf)

