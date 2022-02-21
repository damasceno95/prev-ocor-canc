
# PREVENDO OCORRÊNCIAS DE CâNCER EM DETERMINADO TIPO DE POPULAÇÃO

# MODELO DE CLASSIFICAÇÃO: PREVISÃO DE VALORES QUALITATIVOS

# DEFININDO O PROBLEMA DE NEGÓCIO A SER AVALIADO: PREVISÃO DE OCORRÊNCIA DE CANCER DE MAMA.
# UTILIZADO O REPOSITÓRIO DA UCI. DADOS VERÍDICOS.
# MODELO CLASSIFICATÓRIO SENDO UTILIZADO NO CAMPO MÉDICO.



# SETANDO O DIRETÓRIO DE TRABALHO

setwd(choose.dir())
getwd()

# ETAPA 01 - COLENTANDO OS DADOS

# DADOS CONTÉM INFORMAÇÕES SOBRE CÂNCER DE MAMA SENDO QUE:
# 569 SÃO AS OBSERVAÇÕES COLETADAS POR PACIENTES DENTRO DE 32 VARIÁVEIS DE MEDIDAS LABORATORIAIS.
# CLASSIFICAÇÃO DA DIAGNOSES: M = MALIGNO E B = BENIGNO. 

# CARREGAMENTO DOS DADOS

prevcancer <- read.csv('dataset.csv', stringsAsFactors = FALSE)
View(prevcancer)

# ANÁLISE EXPLORATÓRIA

summary(prevcancer)
str(prevcancer)

# ETAPA 2 - PRÉ - PROCESSAMENTO

# EXCLUSÃO DA COLUNA ID DO DATASET. O MODELO NÃO PRECISA DESTA INFORMAÇÃO. 
# O RESULTADO DESEJADO VISA A GENERALIZAÇÃO DOS DADOS, OU SEJA, BUSCAMOS APLICAÇÕES NÃO
# APENAS EM RELAÇÃO AS PESSOAS DESTE DATASET.

prevcancer$id = NULL
View(prevcancer)

# AJUSTANDO O TEXTO DENTRO DA TARGET VAR.

prevcancer$diagnosis = sapply(prevcancer$diagnosis, function(x){ifelse(x == "M","Maligno","Benigno")})
View(prevcancer)

table(prevcancer$diagnosis)

# TRANSFORMANDO OS DADOS EM FATOR.

prevcancer$diagnosis <- factor(prevcancer$diagnosis, levels = c('Maligno', 'Benigno'), labels = c('Maligno','Benigno'))
str(prevcancer)


# MODELO APRESENTADO ENCONTRA-SE METRIFICADO EM ESCALAS DIFERENTES E FORA DE NORMALIZAÇÃO
# FUNÇÃO PARA NORMALIZAR OS DADOS

normalizar <- function(x){
  return ((x - min(x)) / (max(x) - min(x)))
}

# TESTANDO A FUNÇÃO NORMALIZAR ACIMA

normalizar(c(1,2,3,4,5))
normalizar(c(10,20,30,40,50))

# APLICANDO A NORMALIZAÇÃO DENTRO DO DATASET

prevcancer_norm <- as.data.frame(lapply(prevcancer[2:31],normalizar))
View(prevcancer_norm)


# ETAPA 3 - CRIANDO O MODELO COM KNN

# INSTALANDO E CARRENDO O PACOTE PARA UTILIZAR O KNN
install.packages('class')
library(class)

# SEPARANDO OS DADOS DE TREINO E DADOS DE TESTES (DADOS NÃO RANDÔMICOS)

prevcancer_treino <- prevcancer_norm[1:469,]
prevcancer_teste <- prevcancer_norm[470:569,]

# LABELS PARA OS DADOS DE TREINO E DE TESTE

prevcancer_treino_label <- prevcancer[1:469,1]
prevcancer_teste_label <- prevcancer[470:569,1]  
 

length(prevcancer_treino_label)
length(prevcancer_teste_label)

# CRIANDO O MODELO DE CLASSIFICAÇÃO KNN

modelo_v1 <- knn(train = prevcancer_treino, 
                 test = prevcancer_teste,
                 cl = prevcancer_treino_label, 
                 k = 21)

summary(modelo_v1)

# ETAPA 4 : AVALIANDO O MODELO 

library(gmodels)

# CRIANDO A CONFUSION MATRIX PARA VALIDAR A VERACIDADE DOS DADOS OBSERVADOS X PREVISTOS PELO MODELO.
# USANDO 100 OBSERVAÇÕES

CrossTable(x = prevcancer_teste_label, y = modelo_v1, prop.chisq = F)

# VERIFICANDO:
# VERDADEIROS POSITIVOS, VERDADEIROS FALSOS, FALSOS POSITIVOS, FALSOS NEGATIVOS

# OBSERVAÇÕES: 

# MALIGNO (PREVISTO) X MALIGNO (OBSERVADO) = 37 CASOS =  VERDADEIRO POSITIVO
# BENIGNO (PREVISTO) X MALIGNO (OBSERVADO) = 0 CASOS = FALSO NEGATIVO
# MALIGNO (PREVISTO) X BENIGNO (OBSERVADO) = 2 CASOS = FALSO POSITIVO
# BENIGNO  (PREVISTO) X BENIGNO (OBSERVADO) = 61 CASOS = VERDADEIRO NEGATIVO


# MODELO ACERTOU 98 EM 100 CASOS = TAXA DE 98%

# ETAPA 5 : OTMIZANDO A PERFORMANCE DO MODELO
# MODIFICANDO A TÉCNICA DE PRÉ-PROCESSAMENTO.
# MODIFICANDO O Z-SCORE.

dados_z <- as.data.frame(scale(prevcancer[-1]))
summary(dados_z$area_mean)


# CRIANO NOVOS DATASETS DE TREINO E DE TESTE

dados_treino <- dados_z[1:469,]
dados_teste <- dados_z[470:569,]

dados_treino_label <- prevcancer[1:469,1]
dados_teste_label <- prevcancer[470:569,1]

# RECLASSIFICANDO O MODELO

modelo_v2 <- knn(train = dados_treino,
                 test = dados_teste,
                 cl = dados_treino_label,
                 k = 21)

CrossTable(x = dados_teste_label, y = modelo_v2, prop.chisq = F)


# MODELO 2 APRESENTOU PERFORMANCE MENOR DO QUE O PRIMEIRO MODELO.

# ETAPA 5 - MODIFICANDO O ALGORITMO

# CONSTRUINDO MODELO COM SUPORT VECTOR MACHINE (SVM)

# DEFININDO A SEMENTE PARA REPRODUÇÃO DE DADOS

set.seed(40)

# PREPARAÇÃO DO DATASET

dados <- read.csv('dataset.csv')
dados$id = NULL
dados[,'index'] <- ifelse(runif(nrow(dados)) < 0.8,1,0)


# DIVISÃO DADOS TREINO E TESTE

trainset <- dados[dados$index == 1,]
testset <- dados[dados$index == 2,]


# OBTER INDICE DOS DADOS DE TREINO
traincolset <- grep('index',names(trainset))

# REMOVER O INDICE DOS DATASETS

trainset <- trainset[,-traincolset] 
testset <- trainset[,-traincolset]

# OBTER INDICE DO COLUNA DA VAR TARGET  

traincolset <- grep('diag', names(dados))

# CRIAR O MODELO 

library(e1071)

modelo_v3_smv <- svm(diagnosis ~ .,
                     data = trainset,
                     type = 'C-classification',
                     kernel = 'radial')

# PREVISÕES

#PREVISÃO COM OS DADOS DE TREINO
pred_set <- predict(modelo_v3_smv,trainset)

# PERCENTUAL DE PREVISÕES CORRETAS COM OS DADOS DE TREINO

mean(pred_set == trainset$diagnosis)

# PREVISÕES COM OS DADOS DE TESTE

prev_test <- predict(modelo_v3_smv, testset)

# VERIFICAR ACURÁCIA

mean(prev_test == testset$diagnosis)


# CONFUSION MATRIX

table(prev_test, testset$diagnosis)


# ETAPA 07 - CRIANDO OUTRO MODELO (RANDOM FOREST)
library(rpart)

model_v4_rf = rpart(diagnosis ~ ., data = trainset, control = rpart.control(cp = .0005))

# PREVISÕES DE TESTE

tree_pred = predict(model_v4_rf, testset, type = 'class')

# PRECENTUAL PREVISTO

mean(tree_pred == testset$diagnosis)

# CONFUSION MATRIX

table(tree_pred, testset$diagnosis)


