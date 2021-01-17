library(text2vec)
library(slam)
library(glmnet)
library(pROC)
all = read.table("data.tsv",stringsAsFactors = F,header = T)
splits = read.table("splits.csv", header = T)
s = 3  # Here we get the 3rd training/test split.
all$review = gsub('<.*?>', ' ', all$review)
train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]
#read the vocabulary
tmp = unlist(read.table('myVocab.txt',stringsAsFactors=F,header = T))
it_tmp = itoken(tmp,
                preprocess_function = tolower,
                tokenizer = word_tokenizer)
stop_words = c("i", "me", "my", "myself", 
               "we", "our", "ours", "ourselves", 
               "you", "your", "yours", 
               "their", "they", "his", "her", 
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were", 
               "him", "himself", "has", "have", 
               "it", "its", "of", "one", "for", 
               "the", "us", "this")
myvocab = create_vocabulary(it_tmp, ngram = c(1L, 4L), 
                            stopwords = stop_words)

# remove terms not in words[id]
tmp.term =  gsub(" ","_", tmp)
myvocab = myvocab[myvocab$term %in% tmp.term, ]
bigram_vectorizer = vocab_vectorizer(myvocab)

#create dtm_matrix for train and test
it_train = itoken(train$review, preprocess_function = tolower,
                  tokenizer = word_tokenizer)
dtm_train = create_dtm(it_train, bigram_vectorizer)
it_test = itoken(test$review, preprocess_function = tolower,
                 tokenizer = word_tokenizer)
dtm_test = create_dtm(it_test, bigram_vectorizer)

#model prediction
set.seed(1821)
NFOLDS = 10
mycv = cv.glmnet(x=dtm_train, y=train$sentiment, 
                 family='binomial',type.measure = "auc", 
                 nfolds = NFOLDS, alpha=0)
myfit = glmnet(x=dtm_train, y=train$sentiment, 
               lambda = mycv$lambda.min, family='binomial', alpha=0)
logit_pred = predict(myfit, dtm_test, type = "response")
result = cbind(test$new_id,logit_pred)

#write result
write.table(result, 'mysubmission.txt',row.names=F,col.names = c('new_id','prob'))
roc_obj = roc(test$sentiment, as.vector(logit_pred))
auc(roc_obj) 
