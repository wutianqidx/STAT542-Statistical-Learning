library(text2vec)
library(slam)
library(crayon)
all = read.table("data.tsv",stringsAsFactors = F,header = T)
all$review = gsub('<.*?>', ' ', all$review)
splits = read.table("splits.csv", header = T)
s = 3  # Here we get the 3rd training/test split. 
train = all[-which(all$new_id%in%splits[,s]),]
test = all[which(all$new_id%in%splits[,s]),]

it = itoken(train$review, preprocess_function = tolower,
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
v = create_vocabulary(it, ngram = c(1L, 4L), 
                      stopwords = stop_words)
#remove very common and uncommon words
pruned_vocab = prune_vocabulary(v,
                                term_count_min = 5, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)

vectorizer = vocab_vectorizer(pruned_vocab)
dtm_train = create_dtm(it, vectorizer)
it_test = itoken(test$review, preprocess_function = tolower,
                 tokenizer = word_tokenizer)
dtm_test = create_dtm(it_test, vectorizer)

v.size = dim(dtm_train)[2]
ytrain = train$sentiment

summ = matrix(0, nrow=v.size, ncol=4)
summ[,1] = colapply_simple_triplet_matrix(as.simple_triplet_matrix(dtm_train[ytrain==1, ]),mean)
summ[,2] = colapply_simple_triplet_matrix(as.simple_triplet_matrix(dtm_train[ytrain==1, ]),var)
summ[,3] = colapply_simple_triplet_matrix(as.simple_triplet_matrix(dtm_train[ytrain==0, ]),mean)
summ[,4] = colapply_simple_triplet_matrix(as.simple_triplet_matrix(dtm_train[ytrain==0, ]),var)
n1=sum(ytrain); 
n=length(ytrain)
n0= n - n1

myp = (summ[,1] - summ[,3])/
  sqrt(summ[,2]/n1 + summ[,4]/n0)
words = colnames(dtm_train)
id = order(abs(myp), decreasing=TRUE)[1:2000]
pos.list = words[id[myp[id]>0]]
neg.list = words[id[myp[id]<0]]

#Visualizaiton
reviews = all$review[1:3]
for (x in reviews)
{
  reviews_split = unlist(strsplit(x," "))
  for (i in 1:length(reviews_split))
  {
    if (reviews_split[i] %in% pos.list)
    {
      reviews_split[i] = blue(reviews_split[i])
    }
    if (reviews_split[i] %in% neg.list)
    {
      reviews_split[i] = red(reviews_split[i])
    }
  }
  cat(reviews_split)
  cat("\n\n")
}


