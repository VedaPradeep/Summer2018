# Code for unmerging the given set of merged words 

library(tictoc)
library(rvest)
library(stringdist)
library(stringi)
library(stringr)
library(csv)
library(xml2)

MAX_VAL <- 29000

#test_words <- read.table("post_crt_preout_data2.csv",header=FALSE, sep="\n",quote = "\"'",colClasses = "character")[[1]]
tic("t_div")
final_words <- read.table("sorted_words_data4.csv",header=FALSE, sep="\n",quote = "\"'",colClasses = "character")[[1]]

correct_word <- function(word){
  #min_freq <- 2
  
  ls_word <- c()
  for(j in 2:MAX_VAL){
    #print(j)
    #print(final_words[j])
    temp <- grep(final_words[j],word) 
    #print(typeof(temp))
    if(all.equal(temp,1)==TRUE){
      #print(j)
      ls_word <- c(ls_word,final_words[j]) 
    }
  }  
 
   for(i in ls_word) print(i)
  return(ls_word)
}
print("pradeep")
temp1 <- grep("search","searchxmasdogchristmasoutfit" )
print(temp1)
ls_word <- correct_word("searchxmasdogchristmasoutfit")

#Sorting found words in decreasing order of length 
ls_len <- c()
for(i in 0:(length(ls_word)-1)){
  typeof()
  ls_len <- c(ls_len,length(ls_word[i]))
}
#for(i in ls_len) print(i)
df <- data.frame(2,ls_word,ls_len)
table_words <- table(df)
sort_words <- names(sort(table_words,decreasing = TRUE))
for(i in sort_words) print(i)


toc()

