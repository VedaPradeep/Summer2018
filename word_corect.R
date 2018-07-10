#source("word_freq_gen.R")
library(tictoc)

tic("it_time")
MAX_VAL <- 29156


#sorted_words[adist(word,sorted_words) <= min(adist(word,sorted_words),2)]
#adist(x, y = NULL, costs = NULL, counts = FALSE, fixed = TRUE,partial = !fixed, ignore.case = FALSE, useBytes = FALSE)


final_words <- read.table("sorted_words_data3.csv",header=FALSE, sep="\n",quote = "\"'",colClasses = "character")[[1]]



correct_word <- function(word){
  "
   min_val <- 3
   min_val <- function(){
     for(j in final_words){ 
       if(adist(word,final_words[j]) < min_val ) {min_val <- adist(word,final_words[j])}
     }
   }
   print(min_val)
"
  #print("test 1")
   min_val <-3
   for(j in final_words){
     if(adist(word,j) < min_val) min_val = adist(word,j)
   }
   print(min_val)
   output <- c()
   for (i in 2:MAX_VAL){
     if(adist(word,final_words[i]) == min_val){
       output <- c(output,final_words[i])
       break
     }
   }
   if(length(output) != 0) { output}
   else c(word)
}

correct_word("oug")
#tic("read_time")
#correct_word("puppi")
#correct_word("thaat")
#correct_word("prozect")
"
toc()

typeof(final_words)
length(final_words)
"
toc()
"
a <- adist(\"piese\",sorted_words[2])
typeof(a)
a

b <- min(adist(\"piese\",sorted_words))
typeof(b)
b
"