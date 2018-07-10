#correcting all sentences of given csv file

library(stringi)
library(stringr)
library(stringdist)
library(tictoc) #library for measuring time 

tic("final_time")
# words and frequiences r stored in sorted_words_data3.csv
final_words <- read.table("sorted_words_data3.csv",header=FALSE, sep="\n",quote = "\"'",colClasses = "character")[[1]]

MAX_VAL <- length(final_words)

# word corrector function 
correct_word <- function(word){
  
  #to find min of all adist
  min_val <-3
  for(j in final_words){
    adist_wd <- adist(word,j)
    if(adist_wd == 0) return(j)
    else if(adist_wd < min_val) min_val = adist_wd
  }
  #print(min_val)
  
  #geting all words which r of min adist
  output <- c()
  for (i in 2:MAX_VAL){
    if(adist(word,final_words[i]) == min_val){
      output <- c(output,final_words[i])
      break #we r breaking after reaching min value
    }
  }
  if(length(output) != 0) { output}
  else c(word)
}

# correcting all words in  the given line
correct_line <- function(inp_string){
  pre_words <- strsplit(inp_string,"[^a-z]+")[[1]]
  post_words <- character(0)
  for(i in pre_words){
    temp <- correct_word(i)
    post_words <- paste(post_words,temp[1])
  }
  post_words
}
toc()