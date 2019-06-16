source("word_corect2.R")

library(stringi)
library(stringr)
library(stringdist)
# for caliculating time
library(tictoc)

tic("pra_time")
"
correct_word(\"howw\")[1]
correct_word(\"ar\")[1]
"
correct_line <- function(inp_string){
  pre_words <- strsplit(inp_string,"[^a-z]+")[[1]]
  post_words <- character(0)
  for(i in pre_words){
    temp <- correct_word(i)
    post_words <- paste(post_words,temp[1])
  }
  
  post_words
}
#correct_line("pedigre pouch")


toc()