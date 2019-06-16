library(stringi)
library(stringr)
source("line_corect.R")
library(tictoc)
tic("final_time")

Max_val <- 49999
df1 <- read.csv("pre_crt_preout_data.csv",header = FALSE,sep = ",", quote = "\"")[1]
typeof(df1[[1]])
length(df1[[1]])
#write.csv("After correction",file="post_crt_preout_data.csv",col.names = FALSE, append=FALSE )
close(file("post_crt_preout_data2.csv",open = 'w'))
for(i in 2:100){
  temp <- toString(df1[[1]][i])
  write.table(correct_line(temp),file="post_crt_preout_data2.csv",col.names = FALSE, append = TRUE )
}

print("pra")
toc()