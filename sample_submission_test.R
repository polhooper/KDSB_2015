#..quick and dirty little R script to mockup a sample test submission file: 
num_id <- rep(701:1140, each = 2)
char_id <- rep(c('_Diastole', '_Systole'), length(num_id)/2) 
Id <- paste0(num_id, char_id)

zeros <- as.data.frame(matrix(0, length(Id), 600))
names(zeros) <- paste0('P', 0:599)

sample_submission_test <- data.frame(Id, zeros)
write.csv(sample_submission_test, 'data/sample_submission_test.csv', row.names = FALSE)
