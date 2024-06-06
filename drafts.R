require(ggplot2)
require(reshape2)
require(gridExtra)
setwd('/Users/karina/Google Drive/github/SlimShady/')


make_plot <- function(data, plot_subtitle){
  
  data_plot <- data
  data_plot$generation <- seq(1, nrow(data_plot))
  names(data_plot) <- c('loop', 'parallel_1', 'parallel_2', 'generation')
  
  data_melted <- melt(data_plot, id.vars = 'generation')
  
  ggplot(data_melted, aes(x = generation, y = value, color = variable)) +
    geom_line() +
    labs(title = 'Population Evalution Time',
         subtitle = plot_subtitle,
         x = 'Generation',
         y = 'Value',
         color = 'Method') +
    ylim(0, 1) +
    theme_minimal()

}


data <- read.csv('execution_times_pop_1_job_1.csv', header = FALSE)
p_1_1 <- make_plot(data, '1 job - run 1')

data <- read.csv('execution_times_pop_1_job_2.csv', header = FALSE)
p_1_2 <- make_plot(data, '1 job - run 2')

data <- read.csv('execution_times_pop_1_job_3.csv', header = FALSE)
p_1_3 <- make_plot(data, '1 job - run 3')

data <- read.csv('execution_times_pop_3_jobs.csv', header = FALSE)
p_2 <- make_plot(data, '3 jobs')

data <- read.csv('execution_times_pop_5_jobs.csv', header = FALSE)
p_3 <- make_plot(data, '5 jobs')

data <- read.csv('execution_times_pop_-1_jobs.csv', header = FALSE)
p_4 <- make_plot(data, '-1 jobs')

grid.arrange(p_1_1, p_1_2, p_1_3, 
             p_2, p_3, p_4, ncol=3)




