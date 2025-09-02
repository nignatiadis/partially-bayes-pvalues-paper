# Reading in chicken data
library(tidyverse)
chicks <- read.table("chickens.dat", header = TRUE)
J <- nrow(chicks)
x <- chicks$freq
y0 <- chicks$sham_est - 1
se0 <- chicks$sham_se
n0 <- chicks$sham_n
y1 <- chicks$exposed_est - 1
se1 <- chicks$exposed_se
n1 <- chicks$exposed_n
diff <- y1 - y0
diff_se <- sqrt(se1 ^ 2 + se0 ^ 2)

sigma_sq_res <- (n1*(n1-1)*se1^2 + n0*(n0-1)*se0^2) / (n1+n0-2)
diff_se2 <- sqrt( sigma_sq_res)* ( 1/sqrt(n0) + 1/sqrt(n1))  

#pvalue_redo <- 2 * pt(-abs(diff) / diff_se2, n1 + n0 - 2)
#pvalue_orig <- 2 * pt(-abs(diff) / diff_se, n1 + n0 - 2)

#chicks$pvalue_redo <- round(pvalue_redo, digits=3) 
#chicks$pvalue_orig <- round(pvalue_orig, digits=3) 

pvalue_sham <-2 * pnorm(-abs(diff) / diff_se)
pvalue_exp_only <- 2 * pnorm(-abs(y1) / se1)

init_tibble <- tibble(oracle_pvalue=pvalue_exp_only, pvalue=pvalue_sham, type="sham")




## Get ready to fit Bayesian models
library("rstan")
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)


y <- c(y0, y1)
se <- c(se0, se1)
expt_id <- rep(1:J, 2)
z <- c(rep(0, J), rep(1, J))  # treatment:  0 = sham, 1 = exposed
data <- list(
  N = 2 * J,
  J = J,
  y = y,
  se = se,
  x = x,
  expt_id = expt_id,
  z = z,
  diff = diff,
  diff_se = diff_se
)

data_short <- list(
  N = J,
  y = y0,
  se = se0
)

## Fit partially Bayes model

fit_partial <-
  stan(
    "partially_bayes.stan",
    data = data_short,
    control = list(adapt_delta = 0.9),
    refresh = 0
  )
print(fit_partial)

rstan::extract(fit_partial)$sigma_b


bs <- rstan::extract(fit_partial)$b


pval_mat <- matrix(NA, J, 4000)
pval_mat_prime <- matrix(NA, J, 4000)

for (j in seq_len(J)){
  pval_mat[j,] <- 1- pnorm(abs(y1[j]), mean = bs[,j],  sd=se1[j]) + pnorm(-abs(y1[j]), mean = bs[,j],  sd=se1[j] )
  pval_mat_prime[j,] <- 1- pchisq(abs(y1[j])^2/se1[j]^2, df=1, ncp = bs[,j]^2/se1[j]^2) 
} 

# sanity check that two p-value calculation methods are almost identical
max(abs(pval_mat - pval_mat_prime))




library(tidyverse)

# Boxplot for b_i
bs_df <- as.data.frame(bs)
colnames(bs_df) <- paste0("V", 1:38)

bs_long <- bs_df %>% 
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  mutate(
    var_num = as.numeric(gsub("V", "", variable)),
    variable = factor(var_num, levels = 1:38, labels = as.character(1:38))
  )

sigma_df <- data.frame(
  var_num = 1:38,
  variable = factor(1:38, levels = 1:38, labels = as.character(1:38)),
  lower = -se1,
  upper = se1
)

ggplot(bs_long, aes(x = value, y = variable)) +
  geom_boxplot(fill = "lightblue", alpha = 0.7, outlier.size = 0.5) +
  geom_segment(data = sigma_df, 
               aes(x = lower, xend = lower,
                   y = as.numeric(variable) - 0.55,
                   yend = as.numeric(variable) + 0.55,
                   color = "sigma"),
               alpha=0.62,
               linewidth = 1.5, show.legend = TRUE) +
  geom_segment(data = sigma_df, 
               aes(x = upper, xend = upper,
                   y = as.numeric(variable) - 0.55,
                   yend = as.numeric(variable) + 0.55,
                   color = "sigma"),
               alpha = 0.62,
               linewidth = 1.5, show.legend = FALSE) +
  scale_color_manual(values = c("sigma" = "darkgreen"), 
                     labels = c("sigma" = expression(paste("±", sigma[i1]))),
                     name = "") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 6),
        legend.position = "bottom") +
  labs(x = expression(b[i]), y = "Experiment") 

ggsave("bs_boxplots.pdf", width = 4, height = 6, units = "in")


# Boxplot for p-values
pval_df <- as.data.frame(pval_mat)
colnames(pval_df) <- paste0("V", 1:ncol(pval_mat))

pval_long <- pval_df %>% 
  mutate(experiment = 1:n()) %>%
  pivot_longer(cols = -experiment, names_to = "variable", values_to = "value") %>%
  mutate(experiment = factor(experiment, levels = 1:38))

avg_pval <- rowMeans(pval_mat)

markers_df <- data.frame(
  experiment = factor(1:38, levels = 1:38),
  avg_pval = avg_pval,
  pvalue_exp_only = pvalue_exp_only,
  pvalue_sham = pvalue_sham
)

ggplot(pval_long, aes(x = value, y = experiment)) +
  geom_boxplot(fill = "lightblue", alpha = 0.7, outlier.size = 0.5) +
  geom_segment(data = markers_df, 
               aes(x = avg_pval, xend = avg_pval,
                   y = as.numeric(experiment) - 0.55,
                   yend = as.numeric(experiment) + 0.55,
                   color = "avg_pval"),
               linewidth = 1.5,
               alpha=0.62) +
  geom_segment(data = markers_df, 
               aes(x = pvalue_exp_only, xend = pvalue_exp_only,
                   y = as.numeric(experiment) - 0.55,
                   yend = as.numeric(experiment) + 0.55,
                   color = "exp_only"),
               linewidth = 1.5,
               alpha=0.62) +
  geom_segment(data = markers_df, 
               aes(x = pvalue_sham, xend = pvalue_sham,
                   y = as.numeric(experiment) - 0.55,
                   yend = as.numeric(experiment) + 0.55,
                   color = "sham"),
               linewidth = 1.5,
               alpha=0.62) +
  scale_x_log10() +
  scale_color_manual(
    values = c("avg_pval" = "darkgreen", 
               "exp_only" = "blue", 
               "sham" = "red"),
    labels = c("avg_pval" = "Partially Bayes p-value",
               "exp_only" = "Exp only",
               "sham" = "Sham"),
    name = "") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 6),
        legend.position = "bottom") +
  labs(x = "p-value (log scale)", y = "Experiment")


ggsave("pvalue_boxplots.pdf", width = 4, height = 6, units = "in")
