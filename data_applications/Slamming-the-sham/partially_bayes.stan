data {
  int N;
  vector[N] y;
  vector[N] se;
}
parameters {
  real mu_b;
  real<lower=0> sigma_b;
  vector[N] eta_b;
}
transformed parameters {
  vector[N] b;
  b = mu_b + sigma_b * eta_b;
}
model {
  y ~ normal(b, se);
  eta_b ~ normal(0, 1);
}
