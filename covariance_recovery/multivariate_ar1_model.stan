data {
  int<lower=0> N;         //number of years in timeseries
  int<lower=0> K;         //number of timeseries
  matrix[K, N] tsx;       //matrix of the timeseries
  vector[N] p_var;        //variable with potential to affect timeseries values
}


parameters {
  vector[K] alpha;             //tiemseries intercepts
  vector[K] beta;              //effects of variable on timeseries
  vector<lower=0>[K] sigma;    //standard deviation of timeseries
  cholesky_factor_corr[K] L;   //lower triangular of a positive definite correlation matrix
  real<lower=-1,upper=1> phi;  //temporal autocorrelation parameter
}

transformed parameters {
  matrix[K, N] mu;
  mu[1,1] = 0;
  mu[2,1] = 0;
  for (i in 2:N){
    mu[,i] = tsx[,i-1];
  }
}


model {
  tsx[,1] ~ multi_normal_cholesky(beta * p_var[1] + alpha + phi * mu[,1], diag_pre_multiply(sigma, L));
  for(i in 2:N){
    tsx[,i] ~ multi_normal_cholesky(beta * p_var[i] + alpha + phi * mu[,i], diag_pre_multiply(sigma, L));  //multivariate model
  }
  
  alpha ~ normal(6, 3);        //prior on intercepts
  beta ~ normal(0, 10);        //prior on slopes
  sigma ~ cauchy(0, 10);       //SD of timeseries
  L ~ lkj_corr_cholesky(1);    //prior for lower triangular
}

generated quantities {
  matrix[K,N] nuY;
  matrix[K,K] Omega;
  matrix[K,K] Sigma;
  Omega = multiply_lower_tri_self_transpose(L);
  Sigma = quad_form_diag(Omega, sigma);
  nuY[,1] = multi_normal_cholesky_rng(beta * p_var[1] + alpha + phi * mu[,1], diag_pre_multiply(sigma, L));
  for(i in 2:N){
    nuY[,i] = multi_normal_cholesky_rng(beta * p_var[i] + alpha + phi * mu[,i], diag_pre_multiply(sigma, L));
  }
}

