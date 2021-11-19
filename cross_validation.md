Cross validation
================
AnMei Chen
11/16/2021

``` r
library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.1 ──

    ## ✓ ggplot2 3.3.5     ✓ purrr   0.3.4
    ## ✓ tibble  3.1.5     ✓ dplyr   1.0.7
    ## ✓ tidyr   1.1.3     ✓ stringr 1.4.0
    ## ✓ readr   2.0.1     ✓ forcats 0.5.1

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
library(modelr)
library(mgcv)
```

    ## Loading required package: nlme

    ## 
    ## Attaching package: 'nlme'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     collapse

    ## This is mgcv 1.8-38. For overview type 'help("mgcv-package")'.

``` r
knitr::opts_chunk$set(
  fig.width = 6,
  fig.asp = .6,
  out.width = "90%"
)

options(
  ggplot2.continuous.colour = "viridis" , 
  ggplot2.continuous.fill = "viridis"
)

scale_colour_discrete = scale_colour_viridis_d
scale_fill_discrete = scale_fill_viridis_d
```

## simulate data

``` r
nonlin_df = 
  tibble(
    id = 1:100,
    x = runif(100, 0, 1),
    y = 1 - 10 * (x - .3) ^ 2 + rnorm(100, 0, .3)
  )
```

look at the data

``` r
nonlin_df %>% 
  ggplot(aes(x = x , y = y)) +
  geom_point()
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-3-1.png" width="90%" />

## cross validation – by hand

get training and testing datasets

training df sampled 80 of samples from the nonlin\_df. testing df gets
20 samples from the nonlin\_df that is not included in the train\_df

``` r
train_df = sample_n(nonlin_df, size = 80)
test_df = anti_join(nonlin_df, train_df, by = "id")
```

Fit three models.

``` r
linear_mod = lm(y ~ x, data = train_df)
smooth_mod = gam( y ~ s(x), data = train_df)
wiggly_mod = gam( y ~ s(x, k = 30), sp = 10e-6, data = train_df)
```

can I see what I just did…

``` r
train_df %>% 
  add_predictions(linear_mod) %>% 
  ggplot(aes(x = x, y = y)) +
  geom_point() + 
  geom_line(aes( y = pred), color = "red")
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-6-1.png" width="90%" />

``` r
train_df %>% 
  add_predictions(smooth_mod) %>% 
  ggplot(aes(x = x, y = y)) +
  geom_point() + 
  geom_line(aes( y = pred), color = "red")
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-6-2.png" width="90%" />

``` r
train_df %>% 
  add_predictions(wiggly_mod) %>% 
  ggplot(aes(x = x, y = y)) +
  geom_point() + 
  geom_line(aes( y = pred), color = "red")
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-6-3.png" width="90%" />

smooth model is the best in terms of accuracy.

use gather\_predictions() and facet\_grid to compare models
simultaneously

``` r
train_df %>% 
  gather_predictions(linear_mod,smooth_mod, wiggly_mod) %>% 
  ggplot(aes(x = x, y = y)) +
  geom_point() + 
  geom_line(aes( y = pred), color = "red") +
  facet_grid(. ~ model)
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-7-1.png" width="90%" />

Look at prediction accuracy.

``` r
rmse(linear_mod, test_df)
```

    ## [1] 0.9161202

``` r
rmse(smooth_mod, test_df)
```

    ## [1] 0.3575102

``` r
rmse(wiggly_mod, test_df)
```

    ## [1] 0.3894079

## cross validation using `modelr`

``` r
cv_df = 
  crossv_mc(nonlin_df, 100)
```

what is happening here …

``` r
cv_df %>% 
  pull(train) %>% 
  .[[1]] %>% 
  as_tibble()
```

    ## # A tibble: 79 × 3
    ##       id      x       y
    ##    <int>  <dbl>   <dbl>
    ##  1     1 0.321   1.30  
    ##  2     4 0.613  -0.283 
    ##  3     5 0.572  -0.115 
    ##  4     7 0.203   1.52  
    ##  5     8 0.280   1.18  
    ##  6    10 0.875  -2.66  
    ##  7    11 0.0840  0.947 
    ##  8    12 0.486   0.815 
    ##  9    13 0.580   0.0553
    ## 10    14 0.665  -0.592 
    ## # … with 69 more rows

``` r
cv_df %>% 
  pull(test) %>% 
  .[[1]] %>% 
  as_tibble()
```

    ## # A tibble: 21 × 3
    ##       id      x      y
    ##    <int>  <dbl>  <dbl>
    ##  1     2 0.0380  0.707
    ##  2     3 0.652  -0.122
    ##  3     6 0.446   0.610
    ##  4     9 0.420   0.893
    ##  5    16 0.269   0.640
    ##  6    25 0.678  -0.633
    ##  7    30 0.612   0.489
    ##  8    36 0.642   0.893
    ##  9    39 0.0470  0.248
    ## 10    43 0.360   0.526
    ## # … with 11 more rows

``` r
cv_df = 
  cv_df %>% 
  mutate(
    train = map(train, as_tibble),
    test = map(test,as_tibble)
  )
```

Let’s try to fit models and get RMSEs for them.

Map across “train” column and fit y against x with linear models into
each elements in train column. use .x as a placeholder of train.

``` r
cv_df = 
  cv_df %>% 
  mutate(
    linear_mod = map(.x = train, ~lm( y ~ x, data = .x)),
    smooth_mod = map(.x = train, ~gam( y ~ s(x), data = .x)),
    wiggly_mod = map(.x = train, ~gam( y ~ s(x, k = 30), sp = 10e-6, data = .x))
  ) %>% 
  mutate(
    rmse_linear = map2_dbl(.x = linear_mod, .y = test, ~rmse(model = .x, data = .y)),
    rmse_smooth = map2_dbl(.x = smooth_mod, .y = test, ~rmse(model = .x, data = .y)),
    rmse_wiggly = map2_dbl(.x = wiggly_mod, .y = test, ~rmse(model = .x, data = .y))
  )
```

what dod these results say about model choices?

``` r
cv_df %>% 
  select(starts_with("rmse")) %>% 
  pivot_longer(
    everything(),
    names_to = "model",
    values_to = "rmse",
    names_prefix = "rmse_"
  ) %>% 
  ggplot(aes(x = model, y = rmse)) +
  geom_violin()
```

<img src="cross_validation_files/figure-gfm/unnamed-chunk-12-1.png" width="90%" />

This is the distributions across 100 training and testing splits fitting
in the linear, smooth, wiggly models. From this plot we can see that the
linear model makes the worst prediction. The smooth is doing better than
the wiggly.

Compute averages …

``` r
cv_df %>% 
  select(starts_with("rmse")) %>% 
  pivot_longer(
    everything(),
    names_to = "model",
    values_to = "rmse",
    names_prefix = "rmse_"
  ) %>% 
  group_by(model) %>% 
  summarize((avg_smse = mean(rmse)))
```

    ## # A tibble: 3 × 2
    ##   model  `(avg_smse = mean(rmse))`
    ##   <chr>                      <dbl>
    ## 1 linear                     0.764
    ## 2 smooth                     0.329
    ## 3 wiggly                     0.385
