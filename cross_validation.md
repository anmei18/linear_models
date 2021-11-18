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

    ## [1] 0.7279099

``` r
rmse(smooth_mod, test_df)
```

    ## [1] 0.3168397

``` r
rmse(wiggly_mod, test_df)
```

    ## [1] 0.4324704
