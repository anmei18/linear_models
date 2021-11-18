linear models
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
library(p8105.datasets)

set.seed(1)
```

Load NYC airbnb

``` r
data("nyc_airbnb")

nyc_airbnb = 
  nyc_airbnb %>% 
  mutate(stars = review_scores_location / 2) %>% 
  rename(
    borough = neighbourhood_group
  ) %>% 
  filter(borough != "Staten Island") %>% 
  select(price,stars,borough,neighbourhood,room_type)
```

visualization…

``` r
nyc_airbnb %>% 
  ggplot(aes(x = stars, y = price)) +
  geom_point()
```

    ## Warning: Removed 9962 rows containing missing values (geom_point).

![](linear_models_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

Lets fit a linear model …

``` r
fit = lm(price ~ stars +borough, data = nyc_airbnb)
```

Let’s look at this …

``` r
summary(fit)
```

    ## 
    ## Call:
    ## lm(formula = price ~ stars + borough, data = nyc_airbnb)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -169.8  -64.0  -29.0   20.2 9870.0 
    ## 
    ## Coefficients:
    ##                  Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)       -70.414     14.021  -5.022 5.14e-07 ***
    ## stars              31.990      2.527  12.657  < 2e-16 ***
    ## boroughBrooklyn    40.500      8.559   4.732 2.23e-06 ***
    ## boroughManhattan   90.254      8.567  10.534  < 2e-16 ***
    ## boroughQueens      13.206      9.065   1.457    0.145    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 181.5 on 30525 degrees of freedom
    ##   (9962 observations deleted due to missingness)
    ## Multiple R-squared:  0.03423,    Adjusted R-squared:  0.03411 
    ## F-statistic: 270.5 on 4 and 30525 DF,  p-value: < 2.2e-16

``` r
summary(fit)$coef
```

    ##                   Estimate Std. Error   t value     Pr(>|t|)
    ## (Intercept)      -70.41446  14.020697 -5.022180 5.137589e-07
    ## stars             31.98989   2.527500 12.656733 1.269392e-36
    ## boroughBrooklyn   40.50030   8.558724  4.732049 2.232595e-06
    ## boroughManhattan  90.25393   8.567490 10.534465 6.638618e-26
    ## boroughQueens     13.20617   9.064879  1.456850 1.451682e-01

If you want to present output …

``` r
fit %>% 
  broom::tidy() %>% 
  mutate(term = str_replace(term, "borough", "Borough: ")) %>% 
  select(term,estimate,p.value) %>% 
  knitr::kable(digits = 3)
```

| term               | estimate | p.value |
|:-------------------|---------:|--------:|
| (Intercept)        |  -70.414 |   0.000 |
| stars              |   31.990 |   0.000 |
| Borough: Brooklyn  |   40.500 |   0.000 |
| Borough: Manhattan |   90.254 |   0.000 |
| Borough: Queens    |   13.206 |   0.145 |

## Diagnostics

``` r
modelr::add_residuals(nyc_airbnb, fit) %>% 
  ggplot(aes(x = stars, y = resid)) +
  geom_point()
```

    ## Warning: Removed 9962 rows containing missing values (geom_point).

![](linear_models_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

``` r
modelr::add_residuals(nyc_airbnb, fit) %>% 
  ggplot(aes(x = resid)) +
  geom_density() +
  xlim(-200,200)
```

    ## Warning: Removed 11208 rows containing non-finite values (stat_density).

![](linear_models_files/figure-gfm/unnamed-chunk-7-2.png)<!-- -->

## interactions? Nesting?

Lets try a different model …

``` r
fit = lm(price ~ stars * borough + room_type, data = nyc_airbnb)

broom::tidy(fit)
```

    ## # A tibble: 10 × 5
    ##    term                   estimate std.error statistic  p.value
    ##    <chr>                     <dbl>     <dbl>     <dbl>    <dbl>
    ##  1 (Intercept)              128.       74.4     1.72   8.54e- 2
    ##  2 stars                      4.21     16.6     0.253  8.00e- 1
    ##  3 boroughBrooklyn          -46.6      76.1    -0.612  5.40e- 1
    ##  4 boroughManhattan         -52.6      76.8    -0.686  4.93e- 1
    ##  5 boroughQueens             -1.50     82.3    -0.0182 9.85e- 1
    ##  6 room_typePrivate room   -105.        2.05  -51.1    0       
    ##  7 room_typeShared room    -130.        6.15  -21.1    7.15e-98
    ##  8 stars:boroughBrooklyn     15.7      17.0     0.924  3.56e- 1
    ##  9 stars:boroughManhattan    25.4      17.1     1.49   1.37e- 1
    ## 10 stars:boroughQueens        2.74     18.3     0.150  8.81e- 1

Let’s try nesting …

``` r
nyc_airbnb %>% 
  relocate(borough) %>% 
  nest(data = price:room_type) %>% 
  mutate(
    lm_fits = map(.x = data,~lm(price ~ stars + room_type, data = .x)),
    lm_results = map(lm_fits, broom::tidy)
  ) %>% 
  select(borough,lm_results) %>% 
  unnest(lm_results) %>% 
  filter(term == "stars")
```

    ## # A tibble: 4 × 6
    ##   borough   term  estimate std.error statistic  p.value
    ##   <chr>     <chr>    <dbl>     <dbl>     <dbl>    <dbl>
    ## 1 Bronx     stars     4.45      3.35      1.33 1.85e- 1
    ## 2 Queens    stars     9.65      5.45      1.77 7.65e- 2
    ## 3 Brooklyn  stars    21.0       2.98      7.05 1.90e-12
    ## 4 Manhattan stars    27.1       4.59      5.91 3.45e- 9

Look at neighborhoods in Manhattan…

``` r
manhattan_lm_results_df = 
  nyc_airbnb %>% 
  filter(borough == "Manhattan") %>% 
  select(-borough) %>% 
  relocate(neighbourhood) %>% 
  nest(data = price:room_type) %>% 
  mutate(
    lm_fits = map(data, ~lm(price ~ stars + room_type, data = .x)),
    lm_results = map(lm_fits, broom::tidy)
  ) %>% 
  select(neighbourhood, lm_results) %>% 
  unnest(lm_results)

manhattan_lm_results_df %>% 
  filter(term == "stars") %>% 
  ggplot(aes(x = estimate)) +
  geom_density()
```

![](linear_models_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
manhattan_lm_results_df %>% 
  filter(str_detect(term, "room_type")) %>% 
  ggplot(aes(x = neighbourhood, y = estimate)) +
  geom_point() +
  facet_grid(~term) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
```

![](linear_models_files/figure-gfm/unnamed-chunk-10-2.png)<!-- -->

## Logistic regression

``` r
nyc_airbnb = 
  nyc_airbnb %>% 
  mutate(
    expensive_apt = as.numeric(price > 500)
  ) 
```

Let’s fit a logistic regression for the binary outcome.

``` r
logistic_fit = 
  glm(
    expensive_apt ~ stars + borough, 
    data = nyc_airbnb, 
    family = binomial()
    )

logistic_fit %>% 
  broom::tidy() %>% 
  mutate(
    term = str_replace(term, "borough", "Borough: "),
    estimate = exp(estimate)
  ) %>% 
  select(term, OR = estimate, p.value)
```

    ## # A tibble: 5 × 3
    ##   term                     OR    p.value
    ##   <chr>                 <dbl>      <dbl>
    ## 1 (Intercept)        7.52e-10 0.908     
    ## 2 stars              2.15e+ 0 0.00000292
    ## 3 Borough: Brooklyn  2.49e+ 5 0.945     
    ## 4 Borough: Manhattan 8.11e+ 5 0.940     
    ## 5 Borough: Queens    1.15e+ 5 0.949

``` r
nyc_airbnb %>% 
  modelr::add_predictions(logistic_fit) %>% 
  mutate(pred = boot::inv.logit(pred))
```

    ## # A tibble: 40,492 × 7
    ##    price stars borough neighbourhood room_type       expensive_apt          pred
    ##    <dbl> <dbl> <chr>   <chr>         <chr>                   <dbl>         <dbl>
    ##  1    99   5   Bronx   City Island   Private room                0  0.0000000343
    ##  2   200  NA   Bronx   City Island   Private room                0 NA           
    ##  3   300  NA   Bronx   City Island   Entire home/apt             0 NA           
    ##  4   125   5   Bronx   City Island   Entire home/apt             0  0.0000000343
    ##  5    69   5   Bronx   City Island   Private room                0  0.0000000343
    ##  6   125   5   Bronx   City Island   Entire home/apt             0  0.0000000343
    ##  7    85   5   Bronx   City Island   Entire home/apt             0  0.0000000343
    ##  8    39   4.5 Bronx   Allerton      Private room                0  0.0000000234
    ##  9    95   5   Bronx   Allerton      Entire home/apt             0  0.0000000343
    ## 10   125   4.5 Bronx   Allerton      Entire home/apt             0  0.0000000234
    ## # … with 40,482 more rows
