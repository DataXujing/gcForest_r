 # gcForest  <a href="https://github.com/DataXujing/gcForest_r/"><img src="vignettes/logo.png" align="right" alt="logo" height="120" width="240" /></a>

Xu Jing

## 1.Introduction

The gcForest algorithm was suggested in Zhou and Feng 2017 (<https://arxiv.org/abs/1702.08835> , refer for this paper for technical details) and pylablanche(<https://github.com/pylablanche>) who provide a Python3.X implementation of this algorithm on github (<https://github.com/pylablanche/gcForest>). We provide a R package called gcForest which is the R interface of the  pylablanche's gcForest module (Python3.X). And if you want to known more about gcForest, please read the source paper (Deep Forest).

## 2.Prerequisites

As mentioned in the previous paragraph, we provide a R interface of the Python3.X module gcForest(by pylablanche), If you want to use this R packages like the R's tensorflow and keras you should hava a Python3.X environment first, and then, you will need to have the following installed on your computer to make it work:

+ Python 3.X
+ Numpy >= 1.12.0
+ Scikit-learn >= 0.18.1

after these, you can install this R package like:

+ `install.packages('gcForest')`

+ `devtools::install_github('DataXujing/gcForest_r')`

run `library(gcForest)` in your R Console no mistakes.


## 3.Using gcForest

Supported APIs:

+ `fit(X,y)` Training the gcForest on input data X and associated target y;
+ `predict(X)` Predict the class of unknown samples X;
+ `predict_proba(X)` Predict the class probabilities of unknown samples X;
+ `mg_scanning(X, y=None)` Performs a Multi Grain Scanning on input data;
+ `window_slicing_pred_prob(X, window, shape_1X, y=None)` Performs a window slicing of the input data and send them through Random Forests. If target values 'y' are provided sliced data are then used to train the Random Forests;
+ `cascade_forest(X, y=None)` Perform (or train if 'y' is not None) a cascade forest estimator;
+ `gcdata(x)` Tansform R data structure to Python data structure;
+ `model_save(model,path)` To save trained model in disk;
+ `model_load(path)` To load trained model from disk to R environment;

Example1: iris data set

```r

library(gcForest)

sk <- reticulate::import('sklearn')
train_test_split <- sk$model_selection$train_test_split

data <- sk$datasets$load_iris
iris <- data()
X = iris$data
y = iris$target
data_split = train_test_split(X, y, test_size=0.33)

X_tr <- data_split[[1]]
X_te <- data_split[[2]]
y_tr <- data_split[[3]]
y_te <- data_split[[4]]

gcforest_m <- gcforest(shape_1X=4L, window=2L, tolerance=0.0)
gcforest_m$fit(X_tr,y_tr)
gcf_model <- model_save(gcforest_m,'../gcforest_model.model')

gcf <- model_load('../gcforest_model.model')
gcf$fit(X_tr, y_tr)

```

Example2: Digits data set

```r

library(gcForest)

sk <- sk <- reticulate::import('sklearn')
train_test_split <- sk$model_selection$train_test_split

data <- sk$datasets$load_digits
digits <- data()
X = digits$data
y = digits$target
data_split = train_test_split(X, y, test_size=0.4)

gcforest_m <- gcforest(shape_1X=c(8L,8L), window=c(4L,6L), tolerance=0.0, min_samples_mgs=10L, min_samples_cascade=7L)
gcforest_m$fit(X_tr,y_tr)
gcf_model <- model_save(gcforest_m,'../gcforest_model.model')

gcf <- model_load('../gcforest_model.model')
gcf$fit(X_tr, y_tr)
```

Example3: Using mg-scanning and cascade_forest Sperately

```r
# mg-scanning
gcforest_m <- gcForest(shape_1X=c(8L,8L), window=5L, min_samples_mgs=10L, min_samples_cascade=7L)
X_tr_mgs <- gcforest_m$mg_scanning(X_tr, y_tr)

X_te_mgs <- gcforest_m$mg_scanning(X_te)

# cascade_forest
gcforest_m <- gcForest(tolerance=0.0, min_samples_mgs=10L, min_samples_cascade=7L)
cf <- gcforest_m$cascade_forest(X_tr_mgs, y_tr)

pred_proba <- gcforest_m$cascade_forest(X_te_mgs)
pred_proba <- reticulate::py_to_r(pred_proba)

# then do mean and max
```

Example4: Skipping mg_scanning

```r
gcforest_m <- gcForest(tolerance=0.0, min_samples_cascade=20L)
cf <- gcforest_m$cascade_forest(X_tr, y_tr)
pred_proba <- gcforest_m$cascade_forest(X_te)
pred_proba <- reticulate::py_to_r(pred_proba)

# then do mean and max

```

## 4.Notes

Thanks for the paper of Deep Forest( Zhou and Feng 2017 (<https://arxiv.org/abs/1702.08835>)) and the author of the [gcForest Python3.X moulde](https://github.com/pylablanche/gcForest) (pylablanche <https://github.com/pylablanche>). And We constantly improve gcForest R package, and even consider putting official modules which provide by  [LAMDA(Learning And Mining from DatA)](http://lamda.nju.edu.cn/code_gcForest.ashx?AspxAutoDetectCookieSupport=1)in gcForest R package.