#' @title R for Deep Forest Model (gcForest)
#' @description gcforest() base on a  Python Deep Forest application programming interface (API). Reference \url{https://github.com/pylablanche/gcForest}.
#'
#' @name gcforest
#' @aliases gcforest
#' @author Xu Jing
#' @usage gcforest(shape_1X=NA, n_mgsRFtree=30L, window=NA, stride=1L,
#'     cascade_test_size=0.2, n_cascadeRF=2L, n_cascadeRFtree=101L,
#'     cascade_layer=Inf,min_samples_mgs=0.1, min_samples_cascade=0.05,
#'     tolerance=0.0)
#'
#' @param shape_1X int or tuple list or np.array (default=None)Shape of a single sample element [n_lines, n_cols]. Required when calling mg_scanning!For sequence data a single int can be given.
#' @param n_mgsRFtree int (default=30) Number of trees in a Random Forest during Multi Grain Scanning.
#' @param window int (default=None)List of window sizes to use during Multi Grain Scanning. If 'None' no slicing will be done.
#' @param stride int (default=1)Step used when slicing the data.
#' @param cascade_test_size float or int (default=0.2) Split fraction or absolute number for cascade training set splitting.
#' @param n_cascadeRF int (default=2)Number of Random Forests in a cascade layer. For each pseudo Random Forest a complete Random Forest is created, hence the total numbe of Random Forests in a layer will be 2*n_cascadeRF.
#' @param n_cascadeRFtree int (default=101) Number of trees in a single Random Forest in a cascade layer.
#' @param min_samples_mgs float or int (default=0.1) Minimum number of samples in a node to perform a split during the training of Multi-Grain Scanning Random Forest. If int number_of_samples = int. If float, min_samples represents the fraction of the initial n_samples to consider.
#' @param min_samples_cascade float or int (default=0.1) Minimum number of samples in a node to perform a split during the training of Cascade Random Forest. If int number_of_samples = int. If float, min_samples represents the fraction of the initial n_samples to consider.
#' @param cascade_layer int (default=np.inf) mMximum number of cascade layers allowed. Useful to limit the contruction of the cascade.
#' @param tolerance float (default=0.0) Accuracy tolerance for the casacade growth. If the improvement in accuracy is not better than the tolerance the construction is stopped.
#'
#' @details gcForest provides several important function interfaces, just like the style of Python sklearn.
#' \enumerate{
#'     \item \strong{fit(X,y)} Training the gcForest on input data X and associated target y;
#'     \item \strong{predict(X)} Predict the class of unknown samples X;
#'     \item \strong{predict_proba(X)} Predict the class probabilities of unknown samples X;
#'     \item \strong{mg_scanning(X, y=None)} Performs a Multi Grain Scanning on input data;
#'     \item \strong{window_slicing_pred_prob(X, window, shape_1X, y=None)} Performs a window slicing of the input data and send them through Random Forests. If target values 'y' are provided sliced data are then used to train the Random Forests;
#'     \item \strong{cascade_forest(X, y=None)} Perform (or train if 'y' is not None) a cascade forest estimator;
#'     }
#'
#' @import reticulate
#' @import pkgdown
#' @import crayon
#' @import cli
#' @import utils
#'
#' @examples
#' \dontrun{
#'
#' sk <- NULL
#'
#' .onLoad <- function(libname, pkgname) {
#'     sk <<- reticulate::import("sklearn", delay_load = TRUE)
#'   }
#'
#' train_test_split <- sk$model_selection$train_test_split
#'
#' data <- sk$datasets$load_iris
#' iris <- data()
#' X = iris$data
#' y = iris$target
#' data_split = train_test_split(X, y, test_size=0.33)
#'
#' X_tr <- data_split[[1]]
#' X_te <- data_split[[2]]
#' y_tr <- data_split[[3]]
#' y_te <- data_split[[4]]
#'
#' gcforest_m <- gcforest(shape_1X=4L, window=2L, tolerance=0.0)
#'
#' gcforest_m$fit(X_tr, y_tr)
#'
#' pred_X = gcforest_m$predict(X_te)
#' print(pred_X)
#' }
#'
#' @export gcforest
gcforest <- function(shape_1X=NA, n_mgsRFtree=30L, window=NA, stride=1L,
                     cascade_test_size=0.2, n_cascadeRF=2L, n_cascadeRFtree=101L, cascade_layer=Inf,
                     min_samples_mgs=0.1, min_samples_cascade=0.05, tolerance=0.0){

  # reticulate::source_python("exec/GCForest.py")
  # environment(gcForest) <- globalenv()
  gcForest <- NULL
  py_path <- system.file("python", "GCForest.py", package = "gcForest")
  reticulate::source_python(py_path,envir=globalenv())

  message(paste0(crayon::green(cli::symbol$tick),crayon::blue(" Welcome to use gcForest packages! \n"),
                 paste0(crayon::green(cli::symbol$tick),crayon::blue(" read the docs form https://gituhb.com/DataXujing/gcForest_r"))))

  gcf <-  gcForest(shape_1X=shape_1X, n_mgsRFtree=n_mgsRFtree, window=window, stride=stride,
                   cascade_test_size=cascade_test_size, n_cascadeRF=n_cascadeRF,
                   n_cascadeRFtree=n_cascadeRFtree, cascade_layer=cascade_layer,
                   min_samples_mgs= min_samples_mgs, min_samples_cascade=min_samples_cascade,
                   tolerance=tolerance, n_jobs=1L)
  return(gcf)
}


#' @title R Data Transform to Python Data
#' @description  A function to tansform R data structure to Python data structure,
#' which based on the reticulate package.
#' @name gcdata
#' @aliases gcdata
#' @author Xu Jing
#' @usage gcdata(x)
#'
#' @param x  The R project like data.frame,vector, array etc..
#'
#' @import reticulate
#' @import pkgdown
#' @import crayon
#' @import cli
#'
#' @examples
#' \dontrun{
#' r_dat <- data.frame('x1'=c(1L,2L,3L),'x2'=c(2L,3L,4L))
#' py_dat <- gcdata(r_dat)
#' class(py_dat)
#'
#' r_vec <- c('a','b','c')
#' py_vec <- gcdata(r_vec)
#' class(py_vec)
#'
#' }
#'
#' @export gcdata
gcdata <- function(x){
  tryCatch(y <- reticulate::r_to_py(x),
           error = function(e){
             "Your input data is wrongful, place Check!"
           },
           finally = {return(y)})
}





#' @title gcForest Model Persistence Function
#' @description  It is a sklearn APIs to save your training model, and load it to predict, now you can
#' use R to callback. see also \code{\link{model_load}}
#' @name model_save
#' @aliases model_save
#' @author Xu Jing
#' @usage model_save(model,path)
#'
#' @param model The train model,like gcforest(see also \code{\link{gcforest}}).
#' @param path The path to save model.
#'
#' @import reticulate
#' @import pkgdown
#' @import crayon
#' @import cli
#'
#' @examples
#' \dontrun{
#' sk <- NULL
#'
#' .onLoad <- function(libname, pkgname) {
#'     sk <<- reticulate::import("sklearn", delay_load = TRUE)
#'   }
#'
#' train_test_split <- sk$model_selection$train_test_split
#'
#' data <- sk$datasets$load_iris
#' iris <- data()
#' X = iris$data
#' y = iris$target
#' data_split = train_test_split(X, y, test_size=0.33)
#'
#' X_tr <- data_split[[1]]
#' X_te <- data_split[[2]]
#' y_tr <- data_split[[3]]
#' y_te <- data_split[[4]]
#'
#' gcforest_m <- gcforest(shape_1X=4L, window=2L, tolerance=0.0)
#' gcforest_m$fit(X_tr, y_tr)
#' gcf_model <- model_save(gcforest_m,'gcforest_model.model')
#'
#' gcf <- model_load('gcforest_model.model')
#' gcf$predict(X_te)
#'
#' }
#'
#' @export model_save
model_save <- function(model,path){

  # skjoblib <- NULL
  #
  # .onLoad <- function(libname, pkgname) {
  #   skjoblib <<- reticulate::import("sklearn", delay_load = TRUE)
  # }
  skjoblib <- reticulate::import('sklearn')
  joblib_r <- skjoblib$externals$joblib

  cat(paste0(crayon::green(cli::symbol$tick)," Model has been saved in: ",path))
  return(joblib_r$dump(model,path))

}



#' @title gcForest Model Persistence Function
#' @description  It is a sklearn APIs to save your training model, and load it to predict, now you can
#' use R to callback. see also \code{\link{model_save}}
#' @name model_load
#' @aliases model_load
#' @author Xu Jing
#' @usage model_load(path)
#'
#' @param path The path to save model(see also \code{\link{model_save}}.
#'
#' @import reticulate
#' @import pkgdown
#' @import crayon
#' @import cli
#'
#' @examples
#' \dontrun{
#' sk <- NULL
#'
#' .onLoad <- function(libname, pkgname) {
#'     sk <<- reticulate::import("sklearn", delay_load = TRUE)
#'   }
#'
#' train_test_split <- sk$model_selection$train_test_split
#'
#' data <- sk$datasets$load_iris
#' iris <- data()
#' X = iris$data
#' y = iris$target
#' data_split = train_test_split(X, y, test_size=0.33)
#'
#' X_tr <- data_split[[1]]
#' X_te <- data_split[[2]]
#' y_tr <- data_split[[3]]
#' y_te <- data_split[[4]]
#'
#' gcforest_m <- gcforest(shape_1X=4L, window=2L, tolerance=0.0)
#' gcforest_m$fit(X_tr, y_tr)
#' gcf_model <- model_save(gcforest_m,'gcforest_model.model')
#'
#' gcf <- model_load('gcforest_model.model')
#' gcf$predict(X_te)
#'
#' }
#'
#' @export model_load
model_load <- function(path){
  # skjoblib <- NULL
  #
  # .onLoad <- function(libname, pkgname) {
  #   skjoblib <<- reticulate::import("sklearn", delay_load = TRUE)
  # }
  skjoblib <- reticulate::import('sklearn')
  joblib_r <- skjoblib$externals$joblib

  cat(paste0(crayon::green(cli::symbol$tick)," Model has been load from: ",path))
  return(joblib_r$load(path))
}




