#' gcForest-package
#'
#' R application programming interface (API) for Deep Forest which based on Zhi-hua Zhou and Ji Feng. Deep Forest: Towards an Alternative to Deep Neural Networks.
#' In IJCAI-2017. (\url{https://arxiv.org/abs/1702.08835v2}) or Zhi-hua Zhou and Ji Feng. Deep Forest. In IJCAI-2017.(<https://arxiv.org/abs/1702.08835>),
#' and the Python application programming interface (API) (\url{https://github.com/pylablanche/gcForest})
#'
#' @name gcForest-package
#' @aliases gcForest-package
#' @docType package
#' @author Xu Jing
#'
#'
#' @seealso
#'
#' [1] Zhi-hua Zhou and Ji Feng. Deep Forest: Towards an Alternative to Deep Neural Networks.In IJCAI-2017. (\url{https://arxiv.org/abs/1702.08835v2})
#'
#' [2] Zhi-hua Zhou and Ji Feng. Deep Forest. In IJCAI-2017.(\url{https://arxiv.org/abs/1702.08835})
#'
#' [3] \url{https://github.com/pylablanche/gcForest}
#'
#'
#' @examples
#'
#' \dontrun{
#'
#' # ========= Model train=======
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
NULL
