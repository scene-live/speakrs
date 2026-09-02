#[cfg(not(any(
    feature = "default-linalg",
    feature = "intel-mkl",
    feature = "openblas-static",
    feature = "openblas-system"
)))]
compile_error!(
    "speakrs requires a BLAS backend; enable default features or choose exactly one of `intel-mkl`, `openblas-static`, or `openblas-system`"
);

#[cfg(any(
    all(
        feature = "default-linalg",
        any(feature = "intel-mkl", feature = "openblas-static")
    ),
    all(feature = "intel-mkl", feature = "openblas-static"),
    all(feature = "openblas-static", feature = "openblas-system")
))]
compile_error!(
    "speakrs supports only one BLAS backend; disable default features before enabling `intel-mkl`, `openblas-static`, or `openblas-system`"
);

#[cfg(all(feature = "intel-mkl", not(target_arch = "x86_64")))]
compile_error!("the `intel-mkl` feature is only supported on x86_64 targets");

#[cfg(feature = "intel-mkl")]
pub(crate) use ndarray_linalg_mkl::{Eigh, Inverse, UPLO, error::LinalgError};

#[cfg(feature = "openblas-static")]
pub(crate) use ndarray_linalg_static::{Eigh, Inverse, UPLO, error::LinalgError};

// bq-Meeting PoC ローカル改変: default-linalg/openblas-system は runner の target 別
// dependency 解決で同時に feature 化され得るが、実 target では同じ alias を指す。
#[cfg(any(
    feature = "openblas-system",
    all(feature = "default-linalg", not(any(feature = "intel-mkl", feature = "openblas-static")))
))]
pub(crate) use ndarray_linalg_default::{Eigh, Inverse, UPLO, error::LinalgError};
