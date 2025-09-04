pub mod history;
pub mod history_reader;
pub mod history_repository;
pub mod history_action;
pub mod instances;
pub mod radix_tree;
pub mod roots_store;
pub mod root_repository;
pub mod cold_store;
pub mod history_repository_impl;

// PartialEq needed here for testing purposes
#[derive(Debug, PartialEq, Clone)]
pub enum Either<L, R> {
    Left(L),
    Right(R),
}

impl<L, R> Either<L, R> {
    pub fn map<F, T>(self, f: F) -> Either<L, T>
    where
        F: FnOnce(R) -> T,
    {
        match self {
            Either::Left(l) => Either::Left(l),
            Either::Right(r) => Either::Right(f(r)),
        }
    }

    pub fn and_then<F, T>(self, f: F) -> Either<L, T>
    where
        F: FnOnce(R) -> Either<L, T>,
    {
        match self {
            Either::Left(l) => Either::Left(l),
            Either::Right(r) => f(r),
        }
    }

    pub fn left_then<F>(self, f: F) -> Either<L, R>
    where
        F: FnOnce(L) -> Either<L, R>,
    {
        match self {
            Either::Left(l) => f(l),
            Either::Right(r) => Either::Right(r),
        }
    }
}
