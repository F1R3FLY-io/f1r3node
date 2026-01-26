// See rholang/src/main/scala/coop/rchain/rholang/interpreter/RhoType.scala

use models::rhoapi::g_unforgeable::UnfInstance;
use models::rhoapi::ETuple;
use models::rhoapi::GPrivate;
use models::rhoapi::GSysAuthToken;
use models::rhoapi::GUnforgeable;
use models::rhoapi::{expr::ExprInstance, Expr, GDeployerId, GDeployId, Par};
use models::rust::rholang::implicits::{single_expr, single_unforgeable};
use rspace_plus_plus::rspace::history::Either;

pub struct RhoNil;

impl RhoNil {
    pub fn unapply(p: &Par) -> bool {
        p.is_nil()
    }

    pub fn create_par() -> Par {
        Par::default()
    }
}

pub struct RhoByteArray;

impl RhoByteArray {
    pub fn unapply(p: &Par) -> Option<Vec<u8>> {
        if let Some(expr) = single_expr(p) {
            if let Expr {
                expr_instance: Some(ExprInstance::GByteArray(bs)),
            } = expr
            {
                return Some(bs);
            }
        }
        None
    }

    pub fn create_par(bytes: Vec<u8>) -> Par {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::GByteArray(bytes)),
        }])
    }
}

pub struct RhoString;

impl RhoString {
    pub fn unapply(p: &Par) -> Option<String> {
        if let Some(expr) = single_expr(p) {
            if let Expr {
                expr_instance: Some(ExprInstance::GString(str)),
            } = expr
            {
                return Some(str);
            }
        }
        None
    }

    pub fn create_par(s: String) -> Par {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::GString(s)),
        }])
    }
}

pub struct RhoBoolean;

impl RhoBoolean {
    pub fn create_expr(b: bool) -> Expr {
        Expr {
            expr_instance: Some(ExprInstance::GBool(b)),
        }
    }

    pub fn unapply(p: &Par) -> Option<bool> {
        if let Some(expr) = single_expr(p) {
            if let Expr {
                expr_instance: Some(ExprInstance::GBool(b)),
            } = expr
            {
                return Some(b);
            }
        }
        None
    }
}

pub struct RhoNumber;

impl RhoNumber {
    pub fn create_expr(i: i64) -> Expr {
        Expr {
            expr_instance: Some(ExprInstance::GInt(i)),
        }
    }

    pub fn create_par(i: i64) -> Par {
        Par::default().with_exprs(vec![RhoNumber::create_expr(i)])
    }

    pub fn unapply(p: &Par) -> Option<i64> {
        if let Some(expr) = single_expr(&p) {
            if let Expr {
                expr_instance: Some(ExprInstance::GInt(v)),
            } = expr
            {
                return Some(v);
            }
        }
        None
    }
}

pub struct RhoTuple2;

impl RhoTuple2 {
    pub fn create_par(tuple: (Par, Par)) -> Par {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::ETupleBody(ETuple {
                ps: vec![tuple.0, tuple.1],
                locally_free: Vec::new(),
                connective_used: false,
            })),
        }])
    }

    pub fn unapply(p: Par) -> Option<(Par, Par)> {
        if let Some(expr) = single_expr(&p) {
            if let Expr {
                expr_instance: Some(ExprInstance::ETupleBody(ETuple { ps, .. })),
            } = expr
            {
                if ps.len() == 2 {
                    return Some((ps[0].clone(), ps[1].clone()));
                } else {
                    return None;
                }
            }
        }
        None
    }
}

pub struct RhoUri;

impl RhoUri {
    pub fn create_par(s: String) -> Par {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::GUri(s)),
        }])
    }

    pub fn unapply(p: &Par) -> Option<String> {
        if let Some(expr) = single_expr(&p) {
            if let Expr {
                expr_instance: Some(ExprInstance::GUri(s)),
            } = expr
            {
                return Some(s);
            }
        }
        None
    }
}

pub struct RhoDeployerId;

impl RhoDeployerId {
    pub fn create_par(bytes: Vec<u8>) -> Par {
        Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GDeployerIdBody(GDeployerId {
                public_key: bytes,
            })),
        }])
    }

    pub fn unapply(p: &Par) -> Option<Vec<u8>> {
        if let Some(expr) = single_unforgeable(&p) {
            if let GUnforgeable {
                unf_instance: Some(UnfInstance::GDeployerIdBody(id)),
            } = expr
            {
                return Some(id.public_key);
            }
        }
        None
    }
}

pub struct RhoDeployId;

impl RhoDeployId {
    pub fn create_par(bytes: Vec<u8>) -> Par {
        Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GDeployIdBody(GDeployId {
                sig: bytes,
            })),
        }])
    }

    pub fn unapply(p: &Par) -> Option<Vec<u8>> {
        if let Some(expr) = single_unforgeable(&p) {
            if let GUnforgeable {
                unf_instance: Some(UnfInstance::GDeployIdBody(id)),
            } = expr
            {
                return Some(id.sig);
            }
        }
        None
    }
}

pub struct RhoName;

impl RhoName {
    pub fn create_par(gprivate: GPrivate) -> Par {
        Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GPrivateBody(gprivate)),
        }])
    }

    pub fn unapply(p: &Par) -> Option<GPrivate> {
        if let Some(expr) = single_unforgeable(&p) {
            if let GUnforgeable {
                unf_instance: Some(UnfInstance::GPrivateBody(gprivate)),
            } = expr
            {
                return Some(gprivate);
            }
        }
        None
    }
}

pub struct RhoExpression;

impl RhoExpression {
    pub fn create_par(expr: Expr) -> Par {
        Par::default().with_exprs(vec![expr])
    }

    pub fn unapply(p: &Par) -> Option<Expr> {
        single_expr(p)
    }
}

pub struct RhoUnforgeable;

impl RhoUnforgeable {
    pub fn create_par(unforgeable: GUnforgeable) -> Par {
        Par::default().with_unforgeables(vec![unforgeable])
    }

    pub fn unapply(p: &Par) -> Option<GUnforgeable> {
        single_unforgeable(p)
    }
}

pub struct RhoSysAuthToken;

impl RhoSysAuthToken {
    pub fn create_par(token: GSysAuthToken) -> Par {
        Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GSysAuthTokenBody(token)),
        }])
    }

    pub fn unapply(p: &Par) -> Option<GSysAuthToken> {
        if let Some(expr) = single_unforgeable(&p) {
            if let GUnforgeable {
                unf_instance: Some(UnfInstance::GSysAuthTokenBody(token)),
            } = expr
            {
                return Some(token);
            }
        }
        None
    }
}

pub trait Extractor<RhoType> {
    type RustType;

    fn unapply(p: &Par) -> Option<Self::RustType>;
}

impl Extractor<RhoBoolean> for RhoBoolean {
    type RustType = bool;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        RhoBoolean::unapply(p)
    }
}

impl Extractor<RhoString> for RhoString {
    type RustType = String;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        RhoString::unapply(p)
    }
}

impl Extractor<RhoNil> for RhoNil {
    type RustType = ();

    fn unapply(p: &Par) -> Option<Self::RustType> {
        if RhoNil::unapply(p) {
            Some(())
        } else {
            None
        }
    }
}

impl Extractor<RhoByteArray> for RhoByteArray {
    type RustType = Vec<u8>;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        RhoByteArray::unapply(p)
    }
}

impl Extractor<RhoDeployerId> for RhoDeployerId {
    type RustType = Vec<u8>;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        RhoDeployerId::unapply(p)
    }
}

impl Extractor<RhoDeployId> for RhoDeployId {
    type RustType = Vec<u8>;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        RhoDeployId::unapply(p)
    }
}

impl Extractor<RhoName> for RhoName {
    type RustType = GPrivate;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        RhoName::unapply(p)
    }
}

impl Extractor<RhoNumber> for RhoNumber {
    type RustType = i64;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        RhoNumber::unapply(p)
    }
}

impl Extractor<RhoUri> for RhoUri {
    type RustType = String;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        RhoUri::unapply(p)
    }
}

impl Extractor<RhoUnforgeable> for RhoUnforgeable {
    type RustType = GUnforgeable;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        RhoUnforgeable::unapply(p)
    }
}

impl Extractor<RhoExpression> for RhoExpression {
    type RustType = Expr;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        RhoExpression::unapply(p)
    }
}

impl Extractor<RhoSysAuthToken> for RhoSysAuthToken {
    type RustType = GSysAuthToken;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        RhoSysAuthToken::unapply(p)
    }
}

impl<A, B> Extractor<(A, B)> for (A, B)
where
    A: Extractor<A>,
    B: Extractor<B>,
{
    type RustType = (A::RustType, B::RustType);

    fn unapply(p: &Par) -> Option<Self::RustType> {
        if let Some((p1, p2)) = RhoTuple2::unapply(p.clone()) {
            if let (Some(a), Some(b)) = (A::unapply(&p1), B::unapply(&p2)) {
                return Some((a, b));
            }
        }
        None
    }
}

impl<A, B> Extractor<Either<A, B>> for Either<A, B>
where
    A: Extractor<A>,
    B: Extractor<B>,
{
    type RustType = Either<A::RustType, B::RustType>;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        if let Some(b) = B::unapply(p) {
            Some(Either::Right(b))
        } else if let Some(a) = A::unapply(p) {
            Some(Either::Left(a))
        } else {
            None
        }
    }
}
