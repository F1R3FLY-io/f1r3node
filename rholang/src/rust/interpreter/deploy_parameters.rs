// See rholang/src/main/scala/coop/rchain/rholang/interpreter/DeployParameters.scala

use models::rhoapi::Par;

pub struct DeployParameters {
    pub user_id: Par,
}

impl DeployParameters {
    pub fn empty() -> Self {
        Self {
            user_id: Par::default(),
        }
    }
}
