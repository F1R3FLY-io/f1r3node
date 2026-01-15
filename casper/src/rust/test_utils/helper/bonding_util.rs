// See casper/src/test/scala/coop/rchain/casper/helper/BondingUtil.scala
// Moved from casper/tests/helper/bonding_util.rs to casper/src/rust/test_utils/helper/bonding_util.rs
// All imports fixed for library crate context

use crate::rust::errors::CasperError;
use crate::rust::util::construct_deploy;
use crypto::rust::private_key::PrivateKey;
use crypto::rust::signatures::signed::Signed;
use models::rust::casper::protocol::casper_message::DeployData;

/// Creates a bonding deploy
/// Scala equivalent: BondingUtil.bondingDeploy[F]
/// 
/// Note: In original Scala code, the 'amount' parameter is accepted but not used!
/// The hardcoded value 1000 is used instead (line 23 in BondingUtil.scala).
/// This is likely a bug, but we port it 1:1 for now.
pub fn bonding_deploy(
    amount: i64,
    private_key: &PrivateKey,
    shard_id: Option<String>,
) -> Result<Signed<DeployData>, CasperError> {
    // WARNING: Scala bug - 'amount' parameter is ignored, hardcoded 1000 is used
    let _ = amount; // Explicitly mark as unused to match Scala behavior
    
    let source = r#"
new retCh, PoSCh, rl(`rho:registry:lookup`), stdout(`rho:io:stdout`), deployerId(`rho:rchain:deployerId`) in {
  rl!(`rho:rchain:pos`, *PoSCh) |
  for(@(_, PoS) <- PoSCh) {
    @PoS!("bond", *deployerId, 1000, *retCh)
  }
}
"#.to_string();

construct_deploy::source_deploy_now_full(
  source,
  None,
  None,
  Some(private_key.clone()),
  None,
  shard_id,
)
}


