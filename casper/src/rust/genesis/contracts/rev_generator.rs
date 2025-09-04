// See casper/src/main/scala/coop/rchain/casper/genesis/contracts/RevGenerator.scala

use super::vault::Vault;

pub struct RevGenerator {
    pub supply: i64,
    pub code: String,
}

impl RevGenerator {
    pub fn new(supply: i64, code: String) -> Self {
        Self { supply, code }
    }

    pub fn create_from_user_vaults(
        user_vaults: Vec<Vault>,
        supply: i64,
        is_last_batch: bool,
    ) -> Self {
        let vault_balance_list = user_vaults
            .iter()
            .map(|v| format!("(\"{}\", {})", v.rev_address.to_base58(), v.initial_balance))
            .collect::<Vec<String>>()
            .join(", ");

        let continue_clause = if !is_last_batch {
            "| initContinue!()"
        } else {
            ""
        };

        let code = format!(
            r#" 
            new rl(`rho:registry:lookup`), revVaultCh in {{
              rl!(`rho:rchain:revVault`, *revVaultCh) |
              for (@(_, RevVault) <- revVaultCh) {{
                new revVaultInitCh in {{
                  @RevVault!("init", *revVaultInitCh) |
                  for (TreeHashMap, @vaultMap, initVault, initContinue <- revVaultInitCh) {{
                    match [{}] {{
                      vaults => {{
                        new iter in {{
                          contract iter(@[(addr, initialBalance) ... tail]) = {{
                          iter!(tail) |
                          new vault, setDoneCh in {{
                            initVault!(*vault, addr, initialBalance) |
                            TreeHashMap!("set", vaultMap, addr, *vault, *setDoneCh) |
                            for (_ <- setDoneCh) {{ Nil }}
                          }}
                      }} |
                      iter!(vaults) {}
                    }}
                  }}
                }}
              }}
            }}
          }}
        }}
      "#,
            vault_balance_list, continue_clause
        );

        Self::new(supply, code)
    }
}
