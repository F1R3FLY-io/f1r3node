// See casper/src/main/scala/coop/rchain/casper/genesis/contracts/VaultsGenerator.scala

use super::vault::Vault;

pub struct VaultsGenerator {
    pub supply: i64,
    pub code: String,
}

impl VaultsGenerator {
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
            .map(|v| format!("(\"{}\", {})", v.vault_address.to_base58(), v.initial_balance))
            .collect::<Vec<String>>()
            .join(", ");

        let continue_clause = if !is_last_batch {
            "| initContinue!()"
        } else {
            ""
        };

        let code = format!(
            r#" 
            new rl(`rho:registry:lookup`), systemVaultCh in {{
              rl!(`rho:system:systemVault`, *systemVaultCh) |
              for (@(_, SystemVault) <- systemVaultCh) {{
                new systemVaultInitCh in {{
                  @SystemVault!("init", *systemVaultInitCh) |
                  for (TreeHashMap, @vaultMap, initVault, initContinue <- systemVaultInitCh) {{
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
