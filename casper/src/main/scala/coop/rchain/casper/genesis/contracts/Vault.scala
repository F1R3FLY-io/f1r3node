package coop.rchain.casper.genesis.contracts
import coop.rchain.rholang.interpreter.util.VaultAddress

final case class Vault(vaultAddress: VaultAddress, initialBalance: Long)
