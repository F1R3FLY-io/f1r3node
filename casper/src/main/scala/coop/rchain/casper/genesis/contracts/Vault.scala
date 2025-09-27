package coop.rchain.casper.genesis.contracts
import coop.rchain.rholang.interpreter.util.ASIAddress

final case class Vault(asiAddress: ASIAddress, initialBalance: Long)
