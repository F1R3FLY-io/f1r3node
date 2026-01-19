package coop.rchain.node.web

import coop.rchain.casper.protocol.{BlockInfo, DeployInfo, TransferInfo}

/**
  * Utility for enriching BlockInfo with native REV transfer data.
  *
  * This bridges the gap between the Transaction API (which extracts transfers)
  * and the Block API (which returns BlockInfo).
  */
object BlockInfoEnricher {

  /**
    * Maps a TransactionResponse to a lookup map of transfers by deployId.
    *
    * @param response The cached transaction response for a block
    * @return Map from deployId (signature hex) to list of transfers for that deploy
    */
  def mapTransactionsToTransfers(response: TransactionResponse): Map[String, List[TransferInfo]] =
    response.data
      .collect {
        case TransactionInfo(tx, UserDeploy(deployId)) =>
          deployId -> TransferInfo(
            fromAddr = tx.fromAddr,
            toAddr = tx.toAddr,
            amount = tx.amount,
            success = tx.failReason.isEmpty,
            failReason = tx.failReason.getOrElse("")
          )
      }
      .groupBy(_._1)
      .map { case (k, v) => k -> v.map(_._2) }

  /**
    * Enriches a BlockInfo by populating the transfers field in each DeployInfo.
    *
    * @param blockInfo The original BlockInfo from BlockAPI
    * @param transfersByDeploy Map from deployId to transfers (from mapTransactionsToTransfers)
    * @return BlockInfo with transfers populated in each deploy
    */
  def enrichBlockInfo(
      blockInfo: BlockInfo,
      transfersByDeploy: Map[String, List[TransferInfo]]
  ): BlockInfo = {
    val enrichedDeploys = blockInfo.deploys.map { deployInfo =>
      val transfers = transfersByDeploy.getOrElse(deployInfo.sig, Nil)
      deployInfo.copy(transfers = transfers)
    }
    blockInfo.copy(deploys = enrichedDeploys)
  }

  /**
    * Enriches a BlockInfo using a TransactionResponse directly.
    *
    * @param blockInfo The original BlockInfo from BlockAPI
    * @param transactionResponse The cached transaction response for this block
    * @return BlockInfo with transfers populated in each deploy
    */
  def enrichBlockInfo(
      blockInfo: BlockInfo,
      transactionResponse: TransactionResponse
  ): BlockInfo =
    enrichBlockInfo(blockInfo, mapTransactionsToTransfers(transactionResponse))
}
