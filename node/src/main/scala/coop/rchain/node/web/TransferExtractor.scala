package coop.rchain.node.web

import coop.rchain.casper.protocol.{ReportCommProto, ReportProduceProto, SingleReport, TransferInfo}
import coop.rchain.models.Par

/**
  * Utility for extracting native REV transfer information from deploy execution reports.
  *
  * This extracts transfer details by parsing the execution events and looking for
  * RevVault transfer patterns on the transferUnforgeable channel.
  */
object TransferExtractor {

  /**
    * Extract transfer info from a single deploy execution report.
    *
    * @param report The SingleReport from deploy execution
    * @param transferUnforgeable The unforgeable channel used by RevVault for transfers
    * @return List of transfers found in this report
    */
  def extractTransfers(
      report: SingleReport,
      transferUnforgeable: Par
  ): List[TransferInfo] = {

    // Extract raw transfer data from COMM events on the transfer channel
    val rawTransfers = report.events.iterator
      .map(_.report.value)
      .collect {
        case ReportCommProto(consume, produces) =>
          consume.channels.headOption.map((_, produces))
      }
      .flatten
      .collect {
        case (channel, produces) if channel == transferUnforgeable =>
          produces.headOption
      }
      .flatten
      .map { produce =>
        val pars           = produce.data.pars
        val fromAddr       = pars.head.exprs.head.getGString
        val toAddr         = pars(2).exprs.head.getGString
        val amount         = pars(3).exprs.head.getGInt
        val retUnforgeable = pars(5)
        (fromAddr, toAddr, amount, retUnforgeable)
      }
      .toList

    val retUnforgeables = rawTransfers.map(_._4).toSet

    // Find success/failure status for each transfer by looking for produces on return channels
    val resultMap = report.events.iterator
      .map(_.report.value)
      .collect {
        case ReportProduceProto(channel, data) if retUnforgeables.contains(channel) =>
          val tuple   = data.pars.head.exprs.head.getETupleBody.ps
          val success = tuple.head.exprs.head.getGBool
          val failReason =
            if (success) ""
            else tuple(1).exprs.head.getGString
          (channel, (success, failReason))
      }
      .toMap

    // Build TransferInfo objects with success/failure status
    rawTransfers.map {
      case (from, to, amount, retUnforg) =>
        val (success, failReason) = resultMap.getOrElse(retUnforg, (true, ""))
        TransferInfo(
          fromAddr = from,
          toAddr = to,
          amount = amount,
          success = success,
          failReason = failReason
        )
    }
  }

  /**
    * Extract transfers from user deploy reports.
    *
    * A normal deploy execution produces 3 reports: PreCharge, UserDeploy, Refund.
    * User-initiated transfers are in the UserDeploy report (index 1).
    *
    * @param reports The sequence of SingleReports from deploy execution
    * @param transferUnforgeable The unforgeable channel used by RevVault for transfers
    * @return List of user-initiated transfers (excludes system PreCharge/Refund transfers)
    */
  def extractUserTransfers(
      reports: Seq[SingleReport],
      transferUnforgeable: Par
  ): List[TransferInfo] =
    reports.length match {
      case 3 => extractTransfers(reports(1), transferUnforgeable) // UserDeploy is middle report
      case 2 => List.empty // PreCharge + Refund only, user deploy had no transfers logged
      case 1 => List.empty // PreCharge fail only
      case _ => List.empty
    }
}
