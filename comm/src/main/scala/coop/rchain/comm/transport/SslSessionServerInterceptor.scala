package coop.rchain.comm.transport

import coop.rchain.comm.protocol.routing.{Header => RHeader, _}
import coop.rchain.comm.rp.ProtocolHelper
import coop.rchain.crypto.util.CertificateHelper
import coop.rchain.shared.{Log, LogSource}

import io.grpc._
import javax.net.ssl.SSLSession

/**
  * This wart exists because that's how gRPC works
  */
@SuppressWarnings(Array("org.wartremover.warts.Var"))
class SslSessionServerInterceptor(networkID: String) extends ServerInterceptor {

  def interceptCall[ReqT, RespT](
      call: ServerCall[ReqT, RespT],
      headers: Metadata,
      next: ServerCallHandler[ReqT, RespT]
  ): ServerCall.Listener[ReqT] = {
    val sslSession = Option(call.getAttributes.get(Grpc.TRANSPORT_ATTR_SSL_SESSION))
    log.debug(
      s"Intercepting gRPC call: ${call.getMethodDescriptor.getFullMethodName}, SSL session present: ${sslSession.isDefined}"
    )
    if (sslSession.isEmpty) {
      log.warn(s"No SSL session for gRPC call: ${call.getMethodDescriptor.getFullMethodName}")
    }
    new InterceptionListener(next.startCall(call, headers), call)
  }

  implicit private val logSource: LogSource = LogSource(this.getClass)
  private val log                           = Log.logId

  private class InterceptionListener[ReqT, RespT](
      next: ServerCall.Listener[ReqT],
      call: ServerCall[ReqT, RespT]
  ) extends ServerCall.Listener[ReqT] {

    @volatile
    private var closeWithStatus = Option.empty[Status]

    override def onHalfClose(): Unit =
      closeWithStatus.fold(next.onHalfClose())(call.close(_, new Metadata()))

    override def onCancel(): Unit = {
      log.debug(s"gRPC call cancelled: ${call.getMethodDescriptor.getFullMethodName}")
      next.onCancel()
    }

    override def onComplete(): Unit = next.onComplete()
    override def onReady(): Unit    = next.onReady()

    override def onMessage(message: ReqT): Unit = {
      log.debug(
        s"gRPC message received: ${call.getMethodDescriptor.getFullMethodName}, message type: ${message.getClass.getName}"
      )
      message match {
        case TLRequest(Protocol(RHeader(sender, nid), msg)) =>
          if (nid == networkID) {
            if (log.isTraceEnabled) {
              val peerNode = ProtocolHelper.toPeerNode(sender)
              val msgType  = msg.getClass.toString
              log.trace(s"Request [$msgType] from peer ${peerNode.toAddress}")
            }
            val sslSession: Option[SSLSession] = Option(
              call.getAttributes.get(Grpc.TRANSPORT_ATTR_SSL_SESSION)
            )
            if (sslSession.isEmpty) {
              log.warn("No TLS Session. Closing connection")
              close(Status.UNAUTHENTICATED.withDescription("No TLS Session"))
            } else {
              sslSession.foreach { session =>
                val certPublicAddr =
                  CertificateHelper.publicAddress(session.getPeerCertificates.head.getPublicKey)
                val senderIdBytes = sender.id.toByteArray
                val verified      = certPublicAddr.exists(_ sameElements senderIdBytes)

                if (!verified) {
                  import coop.rchain.shared.Base16
                  val certAddrHex = certPublicAddr.map(Base16.encode).getOrElse("NONE")
                  val senderIdHex = Base16.encode(senderIdBytes)
                  log.warn(
                    s"Certificate verification failed. TLS cert address: $certAddrHex, Protocol sender.id: $senderIdHex. Closing connection"
                  )
                }

                if (verified)
                  next.onMessage(message)
                else {
                  close(Status.UNAUTHENTICATED.withDescription("Certificate verification failed"))
                }
              }
            }
          } else {
            val nidStr = if (nid.isEmpty) "<empty>" else nid
            log.warn(s"Wrong network id '$nidStr'. Closing connection")
            close(
              Status.PERMISSION_DENIED
                .withDescription(
                  s"Wrong network id '$nidStr'. This node runs on network '$networkID'"
                )
            )
          }
        case TLRequest(_) =>
          log.warn(s"Malformed message $message")
          close(Status.INVALID_ARGUMENT.withDescription("Malformed message"))
        case _ => next.onMessage(message)
      }
    }

    private def close(status: Status): Unit =
      closeWithStatus = Some(status)
  }
}
