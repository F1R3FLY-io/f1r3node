package coop.rchain.comm.transport

import java.net.Socket
import java.security.KeyStore
import java.security.cert.{CertificateException, X509Certificate}
import coop.rchain.crypto.util.CertificateHelper
import coop.rchain.shared.Base16
import io.netty.handler.ssl.util.SimpleTrustManagerFactory
import io.netty.util.internal.EmptyArrays
import com.typesafe.scalalogging.Logger

import javax.net.ssl._

class HostnameTrustManagerFactory private () extends SimpleTrustManagerFactory {
  def engineInit(keyStore: KeyStore): Unit                                 = {}
  def engineInit(managerFactoryParameters: ManagerFactoryParameters): Unit = {}
  def engineGetTrustManagers(): Array[TrustManager] =
    Array[TrustManager](hostNameTrustManager)

  private[this] val hostNameTrustManager = new HostnameTrustManager()
}

object HostnameTrustManagerFactory {
  val Instance: HostnameTrustManagerFactory = new HostnameTrustManagerFactory()
}

/**
  * This wart exists because that's how grpc works. They looooovveee throwing exceptions
  */
@SuppressWarnings(Array("org.wartremover.warts.Throw", "org.wartremover.warts.NonUnitStatements"))
private class HostnameTrustManager extends X509ExtendedTrustManager {
  private val logger = Logger[HostnameTrustManager]

  def checkClientTrusted(
      x509Certificates: Array[X509Certificate],
      authType: String,
      socket: Socket
  ): Unit =
    throw new CertificateException("Not allowed validation method")

  def checkClientTrusted(
      x509Certificates: Array[X509Certificate],
      authType: String,
      sslEngine: SSLEngine
  ): Unit = {
    Option(sslEngine.getHandshakeSession)
      .getOrElse(throw new CertificateException("No handshake session"))

    val cert = x509Certificates.head
    val peerHost = CertificateHelper
      .publicAddress(cert.getPublicKey)
      .map(Base16.encode)
      .getOrElse(
        throw new CertificateException(s"Certificate's public key has the wrong algorithm")
      )
    checkIdentity(Some(peerHost), cert, "https")
  }

  def checkClientTrusted(x509Certificates: Array[X509Certificate], authType: String): Unit =
    throw new CertificateException("Not allowed validation method")

  def checkServerTrusted(
      x509Certificates: Array[X509Certificate],
      authType: String,
      socket: Socket
  ): Unit =
    throw new CertificateException("Not allowed validation method")

  def checkServerTrusted(
      x509Certificates: Array[X509Certificate],
      authType: String,
      sslEngine: SSLEngine
  ): Unit = {
    val sslSession = Option(sslEngine.getHandshakeSession)
      .getOrElse(throw new CertificateException("No handshake session"))

    // check endpoint identity
    Option(sslEngine.getSSLParameters.getEndpointIdentificationAlgorithm) match {
      case Some(identityAlg) if identityAlg.nonEmpty =>
        val cert     = x509Certificates.head
        val peerHost = Option(sslSession.getPeerHost)
        checkIdentity(peerHost, cert, identityAlg)
        CertificateHelper
          .publicAddress(cert.getPublicKey)
          .map(Base16.encode)
          .filter(_ == peerHost.getOrElse(""))
          .getOrElse(
            throw new CertificateException(
              s"Certificate's public address doesn't match the hostname"
            )
          )

      case _ =>
        throw new CertificateException("No endpoint identification algorithm")
    }
  }

  def checkServerTrusted(x509Certificates: Array[X509Certificate], authType: String): Unit =
    throw new CertificateException("Not allowed validation method")

  def getAcceptedIssuers: Array[X509Certificate] = EmptyArrays.EMPTY_X509_CERTIFICATES

  private def checkIdentity(
      hostname: Option[String],
      cert: X509Certificate,
      algorithm: String
  ): Unit = {
    import sun.security.util.HostnameChecker
    import scala.collection.JavaConverters._

    algorithm.toLowerCase match {
      case "https" =>
        val host = hostname
          .filter(_.startsWith("["))
          .filter(_.endsWith("]"))
          .map(h => h.substring(1, h.length - 1))
          .orElse(hostname)
          .getOrElse("")

        try {
          HostnameChecker.getInstance(HostnameChecker.TYPE_TLS).`match`(host, cert)
        } catch {
          case e: CertificateException =>
            // Extract certificate SANs for debugging
            val sans = Option(cert.getSubjectAlternativeNames)
              .map(_.asScala.toList.flatMap { san =>
                san.asScala.toList match {
                  case List(_: Integer, value: String) => Some(value)
                  case _                               => None
                }
              })
              .getOrElse(List.empty)

            logger.error(
              s"TLS certificate invalid: node ID '$host' not found in certificate SANs [${sans.mkString(", ")}]. " +
                s"Regenerate certificates with 'cd docker && ./generate-genesis-keys.sh' to include node IDs in SANs."
            )
            throw e
        }
      case _ =>
        throw new CertificateException(s"Unknown identification algorithm: $algorithm")
    }
  }
}
