package coop.rchain.node.api

import io.circe.{Decoder, Encoder}
import io.circe.generic.semiauto.{deriveDecoder, deriveEncoder}
import io.circe.syntax._
import io.circe.parser.{decode => circeDecode}

/**
  * Metadata persisted alongside an uploaded file as `<hash>.meta.json`.
  */
final case class FileMetadata(
    fileName: String,
    fileSize: Long,
    uploaderPubKey: String,
    timestamp: Long,
    hash: String
)

object FileMetadata {
  implicit val encoder: Encoder[FileMetadata] = deriveEncoder[FileMetadata]
  implicit val decoder: Decoder[FileMetadata] = deriveDecoder[FileMetadata]

  /** Serialize to JSON string. */
  def toJson(m: FileMetadata): String = m.asJson.noSpaces

  /** Deserialize from JSON string. Returns Left with error message on failure. */
  def fromJson(json: String): Either[String, FileMetadata] =
    circeDecode[FileMetadata](json).left.map(_.getMessage)
}
