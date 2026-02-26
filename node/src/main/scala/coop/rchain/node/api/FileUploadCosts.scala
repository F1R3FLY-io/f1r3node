package coop.rchain.node.api

/**
  * Storage-proportional phlo pricing constants and calculations.
  *
  * Formula: `totalPhlo = BASE_REGISTER_PHLO + (fileSize × phloPerStorageByte)`
  *
  * All arithmetic uses `Math.multiplyExact` / `Math.addExact` to fail-fast
  * on overflow rather than silently wrapping.
  */
object FileUploadCosts {

  /** Base phlogiston cost for the on-chain file registration deploy. */
  val BASE_REGISTER_PHLO: Long = 300L

  /** Default phlogiston charged per byte of stored file data. */
  val DEFAULT_PHLO_PER_STORAGE_BYTE: Long = 1L

  /**
    * Computes the minimum phlo required to upload and register a file.
    *
    * @param fileSize           total file size in bytes
    * @param phloPerStorageByte phlo charged per byte (from config)
    * @param baseRegisterPhlo   base phlo cost for the registration deploy (from config)
    * @return `baseRegisterPhlo + fileSize * phloPerStorageByte`
    * @throws ArithmeticException if the result overflows `Long`
    */
  def totalRequired(
      fileSize: Long,
      phloPerStorageByte: Long,
      baseRegisterPhlo: Long = BASE_REGISTER_PHLO
  ): Long =
    Math.addExact(baseRegisterPhlo, Math.multiplyExact(fileSize, phloPerStorageByte))
}
