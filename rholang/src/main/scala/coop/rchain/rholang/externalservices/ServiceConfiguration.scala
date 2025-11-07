package coop.rchain.rholang.externalservices

/**
  * Configuration for OpenAI service.
  *
  * @param enabled Whether the OpenAI service is enabled
  * @param apiKey OpenAI API key
  * @param validateApiKey Whether to validate the API key at startup
  * @param validationTimeoutSec Timeout for API key validation in seconds
  */
final case class OpenAIConf(
    enabled: Boolean,
    apiKey: String,
    validateApiKey: Boolean,
    validationTimeoutSec: Int
)

/**
  * Configuration for Ollama service.
  *
  * @param enabled Whether the Ollama service is enabled
  * @param baseUrl Base URL for Ollama API
  * @param defaultModel Default model to use when none is specified
  * @param validateConnection Whether to validate connection at startup
  * @param timeoutSec Timeout for API requests in seconds
  */
final case class OllamaConf(
    enabled: Boolean,
    baseUrl: String,
    defaultModel: String,
    validateConnection: Boolean,
    timeoutSec: Int
)

/**
  * Configuration for Nunet DMS service.
  *
  * @param enabled Whether the Nunet service is enabled
  * @param cliPath Path to the nunet CLI executable
  * @param context Context for nunet actor commands (typically "user" or "dms")
  * @param timeout Timeout for CLI commands in seconds
  * @param passphrase DMS passphrase for authentication (passed as DMS_PASSPHRASE env var)
  */
final case class NunetConf(
    enabled: Boolean,
    cliPath: String,
    context: String,
    timeout: Int,
    passphrase: String
)
