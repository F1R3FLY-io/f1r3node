package coop.rchain.rholang

package object externalservices {

  /**
    * Check if OpenAI service is enabled based on configuration and environment variables.
    * This uses the same logic as OpenAIServiceImpl.instance to determine the enabled state.
    * Priority order: 1. Environment variable OPENAI_ENABLED, 2. Configuration, 3. Default (false)
    */
  private[rholang] def isOpenAIEnabled: Boolean = {
    import com.typesafe.config.ConfigFactory
    import java.util.Locale

    val config = ConfigFactory.load()

    // Check environment variable first (highest priority)
    val envEnabled = Option(System.getenv("OPENAI_ENABLED")).flatMap { value =>
      value.toLowerCase(Locale.ENGLISH) match {
        case "true" | "1" | "yes" | "on"  => Some(true)
        case "false" | "0" | "no" | "off" => Some(false)
        case _                            => None // Invalid env var value, ignore it
      }
    }

    // Check configuration as fallback
    val configEnabled = if (config.hasPath("openai.enabled")) {
      Some(config.getBoolean("openai.enabled"))
    } else {
      None
    }

    // Resolve final enabled state: env takes priority, then config, then instance false
    envEnabled.getOrElse(configEnabled.getOrElse(false))
  }

  /**
    * Check if Ollama service is enabled based on configuration and environment variables.
    * This uses the same logic as OllamaServiceImpl.instance to determine the enabled state.
    * Priority order: 1. Environment variable OLLAMA_ENABLED, 2. Configuration, 3. Default (false)
    */
  private[rholang] def isOllamaEnabled: Boolean = {
    import com.typesafe.config.ConfigFactory
    import java.util.Locale

    val config = ConfigFactory.load()

    // Check environment variable first (highest priority)
    val envEnabled = Option(System.getenv("OLLAMA_ENABLED")).flatMap { value =>
      value.toLowerCase(Locale.ENGLISH) match {
        case "true" | "1" | "yes" | "on"  => Some(true)
        case "false" | "0" | "no" | "off" => Some(false)
        case _                            => None // Invalid env var value, ignore it
      }
    }

    // Check configuration as fallback
    val configEnabled = if (config.hasPath("ollama.enabled")) {
      Some(config.getBoolean("ollama.enabled"))
    } else {
      None
    }

    // Resolve final enabled state: env takes priority, then config, then default false
    envEnabled.getOrElse(configEnabled.getOrElse(false))
  }
}
