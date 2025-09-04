package coop.rchain.rholang.build
import com.sun.jna.Memory
import coop.rchain.models.NormalizerEnv.ToEnvMap
import coop.rchain.models.{NormalizerEnv, Par}
import shapeless.HNil
import coop.rchain.models.rholang_scala_rust_types.SourceToAdtParams
import coop.rchain.rholang.JNAInterfaceLoader.RHOLANG_RUST_INSTANCE
import coop.rchain.rholang.build.CompiledRholangSource.sourceToAdt

import coop.rchain.rholang.interpreter.compiler.Compiler
import monix.eval.Coeval

import scala.io.Source

/** TODO: Currently all calls to this class use empty environment. See [[NormalizerEnv]]. */
abstract class CompiledRholangSource[Env](val code: String, val normalizerEnv: NormalizerEnv[Env])(
    implicit ev: ToEnvMap[Env]
) {
  val path: String
  val term: Par = {
    val params = SourceToAdtParams(code, normalizerEnv.toEnv)
    sourceToAdt(params)
  }
  // val term: Par = Compiler[Coeval].sourceToADT(code, normalizerEnv.toEnv).value()

  final def env: Map[String, Par] = normalizerEnv.toEnv
}

object CompiledRholangSource {
  def loadSource(classpath: String) = {
    val fileContent = Source.fromResource(classpath).mkString
    s"""//Loaded from resource file <<$classpath>>
       #$fileContent
       #""".stripMargin('#')
  }

  def apply(classpath: String): CompiledRholangSource[HNil] = apply(classpath, NormalizerEnv.Empty)

  def apply[Env](classpath: String, env: NormalizerEnv[Env])(
      implicit ev: ToEnvMap[Env]
  ): CompiledRholangSource[Env] = new CompiledRholangSource[Env](loadSource(classpath), env) {
    override val path: String = classpath
  }

  def sourceToAdt(params: SourceToAdtParams): Par = {
    // serialize the Protobuf parameters into a byte array
    val paramsBytes = params.toByteArray
    val paramsPtr   = new Memory(paramsBytes.length.toLong)
    paramsPtr.write(0, paramsBytes, 0, paramsBytes.length)

    //  call the `source_to_adt` function through JNA
    val resultPtr = RHOLANG_RUST_INSTANCE.source_to_adt(paramsPtr, paramsBytes.length)
    assert(resultPtr != null, "Rust function returned null pointer")

    try {
      // Decode the result
      val resultBytesLength = resultPtr.getInt(0)
      val resultBytes       = resultPtr.getByteArray(4, resultBytesLength)
      Par.parseFrom(resultBytes) // Return the `Par' object
    } finally {
      // Deallocate native buffer returned by Rust (includes 4-byte prefix)
      val len = resultPtr.getInt(0)
      RHOLANG_RUST_INSTANCE.rholang_deallocate_memory(resultPtr, len + 4)
    }
  }

}

/**
  * Loads code from a resource file while doing macro substitution with the provided values.
  * The macros have the format $$macro$$
  * @param classpath
  * @param env a sequence of pairs macro -> value
  * @return
  */
abstract class CompiledRholangTemplate[Env](
    classpath: String,
    normalizerEnv0: NormalizerEnv[Env],
    env: (String, Any)*
)(implicit ev: ToEnvMap[Env])
    extends CompiledRholangSource[Env](
      CompiledRholangTemplate.loadTemplate(classpath, env),
      normalizerEnv0
    ) {
  override val path: String = classpath
}

object CompiledRholangTemplate {
  def loadTemplate(classpath: String, macros: Seq[(String, Any)]) = {
    val originalContent = Source.fromResource(classpath).mkString
    val finalContent = macros.foldLeft(originalContent) {
      case (content, (name, value)) => content.replace(s"""$$$$$name$$$$""", value.toString)
    }

    s"""//Loaded from resource file <<$classpath>>
       #$finalContent
       #""".stripMargin('#')
  }
}
