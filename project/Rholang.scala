import sbt._
import Keys._
import sbt.io.Path.relativeTo

/*
 * There doesn't seem to be a good way to have a plugin defined in
 * one sub-project be used in another sub-project. So we need to
 * call an assembled jar instead, building it if necessary.
 */
object Rholang {

  lazy val rholangSettings = Seq(
    exportJars := true,
    Compile / packageBin / mappings ++= {
      val generatedProtos = (Compile / resourceManaged).value ** "*.proto"
      generatedProtos pair relativeTo((Compile / resourceManaged).value)
    },
    Test / packageBin / mappings ++= {
      val generatedProtos = (Test / resourceManaged).value ** "*.proto"
      generatedProtos pair relativeTo((Test / resourceManaged).value)
    }
  )
}
