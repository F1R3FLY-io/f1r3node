import Dependencies._
import BNFC._
import Rholang._
import NativePackagerHelper._
import com.typesafe.sbt.packager.docker._
import sys.process._
import javax.print.attribute.standard.RequestingUserName
import java.io.IOException

//allow stopping sbt tasks using ctrl+c without killing sbt itself
Global / cancelable := true

//disallow any unresolved version conflicts at all for faster feedback
//Global / conflictManager := ConflictManager.strict
//resolve all version conflicts explicitly
//Global / dependencyOverrides := Dependencies.overrides

// This for M2 to start sbt-server to local network
Global / serverConnectionType := ConnectionType.Tcp

Global / PB.protocVersion := "3.24.3"

// ThisBuild / libraryDependencies += compilerPlugin("io.tryp" % "splain" % "0.5.8" cross CrossVersion.patch)

// Helper function to safely get git commit hash
def getSafeGitCommit(): String = {
  try {
    val commit = "git rev-parse HEAD".!!.trim
    if (commit.nonEmpty && commit.length == 40) commit else "unknown"
  } catch {
    case _: IOException => "unknown"
    case _: Exception => "unknown"
  }
}

inThisBuild(List(
  publish / skip := true
))

val javaOpens = List(
  "--add-opens",
  "java.base/sun.security.util=ALL-UNNAMED",
  "--add-opens",
  "java.base/java.nio=ALL-UNNAMED",
  "--add-opens",
  "java.base/sun.nio.ch=ALL-UNNAMED"
)
inThisBuild(List(
  Test / javaOptions := javaOpens
))

lazy val projectSettings = Seq(
  organization := "f1r3fly-io",
  scalaVersion := "2.12.15",
  resolvers ++=
    Resolver.sonatypeOssRepos("releases") ++
    Resolver.sonatypeOssRepos("snapshots") ++
    Seq(
      Resolver.mavenLocal,
      "jitpack" at "https://jitpack.io"
    ),
  wartremoverExcluded += sourceManaged.value,
  Compile / compile / wartremoverWarnings ++= Warts.allBut(
    // those we want
    Wart.DefaultArguments,
    Wart.ImplicitConversion,
    Wart.ImplicitParameter,
    Wart.LeakingSealed,
    Wart.Recursion,
    // those don't want
    Wart.Any,
    Wart.AnyVal,
    Wart.ArrayEquals,
    Wart.AsInstanceOf,
    Wart.AutoUnboxing,
    Wart.CaseClassPrivateApply,
    Wart.EitherProjectionPartial,
    Wart.Equals,
    Wart.ExplicitImplicitTypes,
    Wart.FinalCaseClass,
    Wart.FinalVal,
    Wart.GlobalExecutionContext,
    Wart.IterableOps,
    Wart.JavaConversions,
    Wart.JavaNetURLConstructors,
    Wart.ListAppend,
    Wart.ListUnapply,
    Wart.MapUnit,
    Wart.MutableDataStructures,
    Wart.NoNeedImport,
    Wart.NonUnitStatements,
    Wart.Nothing,
    Wart.Null,
    Wart.Option2Iterable,
    Wart.OptionPartial,
    Wart.Overloading,
    Wart.PlatformDefault,
    Wart.Product,
    Wart.PublicInference,
    Wart.RedundantConversions,
    Wart.Return,
    Wart.SeqApply,
    Wart.SeqUpdated,
    Wart.Serializable,
    Wart.SizeIs,
    Wart.SizeToLength,
    Wart.SortedMaxMinOption,
    Wart.StringPlusAny,
    Wart.Throw,
    Wart.ToString,
    Wart.TripleQuestionMark,
    Wart.Var,
    Wart.While
  ),
  scalafmtOnCompile := !sys.env.contains("CI"), // disable in CI environments
  Test / testOptions += Tests.Argument("-oD"), //output test durations
  dependencyOverrides ++= Seq(
    "io.kamon" %% "kamon-core" % kamonVersion
  ),
  javacOptions ++= Seq("-source", "11", "-target", "11"),
  Test / fork := true,
  Test / parallelExecution := false,
  Test / testForkedParallel := false,
  assembly / assemblyMergeStrategy := {
    // For some reason, all artifacts from 'io.netty' group contain this file with different contents.
    // Discarding it as it's not needed.
    case path if path.endsWith("io.netty.versions.properties") => MergeStrategy.discard
    // The scala compiler includes native bindings for jansi under the same path jansi does.
    // This should pick the ones provided by jansi.
    case path if path.startsWith("META-INF/native/") && path.contains("jansi") => MergeStrategy.last
    case path                                                                  => MergeStrategy.defaultMergeStrategy(path)
  }
) ++
// skip api doc generation if SKIP_DOC env variable is defined
  Seq(sys.env.get("SKIP_DOC")).flatMap { _ =>
    Seq(
      Compile / packageDoc / publishArtifact := false,
      Compile / doc / sources := Seq.empty
    )
  }

// a namespace for generative tests (or other tests that take a long time)
lazy val SlowcookerTest = config("slowcooker") extend (Test)

lazy val benchmark = taskKey[Unit]("Run benchmark, and update changelog")

benchmark := {
  val log            = streams.value.log
  val currentVersion = version.value

  log.info("Running benchmark tests...")
}

lazy val compilerSettings = CompilerSettings.options ++ Seq(
  crossScalaVersions := Seq(scalaVersion.value)
)

// Before starting sbt export YOURKIT_AGENT set to the profiling agent appropriate
// for your OS (https://www.yourkit.com/docs/java/help/agent.jsp)
lazy val profilerSettings = Seq(
  run / javaOptions ++= sys.env
    .get("YOURKIT_AGENT")
    .map(agent => s"-agentpath:$agent=onexit=snapshot,sampling")
    .toSeq,
  reStart / javaOptions ++= (run / javaOptions).value
)

lazy val commonSettings = projectSettings ++ compilerSettings ++ profilerSettings

lazy val shared = (project in file("shared"))
  .settings(commonSettings: _*)
  .settings(
    libraryDependencies ++= commonDependencies ++ Seq(
      catsCore,
      catsEffect,
      catsMtl,
      catsTagless,
      fs2Core,
      lz4,
      monix,
      scodecCore,
      scodecCats,
      scodecBits,
      scalapbRuntimegGrpc,
      lmdbjava,
      catsEffectLawsTest,
      catsLawsTest,
      catsLawsTestkitTest,
      enumeratum,
      jaxb,
      kittens,
      sourcecode
    )
  )

lazy val graphz = (project in file("graphz"))
  .settings(commonSettings: _*)
  .settings(
    libraryDependencies ++= commonDependencies ++ Seq(
      catsCore,
      catsEffect,
      catsMtl
    )
  )
  .dependsOn(shared)

lazy val casper = (project in file("casper"))
  .configs(SlowcookerTest)
  .settings(commonSettings: _*)
  .settings(rholangSettings: _*)
  .settings(inConfig(SlowcookerTest)(Defaults.testSettings): _*)
  .settings(inConfig(SlowcookerTest)(org.scalafmt.sbt.ScalafmtPlugin.scalafmtConfigSettings))
  .settings(
    name := "casper",
    libraryDependencies ++= commonDependencies ++ protobufLibDependencies ++ Seq(
      catsCore,
      catsRetry,
      catsMtl,
      monix,
      fs2Core,
      fs2Io,
      scalacheck % "slowcooker"
    )
  )
  .dependsOn(
    blockStorage % "compile->compile;test->test",
    comm         % "compile->compile;test->test",
    shared       % "compile->compile;test->test",
    graphz,
    crypto,
    models % "compile->compile;test->test",
    rspace,
    rholang % "compile->compile;test->test"
  )

lazy val comm = (project in file("comm"))
  .settings(commonSettings: _*)
  .settings(
    dependencyOverrides += "org.slf4j" % "slf4j-api" % "1.7.25",
    libraryDependencies ++= commonDependencies ++ kamonDependencies ++ protobufDependencies ++ Seq(
      grpcNetty,
      grpcCensus,
      openCensus,
      nettyBoringSsl,
      scalapbRuntimegGrpc,
      scalaUri,
      weupnp,
      hasher,
      catsCore,
      catsMtl,
      catsTagless,
      monix,
      guava,
      perfmark6,
      perfmark7,
      perfmark9
    ),
    Compile / PB.targets := Seq(
      scalapb.gen(grpc = false)  -> (Compile / sourceManaged).value,
      grpcmonix.generators.gen() -> (Compile / sourceManaged).value
    )
  )
  .dependsOn(shared % "compile->compile;test->test", crypto, models)

lazy val crypto = (project in file("crypto"))
  .settings(commonSettings: _*)
  .settings(
    name := "crypto",
    libraryDependencies ++= commonDependencies ++ protobufLibDependencies ++ Seq(
      guava,
      bouncyPkixCastle,
      bouncyProvCastle,
      scalacheck,
      kalium,
      bitcoin_s,
      scodecBits
    ),
    fork := true
  )
  .dependsOn(shared)

lazy val models = (project in file("models"))
  .settings(commonSettings: _*)
  .settings(
    libraryDependencies ++= commonDependencies ++ protobufDependencies ++ Seq(
      catsCore,
      magnolia,
      scalapbCompiler,
      scalacheck % "test",
      scalacheckShapeless,
      scalapbRuntimegGrpc
    ),
    Compile / PB.targets := Seq(
      coop.rchain.scalapb.gen(flatPackage = true, grpc = false) -> (Compile / sourceManaged).value,
      grpcmonix.generators.gen()                                -> (Compile / sourceManaged).value
    )
  )
  .dependsOn(shared % "compile->compile;test->test", rspace)

lazy val node = (project in file("node"))
  .settings(commonSettings: _*)
  .enablePlugins(JavaAppPackaging, DockerPlugin, RpmPlugin, BuildInfoPlugin)
  .settings(
    name := "rnode",
    maintainer := "F1r3fly.io LCA https://f1r3fly.io/",
    packageSummary := "F1R3FLY Node",
    packageDescription := "F1R3FLY Node - blockchain node server software.",
    // Universal packaging settings
    executableScriptName := "rnode",
    bashScriptConfigLocation := Some("${app_home}/../conf/rnode.conf"),
    // RPM-specific settings
    rpmVendor := "f1r3fly-io",
    rpmLicense := Some("SSL"),
    rpmUrl := Some("https://f1r3fly.io"),
    rpmRelease := "1",
    rpmRequirements ++= Seq("java-17-openjdk"),
    rpmChangelogFile := Some("CHANGELOG.md"),
    // Debian-specific settings
    debianPackageDependencies ++= Seq("java17-runtime-headless"),
    Debian / maintainer := "F1R3FLY.io LCA <support@f1r3fly.io>",
    Debian / packageArchitecture := "all",
    debianChangelog := Some(file("CHANGELOG.md")),
    // File mappings for Linux (Debian and RPM)
    linuxPackageMappings ++= Seq(
      packageMapping(
        (Compile / packageBin).value -> "/usr/share/rnode/rnode.jar"
      ) withPerms "0644" withUser "daemon" withGroup "daemon"
    ) ++ (Universal / mappings).value.collect {
      case (file, "bin/rnode") =>
        packageMapping(file -> "/usr/bin/rnode") withPerms "0755" withUser "daemon" withGroup "daemon"
    },
    // Ensure version is compatible
    Rpm / version := version.value.replace("+", "-").replace("-SNAPSHOT", ""),
    Debian / version := version.value.replace("+", "-").replace("-SNAPSHOT", ""),
    libraryDependencies ++=
      apiServerDependencies ++ commonDependencies ++ kamonDependencies ++ protobufDependencies ++ Seq(
        catsCore,
        catsTagless,
        catsRetry,
        grpcNetty,
        grpcCensus,
        openCensus,
        grpcServices,
        jline,
        scallop,
        scalaUri,
        scalapbRuntimegGrpc,
        circeParser,
        circeGenericExtras,
        pureconfig,
        perfmark6,
        perfmark7,
        perfmark9
      ),
    Compile / PB.targets := Seq(
      scalapb.gen(grpc = false)  -> (Compile / sourceManaged).value / "protobuf",
      grpcmonix.generators.gen() -> (Compile / sourceManaged).value / "protobuf"
    ),
    buildInfoKeys := Seq[BuildInfoKey](name, version, scalaVersion, sbtVersion, "gitHeadCommit" -> getSafeGitCommit()),
    buildInfoPackage := "coop.rchain.node",
    Compile / mainClass := Some("coop.rchain.node.Main"),
    Compile / discoveredMainClasses := Seq(),
    assembly / mainClass := Some("coop.rchain.node.Main"),
    assembly / assemblyMergeStrategy := {
      // TODO: investigate if still needed? Got error after adding ExternalCommunicationService.proto and ExternalCommunicationServiceCommon.proto
      case x if x.startsWith("META-INF") => MergeStrategy.discard
      case x if x.endsWith(".proto") => MergeStrategy.first
      //
      case x if x.endsWith("io.netty.versions.properties") => MergeStrategy.first
      case x if x.endsWith("scala/annotation/nowarn.class") => MergeStrategy.discard
      case x if x.endsWith("scala/annotation/nowarn$.class") => MergeStrategy.discard
      case x if x.endsWith("module-info.class") => MergeStrategy.discard
      case x =>
        val oldStrategy = (assembly / assemblyMergeStrategy).value
        oldStrategy(x)
    },
    /* Dockerization */
    dockerRepository := Option("f1r3flyindustries"),
    dockerUsername := Option(organization.value),
    dockerAliases ++=
      sys.env
        .get("DRONE_BUILD_NUMBER")
        .toSeq
        .map(num => dockerAlias.value.withTag(Some(s"DRONE-${num}"))),
    dockerAlias := dockerAlias.value.withName("f1r3fly-scala-node"),
    dockerUpdateLatest := sys.env.get("DRONE").isEmpty,
    dockerBaseImage := "ghcr.io/graalvm/jdk:ol8-java17-22.3.3",
    dockerEntrypoint := List("/opt/docker/bin/rnode", "--profile=docker", "-XX:ErrorFile=/var/lib/rnode/hs_err_pid%p.log"),
    Docker / daemonUserUid := None,
    Docker / daemonUser := "daemon",
    dockerExposedPorts := List(40400, 40401, 40402, 40403, 40404),
    dockerBuildOptions := {
      val platform = sys.env.getOrElse("PLATFORM", "linux/amd64")
      Seq("--platform", platform, "-t", "f1r3flyindustries/f1r3fly-scala-node:latest")
    },
    dockerCommands := {
      // Retrieve the default Docker commands provided by sbt-native-packager
      val defaultCommands = dockerCommands.value

      // Define the RUN command to install dependencies
      val installCmd = Cmd("RUN", """export ARCH=$(uname -m | sed 's/aarch64/arm64/') && \
                                    microdnf update && \
                                    microdnf install jq gzip && \
                                    curl -LO https://github.com/fullstorydev/grpcurl/releases/download/v1.8.9/grpcurl_1.8.9_linux_$ARCH.tar.gz && \
                                    tar -xzf grpcurl_1.8.9_linux_$ARCH.tar.gz && \
                                    rm -fr LICENSE grpcurl_1.8.9_linux_$ARCH.tar.gz && \
                                    chmod a+x grpcurl && \
                                    mv grpcurl /usr/local/bin""")

      // Find the index of the FROM command with "mainstage" in its arguments
      val mainstageIndex = defaultCommands.indexWhere {
        case Cmd("FROM", args @ _*) => args.exists(_.contains("mainstage"))
        case _ => false
      }

      // Throw an exception if mainstage is not found
      if (mainstageIndex == -1) {
        throw new IllegalStateException("Cannot find mainstage in dockerCommands")
      }

      // Split the commands around the mainstage index
      val beforeMainstage = defaultCommands.take(mainstageIndex + 1)
      val afterMainstage = defaultCommands.drop(mainstageIndex + 1)

      // Combine the commands with additional instructions
      beforeMainstage ++ Seq(
        Cmd("USER", "root"), // Switch to root for installations
        installCmd           // Install jq and grpcurl
      ) ++ afterMainstage ++ Seq(
        Cmd("LABEL", s"""MAINTAINER="${maintainer.value}""""),
        Cmd("LABEL", s"""version="${version.value}""""),
        Cmd("USER", (Docker / daemonUser).value), // Switch back to the default user
        Cmd(
          "HEALTHCHECK CMD",
          """grpcurl -plaintext 127.0.0.1:40401 casper.v1.DeployService.status | jq -e && \
            curl -s 127.0.0.1:40403/status | jq -e"""
        ),
        ExecCmd("CMD", "run") // Start the application
      )
    },
    Universal / javaOptions ++= List(
      "-J--add-opens",
      "-Jjava.base/sun.security.util=ALL-UNNAMED",
      "-J--add-opens",
      "-Jjava.base/java.nio=ALL-UNNAMED",
      "-J--add-opens",
      "-Jjava.base/sun.nio.ch=ALL-UNNAMED"
    ),
    // Replace unsupported character `+`
    Docker / version := { version.value.replace("+", "__") },
    Docker / mappings ++= {
      val base = (Docker / defaultLinuxInstallLocation).value
      directory((rholang / baseDirectory).value / "examples")
        .map { case (f, p) => f -> s"$base/$p" }
    },
    // End of sbt-native-packager settings
    connectInput := true,
    outputStrategy := Some(StdoutOutput),
    libraryDependencies += {
      val version = scalaBinaryVersion.value match {
        case "2.10" => "1.0.3"
        case "2.11" => "1.6.7"
        case _      ⇒ "2.5.11"
      }
      "com.lihaoyi" % "ammonite" % version % "test" cross CrossVersion.full
    },
    (Test / sourceGenerators) += Def.task {
      val file = (Test / sourceManaged).value / "amm.scala"
      IO.write(file, """object amm extends App { ammonite.AmmoniteMain.main(args) }""")
      Seq(file)
    }.taskValue
  )
  .dependsOn(casper % "compile->compile;test->test", comm, crypto, rholang)

lazy val regex = (project in file("regex"))
  .settings(commonSettings: _*)
  .settings(libraryDependencies ++= commonDependencies)

lazy val rholang = (project in file("rholang"))
  .settings(commonSettings: _*)
  .settings(bnfcSettings: _*)
  .settings(
    name := "rholang",
    scalacOptions ++= Seq(
      "-language:existentials",
      "-language:higherKinds",
      "-Yno-adapted-args",
//      "-Xfatal-warnings",
      "-Xlint:_,-missing-interpolator" // disable "possible missing interpolator" warning
    ),
    Compile / packageDoc / publishArtifact := false,
    Compile / doc / sources := Seq.empty,
    libraryDependencies ++= commonDependencies ++ Seq(
      catsMtl,
      catsEffect,
      monix,
      scallop,
      lightningj,
      catsLawsTest,
      catsLawsTestkitTest,
      catsMtlLawsTest,
      cequenceIo,
      akkaHttp,
      akkaHttpSprayJson,
      akkaStream
    ),
    // TODO: investigate if still needed?
    // assembly / mainClass := Some("coop.rchain.rho2rose.Rholang2RosetteCompiler"),
    //constrain the resource usage so that we hit SOE-s and OOME-s more quickly should they happen
    Test / javaOptions ++= Seq("-Xss240k", "-XX:MaxJavaStackTraceDepth=10000", "-Xmx128m")
  )
  .dependsOn(
    models % "compile->compile;test->test",
    rspace % "compile->compile;test->test",
    shared % "compile->compile;test->test",
    crypto
  )

lazy val rholangCLI = (project in file("rholang-cli"))
  .settings(commonSettings: _*)
  .settings(
    assembly / mainClass := Some("coop.rchain.rholang.interpreter.RholangCLI"),
    assembly / assemblyMergeStrategy := {
      case path if path.endsWith("module-info.class") => MergeStrategy.discard
      case path                                       => MergeStrategy.defaultMergeStrategy(path)
    }
  )
  .dependsOn(rholang)

lazy val rholangServer = (project in file("rholang-server"))
  .enablePlugins(NativeImagePlugin)
  .settings(commonSettings)
  .settings(
    libraryDependencies ++= List(
      fs2Io,
      "org.jline"          % "jline"         % "3.21.0",
      "org.scodec"        %% "scodec-stream" % "2.0.3",
      "io.chrisdavenport" %% "fuuid"         % "0.7.0",
      "com.comcast"       %% "ip4s-core"     % "2.0.4",
      "com.monovore"      %% "decline"       % "2.3.0"
    )
  )
  .dependsOn(rholang)

lazy val blockStorage = (project in file("block-storage"))
  .settings(commonSettings: _*)
  .settings(
    name := "block-storage",
    libraryDependencies ++= commonDependencies ++ protobufLibDependencies ++ Seq(
      catsCore,
      catsEffect,
      catsMtl
    )
  )
  .dependsOn(shared, models % "compile->compile;test->test")

// Using dependencyOverrides bc of ConflictManager

lazy val rspace = (project in file("rspace"))
  .configs(IntegrationTest extend Test)
  .enablePlugins(SiteScaladocPlugin, GhpagesPlugin)
  .settings(commonSettings: _*)
  .settings(
    scalacOptions ++= Seq(
//      "-Xfatal-warnings"
    ),
    Defaults.itSettings,
    IntegrationTest / javaOptions := javaOpens,
    IntegrationTest / fork := true,
    IntegrationTest / parallelExecution := false,
    IntegrationTest / testForkedParallel := false,
    name := "rspace",
    libraryDependencies ++= commonDependencies ++ kamonDependencies ++ Seq(
      catsCore,
      fs2Core,
      scodecCore,
      scodecBits
    ),
    /* Tutorial */
    /* Publishing Settings */
    scmInfo := Some(
      ScmInfo(url("https://github.com/rchain/rchain"), "git@github.com:rchain/rchain.git")
    ),
    git.remoteRepo := scmInfo.value.get.connection,
    pomIncludeRepository := { _ =>
      false
    },
    publishMavenStyle := true,
    publishTo := {
      val nexus = "https://oss.sonatype.org/"
      if (isSnapshot.value)
        Some("snapshots" at nexus + "content/repositories/snapshots")
      else
        Some("releases" at nexus + "service/local/staging/deploy/maven2")
    },
    Test / publishArtifact := false,
    licenses := Seq("Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0")),
    homepage := Some(url("https://www.rchain.coop"))
  )
  .dependsOn(shared % "compile->compile;test->test", crypto)

lazy val rspaceBench = (project in file("rspace-bench"))
  .settings(
    commonSettings,
    version := (ThisBuild / version).value,
    libraryDependencies ++= commonDependencies,
    libraryDependencies += "com.esotericsoftware" % "kryo" % "5.0.3",
    dependencyOverrides ++= Seq(
      "org.ow2.asm" % "asm" % "9.0"
    ),
    Jmh / sourceDirectory := (Test / sourceDirectory).value,
    Jmh / classDirectory := (Test / classDirectory).value,
    Jmh / dependencyClasspath := (Test / dependencyClasspath).value,
    // rewire tasks, so that 'jmh:run' automatically invokes 'jmh:compile' (otherwise a clean 'jmh:run' would fail),
    Jmh / compile := (Jmh / compile).dependsOn(Test / compile).value,
    Jmh / run := (Jmh / run).dependsOn(Jmh / Keys.compile).evaluated
  )
  .enablePlugins(JmhPlugin)
  .dependsOn(rspace % "test->test", rholang % "test->test", models % "test->test")

lazy val rchain = (project in file("."))
  .settings(commonSettings: _*)
  .aggregate(
    blockStorage,
    casper,
    comm,
    crypto,
    graphz,
    models,
    node,
    regex,
    rholang,
    rholangCLI,
    rholangServer,
    rspace,
    rspaceBench,
    shared
  )
