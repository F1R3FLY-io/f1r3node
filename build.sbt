import Dependencies._
import BNFC._
import Rholang._
import NativePackagerHelper._
import com.typesafe.sbt.packager.docker._
import sys.process._
import javax.print.attribute.standard.RequestingUserName

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

inThisBuild(List(
  publish / skip := true,
  publishMavenStyle := true,
  publishTo := Option("GitHub Package Registry" at "https://maven.pkg.github.com/F1R3FLY-io/f1r3fly")
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
  Test / javaOptions := javaOpens,
  IntegrationTest / javaOptions := javaOpens
))

lazy val ensureDockerBuildx = taskKey[Unit]("Ensure that docker buildx configuration exists")
lazy val dockerBuildWithBuildx = taskKey[Unit]("Build docker images using buildx")
lazy val dockerBuildxSettings = Seq(
  ensureDockerBuildx := {
    if (Process("docker buildx inspect multi-arch-builder").! == 1) {
      Process("docker buildx create --use --name multi-arch-builder", baseDirectory.value).!
    }
  },
  dockerBuildWithBuildx := {
    streams.value.log("Building and pushing image with Buildx")
    dockerAliases.value.foreach(
      alias => Process("docker buildx build --platform=linux/arm64,linux/amd64 --push -t " +
        alias + " .", baseDirectory.value / "target" / "docker"/ "stage").!
    )
  },
  publish in Docker := Def.sequential(
    publishLocal in Docker,
    ensureDockerBuildx,
    dockerBuildWithBuildx
  ).value
)

lazy val projectSettings = Seq(
  organization := "f1r3fly-io",
  scalaVersion := "2.12.15",
  resolvers ++= Seq(
    Resolver.sonatypeRepo("releases"),
    Resolver.sonatypeRepo("snapshots"),
    Resolver.mavenLocal,
    "jitpack" at "https://jitpack.io"
  ),
  wartremoverExcluded += sourceManaged.value,
  wartremoverWarnings in (Compile, compile) ++= Warts.allBut(
    // those we want
    Wart.DefaultArguments,
    Wart.ImplicitParameter,
    Wart.ImplicitConversion,
    Wart.LeakingSealed,
    Wart.Recursion,
    // those don't want
    Wart.NonUnitStatements,
    Wart.Overloading,
    Wart.Nothing,
    Wart.Equals,
    Wart.PublicInference,
    Wart.IterableOps,
    Wart.ArrayEquals,
    Wart.While,
    Wart.Any,
    Wart.Product,
    Wart.Serializable,
    Wart.OptionPartial,
    Wart.EitherProjectionPartial,
    Wart.Option2Iterable,
    Wart.ToString,
    Wart.JavaConversions,
    Wart.MutableDataStructures,
    Wart.FinalVal,
    Wart.Null,
    Wart.AsInstanceOf,
    Wart.ExplicitImplicitTypes,
    Wart.StringPlusAny,
    Wart.AnyVal
  ),
  scalafmtOnCompile := !sys.env.contains("CI"), // disable in CI environments
  testOptions in Test += Tests.Argument("-oD"), //output test durations
  dependencyOverrides ++= Seq(
    "io.kamon" %% "kamon-core" % kamonVersion
  ),
  javacOptions ++= Seq("-source", "11", "-target", "11"),
  Test / fork := true,
  Test / parallelExecution := false,
  Test / testForkedParallel := false,
  IntegrationTest / fork := true,
  IntegrationTest / parallelExecution := false,
  IntegrationTest / testForkedParallel := false,
  assemblyMergeStrategy in assembly := {
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
      publishArtifact in (Compile, packageDoc) := false,
      publishArtifact in packageDoc := false,
      sources in (Compile, doc) := Seq.empty
    )
  }

// a namespace for generative tests (or other tests that take a long time)
lazy val SlowcookerTest = config("slowcooker") extend (Test)

/*
// changlog update and git tag
lazy val release = taskKey[Unit]("Run benchmark, tag new release, and update changelog")

release := {
  val log            = streams.value.log
  val currentVersion = version.value

  log.info("Creating new release...")
  if (Seq("sbt", "rspaceBench").! == 0) {
    import scala.sys.process._
    log.info("Benchmark tests passed.")

    log.info(s"Tagging new release (v$currentVersion)...")
    val shortCommit = "git rev-parse --short HEAD".!!.trim
    if (Seq("git", "tag", s"v$currentVersion-$shortCommit)").! == 0) {
      log.info(s"New release (v$currentVersion) successfully tagged.")
    } else {
      log.error(s"Failed to tag new release (v$currentVersion).")
      throw new IllegalStateException("Failed to tag new release")
    }

    log.info("Updating changelog...")
    val changelogFile    = new File("CHANGELOG.md")
    val changelogContent = IO.read(changelogFile)
    val formattedDate =
      java.time.LocalDate.now.format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd"))

    val newEntry = s"""
      |## [v$currentVersion] - $formattedDate
      |- Added new features
      |- Fixed bugs
      |- Improved performance
      """.stripMargin

    val updatedChangelogContent = newEntry + "\n\n" + changelogContent
    IO.write(changelogFile, updatedChangelogContent)

    log.info("Changelog successfully updated.")
  } else {
    log.error("Benchmark tests failed. Aborting the release process.")
    throw new IllegalStateException("Benchmark tests failed")
  }
}
*/

lazy val benchmark = taskKey[Unit]("Run benchmark, and update changelog")

benchmark := {
  val log            = streams.value.log
  val currentVersion = version.value

  log.info("Running benchmark tests...")

  if (Seq("sbt", "rspacePlusPlus/test").! == 0) {
    log.info("calling rspace++ tests... place call here")
  }
}

lazy val compilerSettings = CompilerSettings.options ++ Seq(
  crossScalaVersions := Seq(scalaVersion.value)
)

// Before starting sbt export YOURKIT_AGENT set to the profiling agent appropriate
// for your OS (https://www.yourkit.com/docs/java/help/agent.jsp)
lazy val profilerSettings = Seq(
  javaOptions in run ++= sys.env
    .get("YOURKIT_AGENT")
    .map(agent => s"-agentpath:$agent=onexit=snapshot,sampling")
    .toSeq,
  javaOptions in reStart ++= (javaOptions in run).value
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
    PB.targets in Compile := Seq(
      scalapb.gen(grpc = false)  -> (sourceManaged in Compile).value,
      grpcmonix.generators.gen() -> (sourceManaged in Compile).value
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
    PB.targets in Compile := Seq(
      coop.rchain.scalapb.gen(flatPackage = true, grpc = false) -> (sourceManaged in Compile).value,
      grpcmonix.generators.gen()                                -> (sourceManaged in Compile).value
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
    maintainer in Debian := "F1R3FLY.io LCA <support@f1r3fly.io>",
    packageArchitecture in Debian := "all",
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
    version in Rpm := version.value.replace("+", "-").replace("-SNAPSHOT", ""),
    version in Debian := version.value.replace("+", "-").replace("-SNAPSHOT", ""),
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
    PB.targets in Compile := Seq(
      scalapb.gen(grpc = false)  -> (sourceManaged in Compile).value / "protobuf",
      grpcmonix.generators.gen() -> (sourceManaged in Compile).value / "protobuf"
    ),
    buildInfoKeys := Seq[BuildInfoKey](name, version, scalaVersion, sbtVersion, git.gitHeadCommit),
    buildInfoPackage := "coop.rchain.node",
    mainClass in Compile := Some("coop.rchain.node.Main"),
    discoveredMainClasses in Compile := Seq(),
    mainClass in assembly := Some("coop.rchain.node.Main"),
    assemblyMergeStrategy in assembly := {
      case x if x.endsWith("io.netty.versions.properties") => MergeStrategy.first
      case x if x.endsWith("scala/annotation/nowarn.class") => MergeStrategy.discard
      case x if x.endsWith("scala/annotation/nowarn$.class") => MergeStrategy.discard
      case x if x.endsWith("module-info.class") => MergeStrategy.discard
      case x =>
        val oldStrategy = (assemblyMergeStrategy in assembly).value
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
    // dockerBaseImage := "ghcr.io/graalvm/jdk:ol8-java17-22.3.3",
    dockerBaseImage := "azul/zulu-openjdk:17.0.9-jre-headless", // Using this image because resolves error of GLIB_C version for rspace++
    dockerEntrypoint := List("/opt/docker/bin/rnode", "--profile=docker", "-XX:ErrorFile=/var/lib/rnode/hs_err_pid%p.log"),
    daemonUserUid in Docker := None,
    daemonUser in Docker := "daemon",
    dockerExposedPorts := List(40400, 40401, 40402, 40403, 40404),
    dockerBuildOptions := Seq(
        "--builder",
        "default",
        "--platform",
        "linux/amd64,linux/arm64",
        "-t",
        "f1r3flyindustries/f1r3fly-scala-node:latest"
    ),
    dockerCommands ++= {
      Seq(
        Cmd("LABEL", s"""MAINTAINER="${maintainer.value}""""),
        Cmd("LABEL", s"""version="${version.value}""""),
        Cmd("USER", "root"),
        Cmd("USER", (Docker / daemonUser).value),
        Cmd(
          "HEALTHCHECK CMD",
          """grpcurl -plaintext 127.0.0.1:40401 casper.v1.DeployService.status | jq -e && \
                                  curl -s 127.0.0.1:40403/status | jq -e"""
        ),
        ExecCmd("CMD", "run")
      )
    },
    Universal / javaOptions ++= List(
      "-J--add-opens",
      "-Jjava.base/sun.security.util=ALL-UNNAMED",
      "-J--add-opens",
      "-Jjava.base/java.nio=ALL-UNNAMED",
      "-J--add-opens",
      "-Jjava.base/sun.nio.ch=ALL-UNNAMED",
      "-J-Xms6G -J-Xmx8G -J-Xss256m -J-XX:MaxMetaspaceSize=3G"
    ),
    // Replace unsupported character `+`
    version in Docker := { version.value.replace("+", "__") },
    mappings in Docker ++= {
      val base = (defaultLinuxInstallLocation in Docker).value
      directory((baseDirectory in rholang).value / "examples")
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
    publishArtifact in (Compile, packageDoc) := false,
    publishArtifact in packageDoc := false,
    sources in (Compile, doc) := Seq.empty,
    libraryDependencies ++= commonDependencies ++ Seq(
      catsMtl,
      catsEffect,
      monix,
      scallop,
      lightningj,
      catsLawsTest,
      catsLawsTestkitTest,
      catsMtlLawsTest
    ),
    // TODO: investigate if still needed?
    // mainClass in assembly := Some("coop.rchain.rho2rose.Rholang2RosetteCompiler"),
    //constrain the resource usage so that we hit SOE-s and OOME-s more quickly should they happen
    javaOptions in Test ++= Seq("-Xss240k", "-XX:MaxJavaStackTraceDepth=10000", "-Xmx128m")
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
    mainClass in assembly := Some("coop.rchain.rholang.interpreter.RholangCLI"),
    assemblyMergeStrategy in assembly := {
      case path if path.endsWith("module-info.class") => MergeStrategy.discard
      case path                                       => MergeStrategy.defaultMergeStrategy(path)
    }
  )
  .dependsOn(rholang)

lazy val rholangServer = (project in file("rholang-server"))
  .enablePlugins(NativeImagePlugin)
  .settings(commonSettings)
  .settings(
    nativeImageJvm := "graalvm-java17",
    nativeImageVersion := "22.3.3",
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
lazy val rspacePlusPlus = (project in file("rspace++"))
  .settings(commonSettings: _*)
  .settings(
    name := "rspace++",
    // mainClass := Some("BuildRustLibrary"),
    dependencyOverrides += "org.scalactic" %% "scalactic" % "3.2.15",
    dependencyOverrides += "org.scalatest" %% "scalatest" % "3.2.15" % "test",
    libraryDependencies ++= commonDependencies ++ kamonDependencies ++ Seq(
      "net.java.dev.jna" % "jna" % "5.13.0",
      circeParser,
      circeGenericExtras
    ),
    PB.targets in Compile := Seq(
      scalapb.gen(grpc = true) -> (sourceManaged in Compile).value / "protobuf"
    )
  )

lazy val rspace = (project in file("rspace"))
  .configs(IntegrationTest extend Test)
  .enablePlugins(SiteScaladocPlugin, GhpagesPlugin)
  .settings(commonSettings: _*)
  .settings(
    scalacOptions ++= Seq(
//      "-Xfatal-warnings"
    ),
    Defaults.itSettings,
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
    publishArtifact in Test := false,
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
    sourceDirectory in Jmh := (sourceDirectory in Test).value,
    classDirectory in Jmh := (classDirectory in Test).value,
    dependencyClasspath in Jmh := (dependencyClasspath in Test).value,
    // rewire tasks, so that 'jmh:run' automatically invokes 'jmh:compile' (otherwise a clean 'jmh:run' would fail),
    compile in Jmh := (compile in Jmh).dependsOn(compile in Test).value,
    run in Jmh := (run in Jmh).dependsOn(Keys.compile in Jmh).evaluated
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
    rspacePlusPlus,
    shared
  )

lazy val runCargoBuild = taskKey[Unit]("Builds Rust library for rspace++")
runCargoBuild := {
  import scala.sys.process._
  Seq("./scripts/build_rspace++.sh") !
}

(compile in Compile) := ((compile in Compile) dependsOn runCargoBuild).value
