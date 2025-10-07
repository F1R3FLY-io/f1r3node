addSbtPlugin("com.thesamet" % "sbt-protoc" % "1.0.6")
// Yes it's weird to do the following, but it's what is mandated by the scalapb documentation
libraryDependencies += "com.thesamet.scalapb" %% "compilerplugin" % "0.11.11"
libraryDependencies += "net.java.dev.jna"     % "jna"             % "5.13.0"
libraryDependencies += "net.java.dev.jna"     % "jna-platform"    % "5.13.0"

addSbtPlugin("com.typesafe.sbt"       % "sbt-license-report"   % "1.2.0")
addSbtPlugin("org.wartremover"        % "sbt-wartremover"      % "3.1.5")
addSbtPlugin("org.scalameta"          % "sbt-scalafmt"         % "2.4.0")
addSbtPlugin("com.eed3si9n"           % "sbt-assembly"         % "2.1.3")
addSbtPlugin("com.github.tkawachi"    % "sbt-repeat"           % "0.1.0")
addSbtPlugin("com.eed3si9n"           % "sbt-buildinfo"        % "0.10.0")
addSbtPlugin("com.github.sbt"         % "sbt-native-packager"  % "1.11.1")
addSbtPlugin("pl.project13.scala"     % "sbt-jmh"              % "0.4.0")

addSbtPlugin("com.github.sbt"         % "sbt-site-paradox"     % "1.5.0")
addSbtPlugin("com.github.sbt"         % "sbt-ghpages"          % "0.8.0")
addSbtPlugin("com.github.sbt"         % "sbt-pgp"              % "2.2.1")
addSbtPlugin("org.xerial.sbt"         % "sbt-sonatype"         % "3.9.21")

addSbtPlugin("io.spray"               % "sbt-revolver"         % "0.9.1")
addSbtPlugin("com.sksamuel.scapegoat" %% "sbt-scapegoat"       % "1.2.2")

addSbtPlugin("com.github.sbt"         % "sbt-release"          % "1.1.0")

addSbtPlugin("org.scalameta"          % "sbt-native-image"     % "0.3.2")

addDependencyTreePlugin

