package coop.rchain.rholang.externalservices

import cats.effect.Concurrent
import cats.syntax.all._

class DisabledNunetServiceMock extends NunetService {
  def deployEnsemble[F[_]](ensembleYaml: String, timeoutMinutes: Int)(
      implicit
      F: Concurrent[F]
  ): F[String] =
    F.raiseError(new UnsupportedOperationException("NunetService is disabled in tests"))

  def getDeploymentStatus[F[_]](ensembleId: String)(
      implicit
      F: Concurrent[F]
  ): F[DeploymentStatus] =
    F.raiseError(new UnsupportedOperationException("NunetService is disabled in tests"))

  def listDeployments[F[_]]()(implicit F: Concurrent[F]): F[List[DeploymentSummary]] =
    F.raiseError(new UnsupportedOperationException("NunetService is disabled in tests"))

  def getDeploymentLogs[F[_]](ensembleId: String)(implicit F: Concurrent[F]): F[String] =
    F.raiseError(new UnsupportedOperationException("NunetService is disabled in tests"))

  def getDeploymentManifest[F[_]](ensembleId: String)(
      implicit
      F: Concurrent[F]
  ): F[DeploymentManifest] =
    F.raiseError(new UnsupportedOperationException("NunetService is disabled in tests"))

  def generateFireflyEnsemble[F[_]](config: ShardConfig)(implicit F: Concurrent[F]): F[String] =
    F.raiseError(new UnsupportedOperationException("NunetService is disabled in tests"))

  def validateEnsemble[F[_]](ensembleYaml: String)(
      implicit
      F: Concurrent[F]
  ): F[ValidationResult] =
    F.raiseError(new UnsupportedOperationException("NunetService is disabled in tests"))
}

class MockNunetService extends NunetService {
  def deployEnsemble[F[_]](ensembleYaml: String, timeoutMinutes: Int)(
      implicit
      F: Concurrent[F]
  ): F[String] =
    F.pure("mock-ensemble-12345")

  def getDeploymentStatus[F[_]](ensembleId: String)(
      implicit
      F: Concurrent[F]
  ): F[DeploymentStatus] =
    F.pure(
      DeploymentStatus(
        ensembleId = ensembleId,
        status = "running",
        allocations = List(
          AllocationStatus("bootstrap", "running", Some("peer-bootstrap")),
          AllocationStatus("validator1", "running", Some("peer-validator1")),
          AllocationStatus("validator2", "running", Some("peer-validator2"))
        )
      )
    )

  def listDeployments[F[_]]()(implicit F: Concurrent[F]): F[List[DeploymentSummary]] =
    F.pure(
      List(
        DeploymentSummary("ensemble-1", "running", 3),
        DeploymentSummary("ensemble-2", "stopped", 3)
      )
    )

  def getDeploymentLogs[F[_]](ensembleId: String)(implicit F: Concurrent[F]): F[String] =
    F.pure(s"Mock logs for ensemble $ensembleId\nLine 1\nLine 2")

  def getDeploymentManifest[F[_]](ensembleId: String)(
      implicit
      F: Concurrent[F]
  ): F[DeploymentManifest] =
    F.pure(
      DeploymentManifest(
        ensembleId = ensembleId,
        peers = List("peer-bootstrap", "peer-validator1", "peer-validator2"),
        ipAddresses = Map(
          "bootstrap"  -> "192.168.1.10",
          "validator1" -> "192.168.1.11",
          "validator2" -> "192.168.1.12"
        ),
        portMappings = Map(
          "bootstrap" -> List(
            PortMapping(40400, 40400, "bootstrap"),
            PortMapping(40401, 40401, "bootstrap")
          ),
          "validator1" -> List(
            PortMapping(40410, 40400, "validator1"),
            PortMapping(40411, 40401, "validator1")
          )
        )
      )
    )

  def generateFireflyEnsemble[F[_]](config: ShardConfig)(implicit F: Concurrent[F]): F[String] =
    F.pure(s"""version: "1.0"
allocations:
  - name: bootstrap
    image: f1r3flyindustries/f1r3fly-scala-node:latest
    resources:
      cpu: ${config.cpuCores}
      memory: ${config.ramGb}Gi
      disk: ${config.diskGb}Gi
  - name: validator1
    image: f1r3flyindustries/f1r3fly-scala-node:latest
    resources:
      cpu: ${config.cpuCores}
      memory: ${config.ramGb}Gi
      disk: ${config.diskGb}Gi
  - name: validator2
    image: f1r3flyindustries/f1r3fly-scala-node:latest
    resources:
      cpu: ${config.cpuCores}
      memory: ${config.ramGb}Gi
      disk: ${config.diskGb}Gi
""")

  def validateEnsemble[F[_]](ensembleYaml: String)(
      implicit
      F: Concurrent[F]
  ): F[ValidationResult] = {
    val hasVersion     = ensembleYaml.contains("version:")
    val hasAllocations = ensembleYaml.contains("allocations:")

    if (hasVersion && hasAllocations) {
      F.pure(ValidationResult(valid = true, errors = List.empty, warnings = List.empty))
    } else {
      val errors = List(
        if (!hasVersion) Some("Missing 'version' field") else None,
        if (!hasAllocations) Some("Missing 'allocations' field") else None
      ).flatten

      F.pure(ValidationResult(valid = false, errors = errors, warnings = List.empty))
    }
  }
}

object NunetServiceMock {
  lazy val disabledService: NunetService = new DisabledNunetServiceMock
  lazy val mockService: NunetService     = new MockNunetService
}
