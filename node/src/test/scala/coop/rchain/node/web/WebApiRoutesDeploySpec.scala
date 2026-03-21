package coop.rchain.node.web

import cats.effect.IO
import com.google.protobuf.ByteString
import coop.rchain.casper.protocol.{BlockInfo, LightBlockInfo}
import coop.rchain.node.api.WebApi
import coop.rchain.node.api.WebApi._
import coop.rchain.shared.Log
import io.circe.parser._
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import org.http4s._
import org.http4s.implicits._
import org.scalatest.{FlatSpec, Matchers}

class WebApiRoutesDeploySpec extends FlatSpec with Matchers {

  implicit val log: Log[Task] = Log.log[Task]

  val sampleLightBlockInfo: LightBlockInfo = LightBlockInfo(
    blockHash = "7bf8abc123",
    sender = "0487def456",
    seqNum = 17453,
    sig = "3044abcdef",
    sigAlgorithm = "secp256k1",
    shardId = "root",
    extraBytes = ByteString.EMPTY,
    version = 1,
    timestamp = 1770028092477L,
    headerExtraBytes = ByteString.EMPTY,
    parentsHashList = List("parent1hash", "parent2hash"),
    blockNumber = 52331,
    preStateHash = "preState123",
    postStateHash = "postState456",
    bodyExtraBytes = ByteString.EMPTY,
    bonds = List(
      coop.rchain.casper.protocol.BondInfo("validator1", 100),
      coop.rchain.casper.protocol.BondInfo("validator2", 200)
    ),
    blockSize = "4096",
    deployCount = 5,
    faultTolerance = 0.5f,
    justifications = List(
      coop.rchain.casper.protocol.JustificationInfo("validator1", "latestBlockHash1")
    ),
    rejectedDeploys = List.empty
  )

  val sampleDeployLookupResponse: DeployLookupResponse =
    DeployLookupResponse.fromLightBlockInfo(sampleLightBlockInfo)

  /** Stub WebApi that only implements findDeploy and findDeployMinimal */
  val stubWebApi: WebApi[Task] = new WebApi[Task] {
    def status: Task[ApiStatus] = Task.raiseError(new NotImplementedError)
    def prepareDeploy(request: Option[PrepareRequest]): Task[PrepareResponse] =
      Task.raiseError(new NotImplementedError)
    def deploy(request: DeployRequest): Task[String] = Task.raiseError(new NotImplementedError)
    def listenForDataAtName(request: DataAtNameRequest): Task[DataAtNameResponse] =
      Task.raiseError(new NotImplementedError)
    def getDataAtPar(request: DataAtNameByBlockHashRequest): Task[RhoDataResponse] =
      Task.raiseError(new NotImplementedError)
    def lastFinalizedBlock: Task[BlockInfo]               = Task.raiseError(new NotImplementedError)
    def getBlock(hash: String): Task[BlockInfo]           = Task.raiseError(new NotImplementedError)
    def getBlocks(depth: Int): Task[List[LightBlockInfo]] = Task.raiseError(new NotImplementedError)
    def exploratoryDeploy(
        term: String,
        blockHash: Option[String],
        usePreStateHash: Boolean
    ): Task[RhoDataResponse] = Task.raiseError(new NotImplementedError)
    def getBlocksByHeights(
        startBlockNumber: Long,
        endBlockNumber: Long
    ): Task[List[LightBlockInfo]] =
      Task.raiseError(new NotImplementedError)
    def isFinalized(hash: String): Task[Boolean] = Task.raiseError(new NotImplementedError)
    def getTransaction(hash: String): Task[TransactionResponse] =
      Task.raiseError(new NotImplementedError)

    def findDeploy(deployId: String): Task[LightBlockInfo] =
      Task.pure(sampleLightBlockInfo)

    def findDeployMinimal(deployId: String): Task[DeployLookupResponse] =
      Task.pure(sampleDeployLookupResponse)
  }

  val routes: HttpRoutes[Task] = WebApiRoutes.service[Task](stubWebApi)

  private def runRequest(req: Request[Task]): Response[Task] =
    routes.orNotFound.run(req).runSyncUnsafe()

  private def bodyAsString(resp: Response[Task]): String =
    resp.body.compile.toVector.runSyncUnsafe().map(_.toChar).mkString

  "GET /deploy/<id>" should "return full LightBlockInfo when no view param" in {
    val req  = Request[Task](Method.GET, uri"/deploy/abc123def")
    val resp = runRequest(req)

    resp.status should be(Status.Ok)

    val json = parse(bodyAsString(resp)).getOrElse(fail("Invalid JSON"))

    // Full response should contain block-level fields
    json.hcursor.get[String]("blockHash").toOption should be(Some("7bf8abc123"))
    json.hcursor.get[Long]("blockNumber").toOption should be(Some(52331L))
    json.hcursor.get[Long]("timestamp").toOption should be(Some(1770028092477L))
    // Should contain fields that minimal view excludes
    json.hcursor.get[String]("preStateHash").toOption should be(Some("preState123"))
    json.hcursor.get[String]("postStateHash").toOption should be(Some("postState456"))
    json.hcursor.downField("bonds").focus should be(defined)
    json.hcursor.downField("justifications").focus should be(defined)
    json.hcursor.downField("parentsHashList").focus should be(defined)
  }

  it should "return minimal DeployLookupResponse when view=minimal" in {
    val req  = Request[Task](Method.GET, uri"/deploy/abc123def?view=minimal")
    val resp = runRequest(req)

    resp.status should be(Status.Ok)

    val json = parse(bodyAsString(resp)).getOrElse(fail("Invalid JSON"))

    // Minimal response should contain only deploy-centric fields
    json.hcursor.get[String]("blockHash").toOption should be(Some("7bf8abc123"))
    json.hcursor.get[Long]("blockNumber").toOption should be(Some(52331L))
    json.hcursor.get[Long]("timestamp").toOption should be(Some(1770028092477L))
    json.hcursor.get[String]("sender").toOption should be(Some("0487def456"))
    json.hcursor.get[Long]("seqNum").toOption should be(Some(17453L))
    json.hcursor.get[String]("sig").toOption should be(Some("3044abcdef"))
    json.hcursor.get[String]("sigAlgorithm").toOption should be(Some("secp256k1"))
    json.hcursor.get[String]("shardId").toOption should be(Some("root"))
    json.hcursor.get[Long]("version").toOption should be(Some(1L))

    // Should NOT contain block-level fields
    json.hcursor.downField("bonds").focus should be(None)
    json.hcursor.downField("justifications").focus should be(None)
    json.hcursor.downField("parentsHashList").focus should be(None)
    json.hcursor.downField("preStateHash").focus should be(None)
    json.hcursor.downField("postStateHash").focus should be(None)
    json.hcursor.downField("faultTolerance").focus should be(None)
    json.hcursor.downField("deployCount").focus should be(None)
    json.hcursor.downField("blockSize").focus should be(None)
  }

  it should "return full LightBlockInfo when view param has unknown value" in {
    val req  = Request[Task](Method.GET, uri"/deploy/abc123def?view=unknown")
    val resp = runRequest(req)

    resp.status should be(Status.Ok)

    val json = parse(bodyAsString(resp)).getOrElse(fail("Invalid JSON"))

    // Unknown view value should fall back to full response
    json.hcursor.downField("bonds").focus should be(defined)
    json.hcursor.downField("justifications").focus should be(defined)
  }
}
