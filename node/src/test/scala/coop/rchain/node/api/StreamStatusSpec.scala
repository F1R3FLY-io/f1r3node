package coop.rchain.node.api

import coop.rchain.casper.protocol._
import coop.rchain.casper.protocol.deploy.v1._
import monix.eval.Task
import monix.reactive.Observable
import org.scalatest.{FlatSpec, Matchers}

/**
  * Placeholder for future streamStatus integration test.
  * The `streamStatus` RPC and its proto types are not yet available in DeployServiceV1.
  */
class StreamStatusSpec extends FlatSpec with Matchers {

  "StreamStatusSpec" should "be implemented when streamStatus proto is available" in {
    pending
  }
}
