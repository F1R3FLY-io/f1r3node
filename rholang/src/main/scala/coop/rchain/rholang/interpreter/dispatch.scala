package coop.rchain.rholang.interpreter

import cats.Parallel
import cats.effect.Sync
import cats.effect.concurrent.Ref
import cats.syntax.functor._
import coop.rchain.crypto.hash.Blake2b512Random
import coop.rchain.models.TaggedContinuation.TaggedCont.{Empty, ParBody, ScalaBodyRef}
import coop.rchain.models._
import coop.rchain.rholang.interpreter.Dispatch.DispatchType
import coop.rchain.rholang.interpreter.RhoRuntime.RhoTuplespace
import coop.rchain.rholang.interpreter.accounting._

trait Dispatch[M[_], A, K] {
  def dispatch(continuation: K, dataList: Seq[A], isReplay: Boolean, previousOutput: Option[Any]): M[DispatchType]
}

object Dispatch {
  def buildEnv(dataList: Seq[ListParWithRandom]): Env[Par] =
    Env.makeEnv(dataList.flatMap(_.pars): _*)

  sealed trait DispatchType {
    def asParentType: DispatchType = this
  }
  case class NonDeterministicCall(output: Any) extends DispatchType
  case object DeterministicCall extends DispatchType
  case object Skip extends DispatchType
}

class RholangAndScalaDispatcher[M[_]] private (
    _dispatchTable: => Map[Long, (Seq[ListParWithRandom], Boolean, Option[Any]) => M[Any]]
)(implicit s: Sync[M], reducer: Reduce[M])
    extends Dispatch[M, ListParWithRandom, TaggedContinuation] {

  def dispatch(
      continuation: TaggedContinuation,
      dataList: Seq[ListParWithRandom],
      isReplay: Boolean,
      previousOutput: Option[Any]
  ): M[DispatchType] =
    continuation.taggedCont match {
      case ParBody(parWithRand) =>
        val env     = Dispatch.buildEnv(dataList)
        val randoms = parWithRand.randomState +: dataList.toVector.map(_.randomState)
        reducer.eval(parWithRand.body)(env, Blake2b512Random.merge(randoms))
          .map(_ => Dispatch.DeterministicCall)

      case ScalaBodyRef(ref) =>

        val isNonDeterministicCall = SystemProcesses.nonDeterministicCalls.contains(ref)

        _dispatchTable.get(ref) match {
          case Some(f) =>
            f(dataList, isReplay, previousOutput)
              .map(o => dispatchType(ref, isNonDeterministicCall, o))
          case None => s.raiseError(new Exception(s"dispatch: no function for $ref"))
        }
      case Empty =>
        Sync[M].delay(Dispatch.Skip)
    }

  private def dispatchType(id: Long, isNonDeterministicCall: Boolean, output: Any): DispatchType = {
    if (isNonDeterministicCall) Dispatch.NonDeterministicCall(output) else Dispatch.DeterministicCall
  }
}

object RholangAndScalaDispatcher {
  type RhoDispatch[F[_]] = Dispatch[F, ListParWithRandom, TaggedContinuation]

  def apply[M[_]: Sync: Parallel: _cost](
      tuplespace: RhoTuplespace[M],
      dispatchTable: => Map[Long, (Seq[ListParWithRandom], Boolean, Option[Any]) => M[Any]],
      urnMap: Map[String, Par],
      mergeChs: Ref[M, Set[Par]],
      mergeableTagName: Par
  ): (Dispatch[M, ListParWithRandom, TaggedContinuation], Reduce[M]) = {

    implicit lazy val dispatcher: Dispatch[M, ListParWithRandom, TaggedContinuation] =
      new RholangAndScalaDispatcher(dispatchTable)

    implicit lazy val reducer: Reduce[M] =
      new DebruijnInterpreter[M](tuplespace, dispatcher, urnMap, mergeChs, mergeableTagName)

    (dispatcher, reducer)
  }

}
