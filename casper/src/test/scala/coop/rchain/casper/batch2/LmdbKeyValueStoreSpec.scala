package coop.rchain.casper.batch2

import java.nio.file.Files

import cats.effect.Concurrent
import cats.syntax.all._
import cats.effect.Resource
import coop.rchain.shared.Log
import coop.rchain.store.{KeyValueStoreSut, LmdbStoreManager}
import monix.eval.Task
import org.scalacheck.{Arbitrary, Gen, Shrink}
import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

import scala.reflect.io.{Directory, Path}
import scala.util.Random

class LmdbKeyValueStoreSpec
    extends FlatSpec
    with Matchers
    with GeneratorDrivenPropertyChecks
    with BeforeAndAfterAll {
  implicit val scheduler              = monix.execution.Scheduler.global
  implicit def noShrink[A]: Shrink[A] = Shrink(_ => Stream.empty)

  val tempPath = Files.createTempDirectory(s"lmdb-test-")
  val tempDir  = Directory(Path(tempPath.toFile))

  override def beforeAll: Unit = tempDir.deleteRecursively

  override def afterAll: Unit = tempDir.deleteRecursively

  def withSut[F[_]: Concurrent: Log](f: KeyValueStoreSut[F] => F[Unit]) =
    Resource
      .make(
        LmdbStoreManager[F](
          tempPath.resolve(Random.alphanumeric.take(10).mkString),
          1024 * 1024 * 1024
        )
      )(_.shutdown)
      .use { implicit kvm =>
        val sut = new KeyValueStoreSut[F]
        f(sut)
      }

  def genData: Gen[Map[Long, String]] = {
    val arbKV = for {
      arbK <- Gen.long
      arbV <- Gen.alphaNumStr.suchThat(_.size > 0)
    } yield (arbK, arbV)

    Gen.nonEmptyMap(arbKV)
  }

  implicit val log: Log[Task] = new Log.NOPLog[Task]()

  it should "put and get data from the store" in {
    forAll(genData) { expected =>
      val test = withSut[Task] { sut =>
        for {
          result <- sut.testPutGet(expected)
        } yield result shouldBe expected
      }

      test.runSyncUnsafe()
    }
  }

  it should "put and get all data from the store" in {
    forAll(genData) { expected =>
      val test = withSut[Task] { sut =>
        for {
          result <- sut.testPutIterate(expected)
        } yield result shouldBe expected
      }

      test.runSyncUnsafe()
    }
  }

  it should "not have deleted keys in the store" in {
    forAll(genData) { input =>
      val test = withSut[Task] { sut =>
        val allKeys = input.keysIterator.toVector
        // Take some keys for deletion
        val (getKeys, deleteKeys) = allKeys.splitAt(allKeys.size / 2)
        val values                = getKeys.map(input.get)
        // Expected input without deleted keys
        val expected =
          getKeys.zip(values).filter(_._2.nonEmpty).map { case (k, v) => (k, v.get) }.toMap
        for {
          result <- sut.testPutDeleteGet(input, deleteKeys)
        } yield result shouldBe expected
      }

      test.runSyncUnsafe()
    }
  }

}
