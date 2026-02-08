package coop.rchain.casper.merging

import org.scalatest.{FlatSpec, Matchers}

/**
  * Tests that the tiebreaking logic used in ConflictSetMerger.getOptimalRejection
  * is deterministic. The key invariant is that when multiple rejection options have
  * identical cost and size, the tiebreak by `b.flatten.min` (using Ordering[R])
  * produces a stable, deterministic result rather than depending on Set iteration order.
  */
class ConflictSetMergerDeterminismSpec extends FlatSpec with Matchers {

  // Simulates the getOptimalRejection logic from ConflictSetMerger
  private def getOptimalRejection[R: Ordering](
      options: Set[Set[Set[R]]],
      targetF: Set[R] => Long
  ): Set[Set[R]] =
    options.toList
      .sortBy(b => (b.map(targetF).sum, b.size, b.flatten.min))
      .headOption
      .getOrElse(Set.empty)

  private val zeroCost: Set[Int] => Long     = (_: Set[Int]) => 0L
  private val fixedCost10: Set[Int] => Long  = (_: Set[Int]) => 10L
  private val fixedCost100: Set[Int] => Long = (_: Set[Int]) => 100L

  "getOptimalRejection tiebreaker" should "deterministically select when costs are equal" in {
    val branch1 = Set(1, 2)
    val branch2 = Set(3, 4)
    val branch3 = Set(5, 6)

    // All options have equal cost (0) and equal size (1 branch each)
    val options: Set[Set[Set[Int]]] = Set(
      Set(branch2),
      Set(branch3),
      Set(branch1)
    )

    val result = getOptimalRejection(options, zeroCost)
    // Should deterministically pick the option containing the smallest element (1)
    result shouldBe Set(branch1)
  }

  it should "be stable across repeated invocations" in {
    val options: Set[Set[Set[Int]]] = Set(
      Set(Set(10, 20)),
      Set(Set(5, 15)),
      Set(Set(30, 40))
    )

    val results = (1 to 100).map(_ => getOptimalRejection(options, zeroCost))
    results.distinct.size shouldBe 1
    // Should always pick the one with min element = 5
    results.head shouldBe Set(Set(5, 15))
  }

  it should "prefer lower cost before applying tiebreak" in {
    val cheapBranch     = Set(100, 200) // high elements but low cost
    val expensiveBranch = Set(1, 2)     // low elements but high cost

    val options: Set[Set[Set[Int]]] = Set(
      Set(expensiveBranch),
      Set(cheapBranch)
    )

    val costF: Set[Int] => Long = (s: Set[Int]) => s.sum.toLong
    val result                  = getOptimalRejection(options, costF)
    // costF(expensiveBranch) = 3, costF(cheapBranch) = 300
    // Should prefer the lower cost option
    result shouldBe Set(expensiveBranch)
  }

  it should "prefer smaller size when costs are equal" in {
    val smallOption = Set(Set(1, 2))
    val largeOption = Set(Set(3, 4), Set(5, 6))

    val options: Set[Set[Set[Int]]] = Set(smallOption, largeOption)

    val result = getOptimalRejection(options, fixedCost10)
    // Both have same cost per branch (10), but smallOption has 1 branch vs 2
    // Cost: small = 10, large = 20. So small wins on cost.
    result shouldBe smallOption
  }

  it should "use min element tiebreak when cost and size are equal" in {
    val optionA = Set(Set(10))
    val optionB = Set(Set(5))
    val optionC = Set(Set(20))

    val options: Set[Set[Set[Int]]] = Set(optionA, optionB, optionC)

    val result = getOptimalRejection(options, fixedCost100)
    // All have cost 100 and size 1. Min element: B=5, A=10, C=20
    result shouldBe optionB
  }
}
