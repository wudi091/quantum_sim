package quantum

import quantum.algorithm.*
import quantum.topo.Topo
import utils.*
import java.io.BufferedWriter
import java.io.FileWriter
import kotlin.math.pow

/** Bounded reference comparison for the four algorithms in Figures 16--20. */
class OfficialComparisonSmoke {
  companion object {
    @JvmStatic
    fun main(args: Array<String>) {
      visualize = false
      val slots = System.getenv("QCAST_SMOKE_SLOTS")?.toIntOrNull() ?: 300
      val n = 100
      val base = Topo.generate(n, 0.9, 3, 0.1, 6)
      base.alpha = dynSearch(1E-10, 1.0, 0.6, { x ->
        base.links.map { Math.E.pow(-x * +(it.n1.loc - it.n2.loc)) }.average()
      }, false, 0.001)
      base.q = 0.9
      base.k = 3
      val testSet = (1..slots).map {
        (0 until n).shuffled(randGen).take(20).chunked(2).map { it.toPair() }
      }
      val algorithms = listOf(
        OnlineAlgorithm(Topo(base)), CreationRate(Topo(base)),
        GreedyHopRouting(Topo(base)), SingleLink(Topo(base))
      )
      algorithms.forEach { solver ->
        solver.logWriter = BufferedWriter(FileWriter("dist/reference-compare-${solver.name}.txt"))
        val values = testSet.map { ids ->
          solver.work(ids.map { Pair(solver.topo.nodes[it.first], solver.topo.nodes[it.second]) }).second
        }
        solver.logWriter.close()
        println("QCAST_COMPARE_${solver.name}=${values.average()}")
      }
      println("QCAST_COMPARE_SLOTS=$slots")
    }
  }
}
