package quantum

import quantum.algorithm.OnlineAlgorithm
import quantum.topo.Topo
import utils.*
import java.io.BufferedWriter
import java.io.FileWriter
import kotlin.math.pow

/** Bounded, headless reference-setting gold run using the author simulator. */
class OfficialReferenceSmoke {
  companion object {
    @JvmStatic
    fun main(args: Array<String>) {
      visualize = false
      val slots = System.getenv("QCAST_SMOKE_SLOTS")?.toIntOrNull() ?: 100
      val n = 100
      val d = 6
      val p = 0.6
      val q = 0.9
      val k = 3
      val pairs = 10
      val base = Topo.generate(n, 0.9, 5, 0.1, d)
      val alpha = dynSearch(1E-10, 1.0, p, { x ->
        base.links.map { Math.E.pow(-x * +(it.n1.loc - it.n2.loc)) }.average()
      }, false, 0.001)
      base.alpha = alpha
      base.q = q
      base.k = k
      val solver = OnlineAlgorithm(Topo(base))
      solver.logWriter = BufferedWriter(FileWriter("dist/reference-smoke-Online.txt"))
      val throughput = (1..slots).map {
        val requests = (0 until n).shuffled(randGen).take(2 * pairs)
          .chunked(2).map { it.toPair() }
          .map { Pair(solver.topo.nodes[it.first], solver.topo.nodes[it.second]) }
        solver.work(requests).second
      }
      solver.logWriter.close()
      println("QCAST_REFERENCE_SLOTS=$slots")
      println("QCAST_REFERENCE_MEAN=${throughput.average()}")
      println("QCAST_REFERENCE_VALUES=$throughput")
      println(base.getStatistics())
    }
  }
}
