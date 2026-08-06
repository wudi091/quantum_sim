package quantum

import quantum.algorithm.*
import quantum.topo.Topo
import utils.*
import java.io.BufferedWriter
import java.io.File
import java.io.PrintStream
import java.io.Writer
import kotlin.math.pow

/**
 * Headless reproduction runner for the parameter sweeps used by Figures 16--20.
 *
 * This deliberately calls the author's Algorithm implementations directly; it
 * only supplies deterministic request sets, aggregates work()'s returned ebit
 * count, and writes machine-readable output.  No routing logic is changed.
 * Use QCAST_SWEEP_SLOTS/QCAST_SWEEP_TOPOLOGIES to make a short smoke run.
 */
class OfficialSweep {
  private class DiscardWriter : Writer() {
    override fun write(cbuf: CharArray, off: Int, len: Int) {}
    override fun flush() {}
    override fun close() {}
  }

  data class Point(val fig: String, val parameter: String, val value: String,
                   val n: Int = 100, val p: Double = 0.6,
                   val q: Double = 0.9, val k: Int = 3, val m: Int = 10)

  data class Aggregate(val fig: String, val parameter: String, val value: String,
                       val algorithm: String, val n: Int, val p: Double,
                       val q: Double, val k: Int, val m: Int,
                       val slots: Int, val topologies: Int,
                       val throughput: Double, val successPairs: Double)

  companion object {
    private val algorithms = listOf("Online", "CR", "Greedy_H", "SL")

    private fun envInt(name: String, fallback: Int): Int =
      System.getenv(name)?.toIntOrNull()?.takeIf { it > 0 } ?: fallback

    private fun alphaFor(topo: Topo, expected: Double): Double =
      dynSearch(1E-10, 1.0, expected, { x ->
        topo.links.map { Math.E.pow(-x * +(it.n1.loc - it.n2.loc)) }.average()
      }, false, 0.001)

    /** Replace the first four topology header lines while preserving all RNG data. */
    private fun configure(base: Topo, p: Double, q: Double, k: Int): Topo {
      val alpha = alphaFor(base, p)
      val lines = base.toString().lines().mapIndexed { i, line ->
        when (i) { 1 -> alpha.toString(); 2 -> q.toString(); 3 -> k.toString(); else -> line }
      }
      return Topo(lines.joinToString("\n"))
    }

    private fun makeAlgorithm(name: String, topo: Topo): Algorithm = when (name) {
      "Online" -> OnlineAlgorithm(topo)
      "CR" -> CreationRate(topo)
      "Greedy_H" -> GreedyHopRouting(topo)
      "SL" -> SingleLink(topo)
      else -> error("unknown algorithm $name")
    }

    private fun points(fig: String): List<Point> = when (fig.toLowerCase()) {
      "k", "fig16", "16" -> listOf(0, 3, 6, 10000).map { Point("Fig16", "k", it.toString(), k = it) }
      "p", "fig17", "17" -> listOf(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
        .map { Point("Fig17", "p", it.toString(), p = it) }
      "q", "fig18", "18" -> listOf(0.8,0.85,0.9,0.95,1.0)
        .map { Point("Fig18", "q", it.toString(), q = it) }
      "n", "fig19", "19" -> listOf(50,100,200,400,800)
        .map { Point("Fig19", "n", it.toString(), n = it) }
      "m", "fig20", "20" -> (1..10).map { Point("Fig20", "m", it.toString(), m = it) }
      "all", "" -> (listOf("k", "p", "q", "n", "m")).flatMap { points(it) }
      else -> error("QCAST_SWEEP_FIG must be k,p,q,n,m,all (got $fig)")
    }

    private fun requests(topo: Topo, m: Int): List<Pair<Int, Int>> {
      val n = topo.nodes.size
      return (0 until n).shuffled(randGen).take(2 * m).chunked(2).map { it[0] to it[1] }
    }

    private fun json(rows: List<Aggregate>): String {
      fun esc(s: String) = s.replace("\\", "\\\\").replace("\"", "\\\"")
      return rows.joinToString(prefix = "[\n", postfix = "\n]\n", separator = ",\n") { r ->
        """  {"fig":"${esc(r.fig)}","parameter":"${esc(r.parameter)}","value":"${esc(r.value)}","algorithm":"${esc(r.algorithm)}","n":${r.n},"p":${r.p},"q":${r.q},"k":${r.k},"m":${r.m},"slots":${r.slots},"topologies":${r.topologies},"throughput":${r.throughput},"success_pairs":${r.successPairs}}"""
      }
    }

    @JvmStatic
    fun main(args: Array<String>) {
      visualize = false
      val fig = System.getenv("QCAST_SWEEP_FIG") ?: args.firstOrNull() ?: "all"
      val slots = envInt("QCAST_SWEEP_SLOTS", 1000)
      val topologyCount = envInt("QCAST_SWEEP_TOPOLOGIES", 10)
      val valueFilter = System.getenv("QCAST_SWEEP_VALUES")?.split(',')
        ?.map { it.trim() }?.filter { it.isNotEmpty() }?.toSet()
      val selected = points(fig).filter { valueFilter == null || it.value in valueFilter }
      require(selected.isNotEmpty()) { "QCAST_SWEEP_VALUES selected no points for $fig" }
      val rows = mutableListOf<Aggregate>()
      // The author's sim() creates one fixed topology set per |V| and reuses
      // it for every parameter value.  Pre-generate that same design here so
      // curves differ only by the swept parameter (and request RNG).
      val baseTopologies = selected.map { it.n }.distinct().associateWith { n ->
        (1..topologyCount).map { Topo.generate(n, 0.9, 3, 0.1, 6) }
      }
      val originalOut = System.out
      // Algorithm.work prints one line per slot; keep the runner's stdout useful.
      System.setOut(PrintStream(java.io.OutputStream.nullOutputStream()))
      try {
        selected.forEachIndexed { pointIndex, point ->
          val sums = algorithms.associateWith { doubleArrayOf(0.0, 0.0) }.toMutableMap()
          repeat(topologyCount) { topoIndex ->
            val n = point.n
            val generated = baseTopologies[n]!![topoIndex]
            val configured = configure(generated, point.p, point.q, point.k)
            val testSet = (1..slots).map { requests(configured, point.m) }
            algorithms.forEach { algorithmName ->
              val solver = makeAlgorithm(algorithmName, Topo(configured.toString()))
              solver.logWriter = BufferedWriter(DiscardWriter())
              var ebitSum = 0.0
              var pairSum = 0.0
              testSet.forEach { ids ->
                val pairs = ids.map { solver.topo.nodes[it.first] to solver.topo.nodes[it.second] }
                val result = solver.work(pairs)
                pairSum += result.first
                ebitSum += result.second
              }
              solver.logWriter.close()
              sums[algorithmName]!![0] += ebitSum
              sums[algorithmName]!![1] += pairSum
            }
          }
          algorithms.forEach { algorithmName ->
            val total = slots.toDouble() * topologyCount
            val s = sums[algorithmName]!!
            rows += Aggregate(point.fig, point.parameter, point.value, algorithmName,
              point.n, point.p, point.q, point.k, point.m, slots, topologyCount,
              s[0] / total, s[1] / total)
          }
          System.err.println("completed ${point.fig} ${point.parameter}=${point.value} (${pointIndex + 1}/${selected.size})")
        }
      } finally {
        System.setOut(originalOut)
      }

      val outDir = File("../results/qcast_paper").also { it.mkdirs() }
      val valueTag = valueFilter?.let { "_values-" + it.joinToString("-") { v -> v.replace('.', 'p') } } ?: ""
      val stem = "official_sweep_${fig.toLowerCase()}_${slots}slots_${topologyCount}topos$valueTag"
      File(outDir, "$stem.json").writeText(json(rows))
      File(outDir, "$stem.csv").writeText(buildString {
        append("fig,parameter,value,algorithm,n,p,q,k,m,slots,topologies,throughput,success_pairs\n")
        rows.forEach { r -> append(listOf(r.fig,r.parameter,r.value,r.algorithm,r.n,r.p,r.q,r.k,r.m,r.slots,r.topologies,r.throughput,r.successPairs).joinToString(",")).append("\n") }
      })
      println("QCAST_SWEEP_FIG=$fig")
      println("QCAST_SWEEP_SLOTS=$slots")
      println("QCAST_SWEEP_TOPOLOGIES=$topologyCount")
      println("QCAST_SWEEP_JSON=${File(outDir, "$stem.json").canonicalPath}")
      rows.groupBy { it.fig to it.algorithm }.forEach { (key, values) ->
        println("QCAST_SWEEP_${key.first}_${key.second}=" + values.joinToString(";") { "${it.value}:${"%.6f".format(it.throughput)}" })
      }
    }
  }
}
