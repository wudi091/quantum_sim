package quantum

/** Headless entry point for a bounded author-code reproduction smoke run. */
class OfficialSmoke {
  companion object {
    @JvmStatic
    fun main(args: Array<String>) {
      visualize = false
      simpleTest()
    }
  }
}
