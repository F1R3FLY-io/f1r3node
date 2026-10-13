# Regex

> **Deprecated:** this repository is no longer maintained. Use [f1r3node-rust](https://github.com/F1R3FLY-io/f1r3node-rust) instead.

FSM and Regex libraries for the RChain blockchain.

### Prerequisites

* [sbt](http://www.scala-sbt.org/download.html)

### Building

```
sbt compile
```

### Testing

```
sbt test
```

Testing with coverage:

```
sbt clean coverage test
```

Generating a coverage report:

```
sbt coverageReport
```

The HTML version of the generated report is located at:

 `./target/scala-<version>/scoverage-report/index.html`