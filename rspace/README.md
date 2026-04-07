# RSpace (Scala)

This directory contains the original Scala implementation of the RSpace tuple space library. The active Rust implementation is in [`rspace++/`](../rspace++/README.md).

## Overview

RSpace is a disk-backed [tuple space](https://en.wikipedia.org/wiki/Tuple_space) for the Rholang interpreter. It departs from traditional key-value stores:

- Data is associated with **channels**, not keys
- **Continuations** represent actions to carry out when matching data is found
- A continuation is associated with a list of channels and **patterns**
- The two main operations are:
  - **consume**: searches the store for data matching patterns at given channels
  - **produce**: given data at a channel, searches for a matching continuation

For the current Rust implementation, see [rspace++/](../rspace++/README.md).

## Documentation

- [RSpace Module Overview](../docs/rspace/README.md) — Tuple space engine, produce/consume matching, trie history
