# IntegrationTest Deprecation Issue

## Problem

The build uses sbt's `IntegrationTest` configuration which has been deprecated since sbt 1.9.0. This produces the following warnings during build loading:

```
build.sbt:52: warning: value IntegrationTest in trait LibraryManagementSyntax is deprecated (since 1.9.0): Create a separate subproject for testing instead
  IntegrationTest / javaOptions := javaOpens

build.sbt:126-128: warning: value IntegrationTest is deprecated
  IntegrationTest / fork := true,
  IntegrationTest / parallelExecution := false,
  IntegrationTest / testForkedParallel := false,

build.sbt:569: warning: value IntegrationTest is deprecated
  .configs(IntegrationTest extend Test)

build.sbt:576: warning: lazy value itSettings in object Defaults is deprecated
  Defaults.itSettings,
```

## Affected Code

### In `projectSettings` (lines 52, 126-128)
```scala
inThisBuild(List(
  Test / javaOptions := javaOpens,
  IntegrationTest / javaOptions := javaOpens  // deprecated
))

// ... and in projectSettings:
IntegrationTest / fork := true,              // deprecated
IntegrationTest / parallelExecution := false, // deprecated
IntegrationTest / testForkedParallel := false // deprecated
```

### In `rspace` project (lines 569, 576)
```scala
lazy val rspace = (project in file("rspace"))
  .configs(IntegrationTest extend Test)  // deprecated
  .settings(
    Defaults.itSettings,                 // deprecated
    // ...
  )
```

## Why This Can't Be Easily Suppressed

These are sbt DSL deprecation warnings emitted during build file loading, not Scala compiler warnings. There is no `@nowarn` annotation or compiler flag that can suppress them. The warnings are informational - the code continues to work.

## Recommended Fix

The sbt team recommends creating separate subprojects for integration tests instead of using the `IntegrationTest` configuration. This would involve:

1. Creating a new subproject like `rspace-it` for integration tests
2. Moving integration test sources from `src/it/scala` to the new subproject's `src/test/scala`
3. Setting up proper dependencies between the main project and the IT subproject
4. Removing all `IntegrationTest` configuration references

## Impact

- **Current**: Warnings only, no functional impact
- **Future**: The `IntegrationTest` configuration may be removed in a future sbt version

## Projects Affected

- `rspace` - Has integration tests configured with `IntegrationTest`
- All projects in `projectSettings` - Have `IntegrationTest` settings applied globally

---

*Created: December 2025*
*Status: Documented, fix deferred*
