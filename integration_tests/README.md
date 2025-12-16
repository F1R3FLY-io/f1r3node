# Installing the prerequisites

## Step 1. Set up the Nix development environment

This project uses Nix to manage all development dependencies including Python, pyenv, and pipenv. All required tools are provided automatically through the Nix flake.

**Enable the development environment:**

```bash
$ direnv allow
```

This command activates the Nix development shell, making the following tools available:
- `python3` (Python 3.13)
- `pipenv` (Python virtual environment manager)
- `pyenv` (Python version manager)
- All other project dependencies

**Note:** If you don't have `direnv` installed, you can manually enter the Nix shell with `nix develop`, but `direnv` is recommended for automatic environment activation.

## Step 2. Install Docker

If you don't have Docker installed, refer to the platform-specific installation
instructions at [Docker website](https://docs.docker.com/install/).

## Step 3: Install Python Dependencies

The integration tests require Python packages including type stubs for proper type checking with mypy.

```bash
$ cd integration_tests/
$ pipenv sync
```

**Note:** The `Pipfile` includes type stubs (`types-requests`, `types-docker`) required for mypy type checking. These will be installed automatically with the above command.

## Step 4: Create the integration test docker image

Tests use a Docker image called `rchain-integration-tests:latest`. This image must be built before running tests.

**Build the integration test image:**

```bash
$ cd integration_tests/
$ docker build -t rchain-integration-tests:latest .
```

This creates a Docker image based on `f1r3flyindustries/f1r3fly-scala-node:latest` with the necessary tools (curl, etc.) for running integration tests.

**Note:** The `./run_tests` script will automatically build this image if it doesn't exist and no `DEFAULT_IMAGE` environment variable is set. However, if you need to rebuild after changes to the base image, run the build command above manually.

**CI/CD:** If environment variable `${DRONE_BUILD_NUMBER}` is defined, the tests will use `coop.rchain/rnode:DRONE-${DRONE_BUILD_NUMBER}` instead. These images are created on Drone CI for each build.

# Running the tests
## Configuration

The file `pytest.ini` allows some configuration of the test execution. Information about all the available options
can be found in [pytest.ini reference](https://docs.pytest.org/en/latest/reference.html#ini-options-ref)

## Running from Docker

TL;DR: If you want to run these tests from a Docker container, start the
container with `-v /var/run/docker.sock:/var/run/docker.sock -v /tmp:/tmp`.

These tests can be run from a Docker container, but in that case a) the Docker
socket has to be accessible (writeable) by the container, and b) the temporary
directory **in the container** (either `/tmp` or whatever is in the environment
variable `$TMPDIR`) has to be accessible by the host as well **on the same
path**. The reason for the latter is that tests spawn additional containers and
need to share files with them. They do it by mounting files/directories from
`/tmp` (or `$TMPDIR`) into new containers, i.e. by starting new containers with
e.g. `-v /tmp/bonds.txt:/var/lib/rnode/genesis/bonds.txt` arugments. But these
arguments are passed via shared Docker socket to Docker daemon running on the
host. So the Docker daemon has to be able to access `/tmp/bonds.txt` as well.

## Execution

The tests are run using *pytest*. If you want to have a deep understanding of the whole framework you should check
[pytest documentation](https://docs.pytest.org/en/latest/contents.html#toc)

The tests can be run using the bash script

```bash
$ ./run_tests
```

In order to run only specific tests can specify the test subdir where you want
the discovery to start

Examples:
Run the tests for the complete connected network:

```bash
$ ./run_tests test/test_propose.py
```

You can see all the options available by running

```bash
$ ./run_tests --help
```

To stop after the first failing tests or after N failure you can use `-x` or
`--maxfail`:

```bash
$ ./run_tests -x
```

```bash
$ ./run_tests --maxfail=3
```

The test discovery starts in the directories specified in the command line.
If no directory is provided all the tests are run.

If you want to see what tests will be run by a certain command use the parameter `--collect-only`

Examples
```bash
$ ./run_tests --collect-only
```
```bash
$ ./run_tests --collect-only  test/test_star_connected.py
```

## Type Checking

The test suite includes [mypy](https://mypy-lang.org/) static type checking as part of the code quality checks. The type checker runs automatically before tests execute via the `./check_code` script.

**Type stubs:** The project includes type stubs for third-party libraries (`types-requests`, `types-docker`) to ensure accurate type checking. These are installed automatically with `pipenv sync`.

**Running type checks manually:**

```bash
$ ./check_code
```

The tests also support running [pytest-mypy](https://pypi.org/project/pytest-mypy/) plugin for type checking during test discovery:

```bash
$ ./run_tests --mypy
```

If you want to restrict your test run to only perform mypy checks and not any other tests by using the `-m` option.

```bash
$ ./run_tests --mypy -m mypy
```

**Skipping type checks:** To skip type checks during test runs (not recommended), set the environment variable:

```bash
$ _SKIP_CHECK_CODE=1 ./run_tests
```

## Troubleshooting

If you're on macOS and getting exceptions similar to

```
self = <Response [403]>

    def raise_for_status(self):
        """Raises stored :class:`HTTPError`, if one occurred."""

        http_error_msg = ''
        if isinstance(self.reason, bytes):
            # We attempt to decode utf-8 first because some servers
            # choose to localize their reason strings. If the string
            # isn't utf-8, we fall back to iso-8859-1 for all other
            # encodings. (See PR #3538)
            try:
                reason = self.reason.decode('utf-8')
            except UnicodeDecodeError:
                reason = self.reason.decode('iso-8859-1')
        else:
            reason = self.reason

        if 400 <= self.status_code < 500:
            http_error_msg = u'%s Client Error: %s for url: %s' % (self.status_code, reason, self.url)

        elif 500 <= self.status_code < 600:
            http_error_msg = u'%s Server Error: %s for url: %s' % (self.status_code, reason, self.url)

        if http_error_msg:
>           raise HTTPError(http_error_msg, response=self)
E           requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: http+docker://localhost/v1.35/networks/4c5079902ad7d50a0c7a763ac6a022923c2fe2e4ceb608952c67d433b428e891
```

make sure you have at least 4GiB of RAM set for use by the Docker Engine
(macOS system menu bar -> Docker icon -> Preferences... -> Advanced).

## Cleanup

After running integration tests, Docker resources (containers, networks, volumes) may be left behind, especially if tests fail or are interrupted. Use the following commands to clean up:

**Remove stopped test containers:**

```bash
$ docker container prune -f
```

**Remove test containers by name pattern:**

```bash
$ docker ps -a --filter "name=rnode" -q | xargs -r docker rm -f
```

**Remove unused Docker networks:**

```bash
$ docker network prune -f
```

**Remove unused Docker volumes:**

```bash
$ docker volume prune -f
```

**Full cleanup (containers, networks, volumes, and dangling images):**

```bash
$ docker system prune -f
```

**Remove the integration test image (if you need to rebuild):**

```bash
$ docker rmi rchain-integration-tests:latest
```

**Note:** Be careful with `docker system prune` as it removes all unused resources, not just those from integration tests. If you have other Docker projects, consider using the more targeted commands above.
