# RNode - F1R3FLY.io Blockchain Platform

AI assistant guidance for RNode - F1R3FLY.io Blockchain Platform. This file follows the Agentic AI Foundation (Linux Foundation) standard for AI coding assistants.

**Full documentation:** See [CLAUDE.md](CLAUDE.md) for comprehensive project guidelines.

## Deprecation Notice
- This repository is **deprecated and read-only**. Do not start new feature work, fixes, or refactors here.
- Direct developers and users to [F1R3FLY-io/f1r3node-rust](https://github.com/F1R3FLY-io/f1r3node-rust) for all development, issues, and pull requests.

## Project Context
- RNode is a decentralized blockchain platform developed by F1R3FLY.io that implements four distinct consensus mechanisms using the Rholang programming language
- This is a **multi-language blockchain project** with Scala for node infrastructure and Rust for performance-critical components
- The platform provides concurrent smart contract execution with Byzantine Fault Tolerant operations
- If the user does not provide enough information with their prompts, ask the user to clarify before executing the task
- Follow F1R3FLY.io's documentation-first development methodology for all features

## Security Considerations

## Code Style and Standards

- **Style**: Functional programming with Akka actors
- **Build**: SBT with multi-project structure
- **Testing**: ScalaTest with property-based testing
- **Concurrency**: Actor model for concurrent operations

- **Style**: Zero-cost abstractions, ownership model
- **Build**: Cargo with workspace management
- **Testing**: Built-in test framework with proptest
- **Performance**: Optimize for concurrent execution

- **Paradigm**: Process calculus based concurrent programming
- **Communication**: Channel-based message passing
- **Security**: Unforgeable names and capability-based security
- **Pattern Matching**: Structural pattern matching on channels

## AI Artifact Generation Guidelines

**Core strategy:** Default to **Markdown + Mermaid** as the source of truth for all generated artifacts. Use **HTML** only when high engagement or advanced interactivity is required.

**Preferred formats:**

| Format | Use for | Notes |
|--------|---------|-------|
| **Markdown + Mermaid** | Primary. Diagrams (flowcharts, sequences, architecture, timelines, Gantt, ERDs), structured documents, plans, specs | Relative links (`./images/`, `./docs/`) and GitHub/GitLab raw URLs for local/cloud asset referencing |
| **HTML (CSS/JS + embedded Mermaid)** | Secondary. Interactive dashboards, prototypes, dynamic reviews, stakeholder deliverables | When visual polish and engagement are critical (tabs, sliders, clickable elements) |

**Hybrid rule:** Always produce Markdown as the canonical, Git-friendly version first; generate a self-contained HTML export on request.

**Key principles:**

- Prioritize human readability, editability, and Git compatibility (clean diffs, relative paths, native rendering on GitHub/GitLab).
- Maximize information density while avoiding text walls — convert complex information into Mermaid diagrams.
- Support seamless referencing of local files and cloud artifacts (images, other docs, raw Git content).
- Favor Markdown for internal/agent use and long-term storage (token efficiency).
- Use HTML when delivering to stakeholders or for living documents (engagement).

**Output guidance:** When creating artifacts, ask whether HTML interactivity is needed. Default to clean Markdown with embedded Mermaid unless specified.

## UI Test Assertions

React UI tests should prove user-observable DOM structure and state, not incidental copy or formatted sample values. Treat exact text/value assertions as a last resort unless the behavior under test is specifically copy, formatting, or content transformation.

- **Preferred:** Accessibility-first queries for DOM elements and states: `getByRole`, `getByLabelText`, `aria-label`, `aria-labelledby`, `aria-selected`, `aria-expanded`, `aria-disabled`, focus state, and ARIA relationships.
- **Preferred:** Assert semantic regions/components are present and wired correctly (tabs, tabpanels, dialogs, forms, buttons, lists), then assert behavior through state changes or callback/data-source calls.
- **Acceptable:** `data-testid` attributes when no accessible handle exists, the element is purely presentational, or the test needs to identify a stable component boundary rather than text content.
- **Acceptable:** HTML `id` attributes for form elements and ARIA relationships.
- **Avoid:** `getByText`/`queryByText` for literal strings that are merely display copy, repeated metrics, formatted numbers, or mock-data values.
- **Avoid:** `getAllByText(...).length` as a substitute for a meaningful DOM assertion; it couples tests to duplicated visual text rather than behavior.
- **Avoid:** CSS class selectors that may change with styling updates.

Examples:

```tsx
// GOOD: accessibility-first DOM element + state
expect(screen.getByRole("button", { name: "Refresh agents" })).toBeEnabled();

// ACCEPTABLE: fallback when no accessible role/name fits
expect(screen.getByTestId("cost-optimization-panel")).toBeInTheDocument();

// GOOD for data-flow behavior: verify source + rendered component boundary,
// not a duplicated metric string like "7.5%".
expect(fetchQualityPipeline).toHaveBeenCalledTimes(1);
expect(screen.getByRole("tablist", { name: "Quality Pipeline tabs" })).toBeInTheDocument();
expect(screen.getByRole("tab", { name: /overview/i })).toHaveAttribute("aria-selected", "true");

// BAD: brittle copy assertion
expect(screen.getByText("Priority set by BountyForge routing")).toBeInTheDocument();

// BAD: brittle mock-value / duplicated visual text assertion
expect(screen.getByText("7.5%")).toBeInTheDocument();
expect(screen.getAllByText("7.5%").length).toBeGreaterThan(0);
```

**When exact text is appropriate:** Use exact text assertions only when the acceptance criteria is about user-facing copy, accessibility name computation, validation messages, formatting rules, or transformed content. Prefer scoping with `within(...)` to a semantic container so the assertion remains tied to the behavior under test.


When creating or modifying configuration files, follow these conventions to respect existing project preferences:

**JSON Format Preference Order:**

1. **Check for existing files first**: Before creating any `.json` file, check if `.jsonc` or `.json5` variants exist
2. **Prefer existing format**: If `config.jsonc` or `config.json5` exists, use that format instead of creating `config.json`
3. **Default to JSONC**: When creating new config files, prefer `.jsonc` (JSON with Comments) for better maintainability

**Why This Matters:**
- Projects may have established preferences for comment-supporting JSON formats
- Creating duplicate configs (e.g., both `biome.json` and `biome.jsonc`) causes confusion
- JSONC allows inline documentation which improves maintainability

**Examples:**

| If exists... | Don't create... | Instead... |
|--------------|-----------------|------------|
| `biome.jsonc` | `biome.json` | Edit the existing `biome.jsonc` |
| `tsconfig.json5` | `tsconfig.json` | Edit the existing `tsconfig.json5` |
| `eslint.config.jsonc` | `eslint.config.json` | Edit the existing file |
| Nothing | - | Create new file as `.jsonc` when comments are useful |

**File Discovery Pattern:**

Before creating any config file, check for variants:
```bash
# Check for config variants (example for biome)
ls biome.json biome.jsonc biome.json5 2>/dev/null
```

This applies to all slash commands and scripts that create configuration files.

- `/quick-commit` - Stage and commit changes (required in safe mode)
- `/recursive-push` - Push across repositories

- `/nextTask` - Find and select next task to work on
- `/implement` - Begin implementation of a task
- `/epic-review` - Preview and summarize epics
- `/epic-hygiene` - Archive completed epics

- `/harmonize` - Sync workspace policies into this repo
- `/multi-repo-sync` - Workspace-wide sync orchestration

[OPTIONAL_COMMANDS]


**CRITICAL - Before submitting any contribution:**

Contributors MUST ensure their code, commits, and documentation do NOT contain PII:

**Check before committing:**
- [ ] No absolute file paths with usernames in code or documentation
- [ ] No personal email addresses in code (use generic examples like `user@example.com`)
- [ ] No real user data in tests or examples (use synthetic/fake data only)
- [ ] No PII in log statements (sanitize or use user IDs instead)
- [ ] No PII in error messages or stack traces
- [ ] No PII in code comments or documentation
- [ ] No credentials, tokens, or secrets in code (use environment variables)
- [ ] No IP addresses, MAC addresses, or device identifiers in examples

**If you accidentally committed PII:**
1. **DO NOT** push to remote repository
2. Use `git reset` to remove the commit
3. If already pushed, contact maintainers immediately
4. Repository history may need to be rewritten to remove PII

**Use these instead:**
- File paths: Use relative paths or generic placeholders (`[WORKSPACE_ROOT]/project/`)
- Email addresses: Use `user@example.com`, `admin@example.com`
- Names: Use `John Doe`, `Jane Smith`, `User123`
- Phone numbers: Use `+1-555-0100` (officially reserved for examples)
- IP addresses: Use reserved ranges (`192.0.2.1`, `198.51.100.1`, `203.0.113.1`)
- Dates: Use recent but generic dates, not specific personal dates

**For test data:**
- Use test data generators that create realistic but fake data
- Use well-known test fixtures (e.g., `test@example.com`)
- Never use production or real user data in development/testing

## grepai - Semantic Code Search

`grepai` is an optional, MIT-licensed semantic-search tool. If it is installed
and indexed locally for this repo, prefer it for intent-based code exploration;
if it is unavailable, fall back to your harness's native search and file-reading
tools. It is recommended but never required. Nothing here is harness-specific -
substitute your tool's equivalents for the generic actions described below.
Setup (with the privacy-first local Ollama embedder as default) lives in the
CLI Setup guide's "Optional: Semantic Code Search (grepai)" section.


---

**Detailed guidelines:** [CLAUDE.md](CLAUDE.md)
