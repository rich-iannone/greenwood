---
name: release-notes
description: Generate polished release notes for Greenwood from GitHub PR/commit data. Creates an opening summary paragraph and well-organized sections for features, enhancements, fixes, and docs.
argument-hint: "[GitHub release changelog text or PR list]"
---

# Greenwood Release Notes Workflow

Generate professional release notes from raw GitHub release data. The input is: $ARGUMENTS

## Output Format

Release notes follow this structure:

```markdown
## v{VERSION}

_{DATE}_ · [GitHub](https://github.com/rich-iannone/greenwood/releases/tag/v{VERSION})

{SUMMARY_PARAGRAPH}

### New Features

- **{Feature name}** — {Brief description of what it does}. (#{PR_NUMBER})

### Enhancements

- {Description of improvement}. (#{PR_NUMBER})

### Bug Fixes

- {Description of what was fixed}. (#{PR_NUMBER})

### Documentation

- {Description of docs change}. (#{PR_NUMBER})
```

## Presenting the Final Output

Always wrap the final release notes in **4 backticks** (not 3) to facilitate easy copy/paste in VS Code:

`````markdown
```
## v{VERSION}

...release notes content...
```
`````

This prevents issues when the release notes themselves contain triple-backtick code blocks.

## Step 1: Categorize the Changes

Parse the input and categorize each PR by its prefix:

| Prefix   | Category                | Section                          |
| -------- | ----------------------- | -------------------------------- |
| `feat:`  | New functionality       | **New Features**                 |
| `enh:`   | Improvement to existing | **Enhancements**                 |
| `fix:`   | Bug fix                 | **Bug Fixes**                    |
| `doc:`   | Documentation only      | **Documentation**                |
| `chore:` | Maintenance/tooling     | **Maintenance** (if significant) |
| `test:`  | Test additions          | Usually omit unless significant  |

If no prefix, infer from the PR title keywords:

- "add support for", "new", "implement" → New Features
- "improve", "update", "enhance" → Enhancements
- "fix", "correct", "resolve" → Bug Fixes
- "document", "docs", "readme" → Documentation

## Step 2: Research Feature Details

For each **New Feature**, gather context to write accurate descriptions:

1. Search the codebase for the feature implementation
2. Check for user-facing documentation (in `user_guide/` or `recipes/`)
3. Look at config options added (in `config.py`)
4. Identify the main benefit to users

Key questions to answer:

- What problem does this solve?
- What's the user-facing command or config option?
- Does it require configuration or work automatically?

## Step 3: Write the Summary Paragraph

The opening paragraph should:

1. **State the release theme** in broad terms (e.g., "improvements to site presentation and SEO")
2. **Highlight 3-5 major features** by name
3. **Mention key user benefits** (e.g., "catch grammar issues locally", "better mobile experience")
4. Be **2-4 sentences** — concise but informative

Example:

> Greenwood v0.2 brings significant improvements to site presentation, content quality, and SEO. This release introduces styled tooltips throughout, responsive tables that work on any screen size, Mermaid diagram support with dark mode compatibility, and comprehensive SEO features including automatic sitemap generation. A new `proofread` command powered by Harper helps catch grammar and spelling issues locally. Pages can now display creation/modification metadata, and `ROADMAP.md` files are automatically integrated into your documentation.

## Step 4: Format Each Entry

### New Features Format

Use bold for the feature name, em-dash separator, and end with PR number:

```markdown
- **{Feature name}** — {What it does and why it's useful}. (#{PR})
```

Examples:

- **ROADMAP.md support** — Project roadmap files are now auto-detected and integrated into the documentation site with proper navigation links. (#42)
- **Harper proofreading** — New `great-docs proofread` command for local grammar and spelling checks with a built-in technical dictionary and multiple output formats. (#48)

### Enhancements Format

Simpler format without bold feature name:

```markdown
- {What was improved and how}. (#{PR})
```

Example:

- Markdown is now supported in announcement banner text, allowing links and formatting. (#35)

### Bug Fixes Format

Describe what was wrong and that it's now fixed:

```markdown
- {What was broken} was fixed. (#{PR})
```

Or describe the improvement made:

```markdown
- {Component} now {correct behavior}. (#{PR})
```

Examples:

- Python version is now auto-detected for GitHub deployment workflows. (#40)
- Images in README.md files are now properly copied to the output directory. (#57)

### Documentation Format

Brief description of what was documented:

```markdown
- Added documentation on {topic}. (#{PR})
```

## Step 5: Order and Polish

### Section Order

1. New Features (most important)
2. Enhancements
3. Bug Fixes
4. Documentation
5. Maintenance (only if significant)

### Within Each Section

Order entries by:

1. **User impact** — Most impactful first
2. **Related features** — Group related items together

### Final Checklist

- [ ] Summary paragraph captures the release theme
- [ ] All PRs are categorized
- [ ] Feature descriptions explain user benefit
- [ ] Consistent formatting (bold, em-dash, periods, PR numbers)
- [ ] No duplicate information between summary and entries
- [ ] Date is correct (use current date if releasing today)

## Section Templates by Release Type

### Major Release (new capabilities)

Focus on New Features section. Summary emphasizes what users can now do.

### Minor Release (improvements)

Balance New Features and Enhancements. Summary emphasizes improved experience.

### Patch Release (fixes only)

May omit New Features section entirely. Brief summary:

> Greenwood v{X.Y.Z} is a maintenance release with bug fixes and stability improvements.

### Packaging-Only Release

> Packaging fix release — no functional changes from v{PREVIOUS}.
>
> ### Changes
>
> - {What changed in packaging}. (#{PR})
>
> See [v{PREVIOUS} release notes](...) for all features and fixes in this release.
