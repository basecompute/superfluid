# Release notes

Before tagging a release, add `docs/release-notes/<version>.md`, without the
tag's leading `v` (for example, `0.1.3.md` for `v0.1.3`). Commit it before
creating the tag: the release workflow reads the file from the tagged commit.

The file is the GitHub release body verbatim. Start with `# superfluid <version>`
and a short summary of the main user-facing change, then use `## New`,
`## Changed` and `## Fixed` as needed. Use absolute URLs for links so they work
on GitHub's release page too. `CHANGELOG.md` remains the detailed change log.

The workflow refuses to create a new draft when its notes are missing or
empty. A retry for an existing release only replaces its assets, preserving
any edits made to its notes on GitHub. A manual run for an older tag that has
no note file can still upload assets to an existing release.

MkDocs excludes this entire directory from the built site, including search
and the sitemap, for both development and tagged documentation.
