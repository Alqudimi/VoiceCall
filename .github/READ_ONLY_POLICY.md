# GitHub Actions branch-safety policy

No workflow in this local copy is configured to push commits to a branch, create
or update a branch, or merge a pull request. The Dependabot auto-merge workflow
and Dependabot update configuration are omitted because they can create/update
PR branches or merge changes into `main`.

Release/tag workflows are retained as requested. Their explicit ref push is to
`refs/tags/...`, not a branch; release-drafter updates release metadata only.
Other retained workflows may perform their existing non-branch operations,
such as publishing packages/pages or maintaining labels/issues.

This is a local copy only; no changes were pushed to GitHub. Repository-level
Dependabot security-update settings are separate from this file and may need to
be disabled in GitHub settings if they are enabled there.
