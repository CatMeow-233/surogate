#!/usr/bin/env bash
# Decide whether a run of .github/workflows/experimental-release.yml may publish
# experimental-v<version>, and decide it again right before publishing.
#
#   fork/experimental/release_plan.sh build     manual run (workflow_dispatch) or a local preflight:
#                                               report the version and every publish condition that
#                                               fails, exit 0
#   fork/experimental/release_plan.sh publish   build job of a tag push: exit 1 unless every condition holds
#   fork/experimental/release_plan.sh recheck   release job, just before uploading: every condition again,
#                                               read fresh from the remote
#
# Unlike native-release.yml, which only a manual run on the default branch publishes, a tag push
# publishes here: pushing the tag is the owner's explicit act and names one commit, and the build takes
# minutes, so nothing long starts from a push by accident. The tag can still be pushed at any commit,
# so the run publishes only when:
#   1. The run is for a tag refs/tags/experimental-v<V>.
#   2. V is the version in fork/experimental/pyproject.toml at the run's commit, spelled canonically
#      (release_version_ok in fork/tools/release_gate.sh), and not a development release (.devN): the
#      checkout carries the next version with .dev0 between releases, and no published wheel may share it.
#   3. The tag on the remote still names the run's commit: a tag deleted or moved after the push no
#      longer describes this build.
#   4. The commit is in the history of the default branch, fetched now (the clone needs its full history).
# Then the release job runs fork/tools/release_guard.py: no release, published or draft, may exist for the
# tag, and the files are published only after they are verified on a draft.
#
# Environment (GitHub Actions sets the GITHUB_* variables; locally they default to HEAD):
#   GITHUB_SHA, GITHUB_REF  the run's commit and ref (default: HEAD and its branch)
#   DEFAULT_BRANCH          the repository's default branch (default: main)
#   REMOTE                  the fork's remote (default: origin)
#   GITHUB_OUTPUT           build, publish: receives version, tag, publish
# Moves no ref (see release_on_default_branch); needs git and python3 (3.11 or later, for tomllib).
set -euo pipefail

mode=${1:-}
case "$mode" in
    build | publish | recheck) ;;
    *)
        echo "usage: fork/experimental/release_plan.sh build|publish|recheck" >&2
        exit 2
        ;;
esac
cd "$(dirname "$0")/../.."
# shellcheck source=../tools/release_gate.sh
. fork/tools/release_gate.sh

REMOTE=${REMOTE:-origin}
DEFAULT_BRANCH=${DEFAULT_BRANCH:-main}
sha=$(git rev-parse --verify "${GITHUB_SHA:-HEAD}^{commit}")
ref=${GITHUB_REF:-$(git symbolic-ref -q HEAD || echo "a detached HEAD")}
problems=0 # counted by problem (release_gate.sh)

# 1. A tag push
tag=${ref#refs/tags/}
if [ "$ref" = "$tag" ] || [[ $tag != experimental-v* ]]; then
    problem "the run is on $ref; only a push of a tag experimental-v<version> publishes"
    tag=
fi

# 2. The version
read_version='import sys, tomllib; print(tomllib.loads(sys.stdin.read())["project"]["version"])'
if ! version=$(git show "$sha:fork/experimental/pyproject.toml" | python3 -c "$read_version"); then
    problem "cannot read the version from fork/experimental/pyproject.toml at $sha"
    version=
elif ! release_version_ok "$version"; then
    problem "fork/experimental/pyproject.toml has version '$version', not a PEP 440 version without a" \
        "local part in canonical spelling (for example 0.1.0a4, not 0.1.0.a4 or 0.1.0a04)"
elif [[ $version == *.dev* ]]; then
    problem "version $version is a development release: set the release version (without .devN) first"
fi
if [ -n "$tag" ] && [ -n "$version" ] && [ "$tag" != "experimental-v$version" ]; then
    problem "tag $tag does not match fork/experimental/pyproject.toml at $sha (experimental-v$version)"
fi

# 3. The tag still names the commit
if [ -n "$tag" ]; then
    rc=0
    tagged=$(release_remote_tag "$REMOTE" "$tag") || rc=$?
    case $rc in
        0) [ "$tagged" = "$sha" ] || problem "$tag on $REMOTE now names $tagged, not the run's commit $sha" ;;
        2) problem "$tag does not exist on $REMOTE" ;;
        *) problem "cannot list the tags of $REMOTE (git ls-remote exit code $rc)" ;;
    esac
fi

# 4. The commit is on the default branch
release_on_default_branch "$sha" || true

publish=false
if [ "$mode" != build ]; then
    if [ "$problems" -gt 0 ]; then
        say error "refusing to publish ${tag:-$ref} from $sha ($problems problem(s) above)"
        exit 1
    fi
    publish=true
fi
echo "version ${version:-?}, tag ${tag:-none}, commit $sha, publish $publish"
if [ "$mode" != recheck ] && [ -n "${GITHUB_OUTPUT:-}" ]; then
    printf '%s\n' "version=$version" "tag=$tag" "publish=$publish" >>"$GITHUB_OUTPUT"
fi
