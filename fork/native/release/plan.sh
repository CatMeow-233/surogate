#!/usr/bin/env bash
# Decide whether a run of .github/workflows/native-release.yml may publish native-v<VERSION>, and
# decide it again right before publishing.
#
#   fork/native/release/plan.sh build     build-only run, or a local preflight: report the version
#                                         and every publish condition that fails, exit 0
#   fork/native/release/plan.sh publish   plan job of a publish run: exit 1 unless every condition holds
#   fork/native/release/plan.sh recheck   release job, just before publishing: conditions 1-4 again,
#                                         read fresh from the remote (a re-run of the failed jobs reuses
#                                         the plan job's outputs, which may be days old)
#
# Conditions for publishing; VERSION is fork/native/release/VERSION at the run's commit:
#   1. VERSION is a PEP 440 version without a local part (the pattern build-wheel.sh also checks).
#   2. The run is on the default branch.
#   3. The tag native-v<VERSION> does not exist on the remote. A published version is never rebuilt
#      or replaced: set a new VERSION instead.
#   4. The commit is the default branch's current tip or an ancestor of it, and its .github/workflows
#      is identical to the tip's. GitHub does not let GITHUB_TOKEN create a tag on a commit whose
#      workflow files differ from the default branch, and the workflow would find out only after
#      the multi-hour build.
#   5. (build, publish) VERSION is <base>.post<N>[.dev<M>], where <base> is the newest upstream release
#      (a vX.Y.Z tag of UPSTREAM_URL) in the commit's history. An older base would label the wheel
#      as older than upstream releases whose code it contains.
#
# Environment (GitHub Actions sets the GITHUB_* variables; locally they default to HEAD):
#   GITHUB_SHA, GITHUB_REF  the run's commit and ref (default: HEAD and its branch)
#   DEFAULT_BRANCH          the repository's default branch (default: main)
#   RELEASE_TAG             recheck: the tag the plan job chose
#   REMOTE                  the fork's remote (default: origin)
#   UPSTREAM_URL            the upstream repository (default: https://github.com/invergent-ai/surogate.git)
#   GITHUB_OUTPUT           build, publish: receives version, tag, upstream_base, publish, release_tag
# Needs the full history (actions/checkout with fetch-depth: 0). Moves no ref, so it can run as a
# local preflight: the fetch writes only FETCH_HEAD and the fetched objects, and its empty --refmap
# keeps git from updating the remote-tracking branch (refs/remotes/<REMOTE>/<DEFAULT_BRANCH>).
set -euo pipefail

mode=${1:-}
case "$mode" in
    build | publish | recheck) ;;
    *)
        echo "usage: fork/native/release/plan.sh build|publish|recheck" >&2
        exit 2
        ;;
esac
cd "$(dirname "$0")/../../.."

REMOTE=${REMOTE:-origin}
UPSTREAM_URL=${UPSTREAM_URL:-https://github.com/invergent-ai/surogate.git}
DEFAULT_BRANCH=${DEFAULT_BRANCH:-main}
sha=$(git rev-parse --verify "${GITHUB_SHA:-HEAD}^{commit}")
ref=${GITHUB_REF:-$(git symbolic-ref -q HEAD || echo "a detached HEAD")}

say() { # level message
    if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::$1::$2"; else echo "$1: $2"; fi
}

problems=0
problem() {
    problems=$((problems + 1))
    if [ "$mode" = build ]; then say notice "a publish run would refuse: $*"; else say error "$*"; fi
}

# Prints the newest X.Y.Z among the upstream vX.Y.Z tags (git ls-remote --tags output on stdin)
# whose commit is in the history of $sha. Annotated tags are followed to their commit (^{} lines).
newest_upstream_release() {
    local releases name commit
    releases=$(awk '{
            name = $2; sub(/^refs\/tags\//, "", name)
            if (name ~ /\^\{\}$/) { sub(/\^\{\}$/, "", name); peeled[name] = $1 } else { direct[name] = $1 }
         }
         END {
            for (name in direct)
                if (name ~ /^v[0-9]+(\.[0-9]+)*$/)
                    print substr(name, 2), (name in peeled ? peeled[name] : direct[name])
         }' | sort -k1,1Vr)
    while read -r name commit; do
        [ -n "$name" ] || continue
        if git cat-file -e "$commit^{commit}" 2>/dev/null && git merge-base --is-ancestor "$commit" "$sha"; then
            echo "$name"
            return 0
        fi
    done <<<"$releases"
}

# 1. VERSION
version=$(git show "$sha:fork/native/release/VERSION" | tr -d '[:space:]')
tag=native-v$version
pep440='^[0-9]+(\.[0-9]+)*((a|b|rc)[0-9]+)?(\.post[0-9]+)?(\.dev[0-9]+)?$'
[[ $version =~ $pep440 ]] ||
    problem "fork/native/release/VERSION is '$version', not a PEP 440 version without a local part"
if [ "$mode" = recheck ] && [ "${RELEASE_TAG:-}" != "$tag" ]; then
    problem "the plan job chose '${RELEASE_TAG:-}', but VERSION at $sha gives $tag"
fi

# 2. The default branch
[ "$ref" = "refs/heads/$DEFAULT_BRANCH" ] ||
    problem "the run is on $ref; publish only from refs/heads/$DEFAULT_BRANCH"

# 3. The tag must not exist (exit code 2 of ls-remote --exit-code: no matching ref)
rc=0
git ls-remote --exit-code --tags "$REMOTE" "refs/tags/$tag" >/dev/null || rc=$?
case $rc in
    0) problem "$tag already exists on $REMOTE; a published version is never replaced, so set a new VERSION" ;;
    2) ;;
    *) problem "cannot list the tags of $REMOTE (git ls-remote exit code $rc)" ;;
esac

# 4. The commit is on the default branch, with the default branch's workflow files
if [ "$(git rev-parse --is-shallow-repository)" = true ]; then
    problem "the clone is shallow; plan.sh needs the full history (actions/checkout fetch-depth: 0)"
elif ! git fetch --quiet --no-tags --refmap= "$REMOTE" "refs/heads/$DEFAULT_BRANCH"; then
    problem "cannot fetch $DEFAULT_BRANCH from $REMOTE"
else
    tip=$(git rev-parse FETCH_HEAD)
    if ! git merge-base --is-ancestor "$sha" "$tip"; then
        problem "$sha is not in the history of $DEFAULT_BRANCH (now $tip)"
    elif ! git diff --quiet "$sha" "$tip" -- .github/workflows; then
        problem ".github/workflows at $sha differs from $DEFAULT_BRANCH (now $tip), so GITHUB_TOKEN cannot" \
            "create the tag there; run the workflow again on the current $DEFAULT_BRANCH"
    fi
fi

# 5. VERSION names the upstream release the commit is based on
base=
if [ "$mode" != recheck ]; then
    if ! listing=$(git ls-remote --tags "$UPSTREAM_URL" 'v[0-9]*'); then
        problem "cannot list the tags of $UPSTREAM_URL"
    else
        base=$(newest_upstream_release <<<"$listing")
        post="^${base//./\\.}\\.post[0-9]+(\\.dev[0-9]+)?\$"
        if [ -z "$base" ]; then
            problem "no upstream release tag vX.Y.Z of $UPSTREAM_URL is in the history of $sha"
        elif ! [[ $version =~ $post ]]; then
            problem "VERSION $version does not name the upstream release $sha is based on (v$base); use $base.post<N>"
        fi
    fi
fi

publish=false
if [ "$mode" != build ]; then
    if [ "$problems" -gt 0 ]; then
        say error "refusing to publish $tag from $sha ($problems problem(s) above)"
        exit 1
    fi
    publish=true
fi
echo "version $version, tag $tag, commit $sha, upstream base ${base:-unknown}, publish $publish"
if [ "$mode" != recheck ] && [ -n "${GITHUB_OUTPUT:-}" ]; then
    release_tag=
    [ "$publish" = false ] || release_tag=$tag
    printf '%s\n' "version=$version" "tag=$tag" "upstream_base=$base" "publish=$publish" \
        "release_tag=$release_tag" >>"$GITHUB_OUTPUT"
fi
