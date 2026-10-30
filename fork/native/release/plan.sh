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
#   1. VERSION is a PEP 440 version without a local part, spelled canonically (release_version_ok in
#      fork/tools/release_gate.sh, which build-wheel.sh also uses): 1.5.12.post1, not 1.5.12.post01.
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
# build and publish also find the upstream commit the fork last merged: the merge base of the commit
# with upstream's main, as fork/tools/check_divergence.py defines it. fork/dev sync merges upstream's
# main, not a tag, so that commit can be past <base>; the release notes name it and the distance.
#
# Environment (GitHub Actions sets the GITHUB_* variables; locally they default to HEAD):
#   GITHUB_SHA, GITHUB_REF  the run's commit and ref (default: HEAD and its branch)
#   DEFAULT_BRANCH          the repository's default branch (default: main)
#   RELEASE_TAG             recheck: the tag the plan job chose
#   REMOTE                  the fork's remote (default: origin)
#   UPSTREAM_URL            the upstream repository (default: https://github.com/invergent-ai/surogate.git)
#   UPSTREAM_BRANCH         the upstream branch fork/dev sync merges (default: main)
#   GITHUB_OUTPUT           build, publish: receives version, tag, upstream_base, upstream_commit,
#                           upstream_describe (v<base>, or v<base> + <N> commits), publish, release_tag
# Needs the full history (actions/checkout with fetch-depth: 0). Moves no ref, so it can run as a
# local preflight: the fetches write only FETCH_HEAD and the fetched objects, and their empty --refmap
# keeps git from updating a remote-tracking branch (refs/remotes/<REMOTE>/<DEFAULT_BRANCH>).
# The checks it shares with fork/experimental/release_plan.sh are in fork/tools/release_gate.sh.
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
# shellcheck source=../../tools/release_gate.sh
. fork/tools/release_gate.sh

REMOTE=${REMOTE:-origin}
UPSTREAM_URL=${UPSTREAM_URL:-https://github.com/invergent-ai/surogate.git}
UPSTREAM_BRANCH=${UPSTREAM_BRANCH:-main}
DEFAULT_BRANCH=${DEFAULT_BRANCH:-main}
sha=$(git rev-parse --verify "${GITHUB_SHA:-HEAD}^{commit}")
ref=${GITHUB_REF:-$(git symbolic-ref -q HEAD || echo "a detached HEAD")}

problems=0 # counted by problem (release_gate.sh)

# Prints "X.Y.Z <commit>" for the newest X.Y.Z among the upstream vX.Y.Z tags (git ls-remote --tags
# output on stdin) whose commit is in the history of $sha. Annotated tags are followed to their
# commit (^{} lines).
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
            echo "$name $commit"
            return 0
        fi
    done <<<"$releases"
}

# 1. VERSION
version=$(git show "$sha:fork/native/release/VERSION" | tr -d '[:space:]')
tag=native-v$version
release_version_ok "$version" ||
    problem "fork/native/release/VERSION is '$version', not a PEP 440 version without a local part in" \
        "canonical spelling (for example 1.5.12.post1, not 1.5.12.post01 or 1.5.12post1)"
if [ "$mode" = recheck ] && [ "${RELEASE_TAG:-}" != "$tag" ]; then
    problem "the plan job chose '${RELEASE_TAG:-}', but VERSION at $sha gives $tag"
fi

# 2. The default branch
[ "$ref" = "refs/heads/$DEFAULT_BRANCH" ] ||
    problem "the run is on $ref; publish only from refs/heads/$DEFAULT_BRANCH"

# 3. The tag must not exist
rc=0
release_remote_tag "$REMOTE" "$tag" >/dev/null || rc=$?
case $rc in
    0) problem "$tag already exists on $REMOTE; a published version is never replaced, so set a new VERSION" ;;
    2) ;;
    *) problem "cannot list the tags of $REMOTE (git ls-remote exit code $rc)" ;;
esac

# 4. The commit is on the default branch, with the default branch's workflow files
if release_on_default_branch "$sha" && ! git diff --quiet "$sha" "$release_tip" -- .github/workflows; then
    problem ".github/workflows at $sha differs from $DEFAULT_BRANCH (now $release_tip), so GITHUB_TOKEN cannot" \
        "create the tag there; run the workflow again on the current $DEFAULT_BRANCH"
fi

# 5. VERSION names the upstream release the commit is based on. For the release notes, also find the
#    upstream commit the fork merged and how far it is past that release.
base='' upstream_commit='' upstream_describe=''
if [ "$mode" != recheck ]; then
    base_commit=
    if ! listing=$(git ls-remote --tags "$UPSTREAM_URL" 'v[0-9]*'); then
        problem "cannot list the tags of $UPSTREAM_URL"
    else
        read -r base base_commit <<<"$(newest_upstream_release <<<"$listing")"
        post="^${base//./\\.}\\.post[0-9]+(\\.dev[0-9]+)?\$"
        if [ -z "$base" ]; then
            problem "no upstream release tag vX.Y.Z of $UPSTREAM_URL is in the history of $sha"
        elif ! [[ $version =~ $post ]]; then
            problem "VERSION $version does not name the upstream release $sha is based on (v$base); use $base.post<N>"
        fi
    fi
    if ! git fetch --quiet --no-tags --refmap= "$UPSTREAM_URL" "refs/heads/$UPSTREAM_BRANCH"; then
        problem "cannot fetch $UPSTREAM_BRANCH from $UPSTREAM_URL"
    elif ! upstream_commit=$(git merge-base "$sha" FETCH_HEAD); then
        problem "$sha has no commit of $UPSTREAM_BRANCH of $UPSTREAM_URL in its history"
    elif [ -n "$base" ]; then
        # The upstream commits the wheel holds beyond v<base>: base_commit is in the history of $sha,
        # so the wheel's upstream part is the history of base_commit plus that of upstream_commit.
        ahead=$(git rev-list --count "$base_commit..$upstream_commit")
        case $ahead in
            0) upstream_describe=v$base ;;
            1) upstream_describe="v$base + 1 commit" ;;
            *) upstream_describe="v$base + $ahead commits" ;;
        esac
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
upstream=
[ "$mode" = recheck ] || upstream=", upstream ${upstream_describe:-v${base:-?}} at ${upstream_commit:-?}"
echo "version $version, tag $tag, commit $sha$upstream, publish $publish"
if [ "$mode" != recheck ] && [ -n "${GITHUB_OUTPUT:-}" ]; then
    release_tag=
    [ "$publish" = false ] || release_tag=$tag
    printf '%s\n' "version=$version" "tag=$tag" "upstream_base=$base" "upstream_commit=$upstream_commit" \
        "upstream_describe=$upstream_describe" "publish=$publish" "release_tag=$release_tag" >>"$GITHUB_OUTPUT"
fi
