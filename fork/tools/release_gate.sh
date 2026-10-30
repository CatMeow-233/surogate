# Checks shared by the fork's release gates. Source it from the repository root; it defines
# functions and one variable and runs nothing. Its users:
#   fork/native/release/plan.sh          may this run of native-release.yml publish native-v<VERSION>?
#   fork/native/release/build-wheel.sh   the version it writes into the native wheel
# Tested through those scripts by `fork/dev test release`.
#
# say and problem use the caller's mode (build, publish or recheck) and count into its problems:
# a build run reports a failed condition as a notice, a publish run or a re-check as an error.
# shellcheck shell=bash

# The canonical PEP 440 spelling (str(packaging.version.Version(v)) == v) of a public version without
# an epoch or a local part: numbers without leading zeros, then optionally a, b or rc with a number,
# .postN and .devN. Every other spelling of a published version (1.5.12.post01 or 1.5.12post1 for
# 1.5.12.post1, 0.1.0.a3 or 0.1.0-alpha3 for 0.1.0a3) names a tag that does not exist yet, so it would
# pass the once-per-version check and publish a second wheel with the same normalised file name.
# fork/experimental/build_release.py holds the same pattern (its tests compare the two).
RELEASE_VERSION_PATTERN='^(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$'

release_version_ok() { # version
    [[ $1 =~ $RELEASE_VERSION_PATTERN ]]
}

say() { # level message
    if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::$1::$2"; else echo "$1: $2"; fi
}

problem() { # message...
    problems=$((problems + 1))
    if [ "$mode" = build ]; then say notice "a publish run would refuse: $*"; else say error "$*"; fi
}

# release_remote_tag REMOTE TAG: prints the commit TAG names on REMOTE, an annotated tag followed to
# its commit. Exit status 0 when the tag exists, 2 when it does not, anything else when REMOTE cannot
# be listed (the exit status of git ls-remote).
release_remote_tag() {
    local listing rc=0 commit name direct='' peeled=''
    listing=$(git ls-remote --exit-code --tags "$1" "refs/tags/$2" "refs/tags/$2^{}") || rc=$?
    [ "$rc" = 0 ] || return "$rc"
    # ls-remote matches patterns against the end of a ref name; keep only the exact names.
    while read -r commit name; do
        case $name in
            "refs/tags/$2") direct=$commit ;;
            "refs/tags/$2^{}") peeled=$commit ;;
        esac
    done <<<"$listing"
    [ -n "$direct" ] || return 2
    echo "${peeled:-$direct}"
}

# release_on_default_branch SHA: problem unless this clone has its full history and SHA is in the
# history of DEFAULT_BRANCH on REMOTE, fetched now. Returns 0 and sets release_tip to that branch's
# commit when SHA is in its history, 1 otherwise. The fetch writes only FETCH_HEAD and the fetched
# objects: its empty --refmap keeps refs/remotes/<REMOTE>/<DEFAULT_BRANCH> where it was, so a local
# preflight moves no ref.
release_on_default_branch() {
    local sha=$1 tip
    release_tip=
    if [ "$(git rev-parse --is-shallow-repository)" = true ]; then
        problem "the clone is shallow; the release gate needs the full history (actions/checkout fetch-depth: 0)"
    elif ! git fetch --quiet --no-tags --refmap= "$REMOTE" "refs/heads/$DEFAULT_BRANCH"; then
        problem "cannot fetch $DEFAULT_BRANCH from $REMOTE"
    else
        tip=$(git rev-parse FETCH_HEAD)
        if git merge-base --is-ancestor "$sha" "$tip"; then
            release_tip=$tip
            return 0
        fi
        problem "$sha is not in the history of $DEFAULT_BRANCH (now $tip)"
    fi
    return 1
}
