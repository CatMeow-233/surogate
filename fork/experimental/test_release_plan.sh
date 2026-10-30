#!/usr/bin/env bash
# Tests of fork/experimental/release_plan.sh, the gate between a push of an experimental-v* tag and a
# published experimental release. Run with `fork/dev test release` (with the native gate's
# fork/native/release/test_plan.sh); the Fork CI check job runs both.
#
# Each case runs release_plan.sh in a clone of a synthetic fork (origin, with main and tags) and checks
# its exit code, its messages and the outputs it writes to GITHUB_OUTPUT. Everything lives in a
# temporary directory with its own git configuration; the script needs git, bash and python3 (3.11+),
# no network, and changes nothing outside that directory. The last section checks that
# experimental-release.yml still calls release_plan.sh where publishing depends on it.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PLAN=$HERE/release_plan.sh
GATE=$HERE/../tools/release_gate.sh
WORKFLOW=$HERE/../../.github/workflows/experimental-release.yml

T=$(mktemp -d "${TMPDIR:-/tmp}/fork-experimental-plan-test.XXXXXX")
trap 'rm -rf "$T"' EXIT
cat >"$T/gitconfig" <<EOF
[user]
	name = plan test
	email = plan-test@example.invalid
[commit]
	gpgsign = false
[tag]
	gpgsign = false
[init]
	defaultBranch = main
[advice]
	detachedHead = false
EOF
# Only the temporary configuration, and no repository or environment from the caller.
# shellcheck disable=SC2046
unset $(git rev-parse --local-env-vars)
export GIT_CONFIG_GLOBAL=$T/gitconfig GIT_CONFIG_NOSYSTEM=1
unset GITHUB_ACTIONS GITHUB_OUTPUT GITHUB_SHA GITHUB_REF REMOTE DEFAULT_BRANCH

pass=0 fail=0
ok() { pass=$((pass + 1)) && echo "PASS $1"; }
not_ok() { fail=$((fail + 1)) && echo "FAIL $1"; }
check() { # name expected_rc actual_rc output pattern...
    local name=$1 want=$2 got=$3 out=$4 good=1 line
    shift 4
    [ "$want" = "$got" ] || good=0
    for pat in "$@"; do grep -qE -- "$pat" <<<"$out" || { good=0 && echo "    missing /$pat/"; }; done
    if [ $good = 1 ]; then ok "$name (rc=$got)"; else
        not_ok "$name (want rc=$want, got $got)" && while IFS= read -r line; do echo "    | $line"; done <<<"$out"; fi
}

# Fork: main holds the release commit R (version 0.1.0a4) after a .dev0 commit D.
git init -q "$T/seed" && cd "$T/seed" || exit 1
mkdir -p fork/experimental fork/tools
cp "$PLAN" fork/experimental/release_plan.sh
cp "$GATE" fork/tools/release_gate.sh
version() { printf '[project]\nname = "surogate-experimental"\nversion = "%s"\n' "$1" >fork/experimental/pyproject.toml; }
version 0.1.0a4.dev0 && git add -A && git commit -qm D
D=$(git rev-parse HEAD)
version 0.1.0a4 && git commit -qam R
R=$(git rev-parse HEAD)
git clone -q --bare "$T/seed" "$T/origin.git"
git clone -q "$T/origin.git" "$T/work"
cd "$T/work" || exit 1

run() { # mode [VAR=value...]: runs release_plan.sh in $T/work, prints its output, returns its exit code
    local mode=$1
    shift
    env DEFAULT_BRANCH=main GITHUB_OUTPUT="$T/out" "$@" bash fork/experimental/release_plan.sh "$mode" 2>&1
}
out_of() { grep -E "^$1=" "$T/out" | tail -1 | cut -d= -f2-; }
push_tag() { # tag commit [annotated]: (re)creates the tag on origin only
    if [ -n "${3:-}" ]; then git tag -f -a -m "$1" "$1" "$2" >/dev/null; else git tag -f "$1" "$2" >/dev/null; fi
    git push -q -f origin "refs/tags/$1" && git tag -d "$1" >/dev/null
}
TAG=experimental-v0.1.0a4
REF=refs/tags/$TAG

# A. a tag push at the release commit on main publishes
push_tag $TAG "$R"
: >"$T/out"
o=$(run publish GITHUB_SHA="$R" GITHUB_REF=$REF)
check "A tag push on main" 0 $? "$o" "version 0\.1\.0a4, tag $TAG, commit $R, publish true"
if [ "$(out_of publish)" = true ] && [ "$(out_of tag)" = $TAG ] && [ "$(out_of version)" = 0.1.0a4 ]; then
    ok "A outputs: $(tr '\n' ' ' <"$T/out")"
else not_ok "A outputs" && cat "$T/out"; fi
o=$(run recheck GITHUB_SHA="$R" GITHUB_REF=$REF)
check "A recheck" 0 $? "$o" "publish true"

# B. an annotated tag names the same commit
push_tag $TAG "$R" annotated
o=$(run publish GITHUB_SHA="$R" GITHUB_REF=$REF)
check "B annotated tag" 0 $? "$o" "publish true"

# C. main moved on after the tag: the tagged commit is still in its history
echo m >m.txt && git add m.txt && git commit -qm M && git push -q origin main
o=$(run publish GITHUB_SHA="$R" GITHUB_REF=$REF)
check "C main moved on" 0 $? "$o" "publish true"

# D. the finding's case: a tag at a feature-branch commit that is not on main
git checkout -q -b feature "$R" && echo f >f.txt && git add f.txt && git commit -qm F && git push -q origin feature
F=$(git rev-parse HEAD)
push_tag $TAG "$F"
o=$(run publish GITHUB_SHA="$F" GITHUB_REF=$REF)
check "D tag on a feature branch" 1 $? "$o" "error: $F is not in the history of main" "refusing to publish $TAG"
git checkout -q main
push_tag $TAG "$R"

# E. the tag moved or was deleted after the push: the release job's re-check refuses
push_tag $TAG "$D"
o=$(run recheck GITHUB_SHA="$R" GITHUB_REF=$REF)
check "E tag moved after the push" 1 $? "$o" "$TAG on origin now names $D, not the run's commit $R"
git push -q origin ":refs/tags/$TAG"
o=$(run recheck GITHUB_SHA="$R" GITHUB_REF=$REF)
check "E tag deleted after the push" 1 $? "$o" "$TAG does not exist on origin"
push_tag $TAG "$R"

# F. versions: a .dev0 checkout version, the tag of another version, other spellings
push_tag experimental-v0.1.0a4.dev0 "$D"
o=$(run publish GITHUB_SHA="$D" GITHUB_REF=refs/tags/experimental-v0.1.0a4.dev0)
check "F tag of a .dev0 version" 1 $? "$o" "version 0\.1\.0a4\.dev0 is a development release"
push_tag experimental-v0.1.0a5 "$R"
o=$(run publish GITHUB_SHA="$R" GITHUB_REF=refs/tags/experimental-v0.1.0a5)
check "F tag of another version" 1 $? "$o" "tag experimental-v0.1.0a5 does not match .* \(experimental-v0\.1\.0a4\)"
for v in 0.1.0.a4 0.1.0a04 0.1.0-alpha4 0.1.0a4+local; do
    git checkout -q -B vtest "$R" && version "$v" && git commit -qam "v $v" && git push -q -f origin vtest:main
    push_tag "experimental-v$v" "$(git rev-parse HEAD)"
    o=$(run publish GITHUB_SHA="$(git rev-parse HEAD)" GITHUB_REF="refs/tags/experimental-v$v")
    check "F version $v" 1 $? "$o" "not a PEP 440 version without a local part in canonical spelling"
done
git push -q -f origin "$(git rev-parse main):main" && git checkout -q main
git checkout -q -B nover "$R" && git rm -q fork/experimental/pyproject.toml && git commit -qm nover
o=$(run build GITHUB_SHA="$(git rev-parse HEAD)" GITHUB_REF=$REF)
check "F no pyproject.toml" 0 $? "$o" "cannot read the version from fork/experimental/pyproject.toml" "version \?"
git checkout -q main

# G. not a tag push: a build run reports, a publish run refuses
: >"$T/out"
o=$(run build GITHUB_SHA="$R" GITHUB_REF=refs/heads/main)
check "G build on main" 0 $? "$o" \
    "notice: a publish run would refuse: the run is on refs/heads/main; only a push of a tag" "tag none" "publish false"
if [ "$(out_of publish)" = false ] && grep -q '^tag=$' "$T/out"; then ok "G build outputs: $(tr '\n' ' ' <"$T/out")"
else not_ok "G build outputs" && cat "$T/out"; fi
o=$(run build GITHUB_SHA="$R" GITHUB_REF=$REF)
check "G build on the tag (manual run)" 0 $? "$o" "publish false"
o=$(run publish GITHUB_SHA="$R" GITHUB_REF=refs/tags/v1.5.12)
check "G another tag" 1 $? "$o" "the run is on refs/tags/v1\.5\.12; only a push of a tag experimental-v"

# H. a shallow clone cannot prove ancestry; an unreachable remote fails closed
git clone -q --depth 1 "file://$T/origin.git" "$T/shallow"
o=$(cd "$T/shallow" && env DEFAULT_BRANCH=main GITHUB_REF=$REF GITHUB_SHA="$(git -C "$T/shallow" rev-parse HEAD)" \
    bash fork/experimental/release_plan.sh publish 2>&1)
check "H shallow clone" 1 $? "$o" "the clone is shallow"
o=$(run publish GITHUB_SHA="$R" GITHUB_REF=$REF REMOTE=nowhere)
check "H origin unreachable" 1 $? "$o" "cannot list the tags of nowhere" "cannot fetch main from nowhere"

# I. GitHub Actions annotations, the usage error, and no moved ref
o=$(run publish GITHUB_ACTIONS=true GITHUB_SHA="$F" GITHUB_REF=$REF)
check "I Actions annotations" 1 $? "$o" "^::error::.* is not in the history of main" "^::error::refusing to publish"
o=$(run frobnicate)
check "I usage" 2 $? "$o" "usage: fork/experimental/release_plan.sh build\|publish\|recheck"
refs_before=$(git for-each-ref)
o=$(run recheck GITHUB_SHA="$R" GITHUB_REF=$REF)
if [ "$(git for-each-ref)" = "$refs_before" ]; then ok "I release_plan.sh moved no ref"; else not_ok "I moved a ref"; fi

# W. experimental-release.yml decides in the build job before testing and building, publishes only from a
# tag push the plan accepted, and re-checks in the release job before uploading
line_of() { grep -nF -- "$1" "$WORKFLOW" | head -1 | cut -d: -f1; }
# shellcheck disable=SC2016 # the workflow's literal text
plan_at=$(line_of 'run: fork/experimental/release_plan.sh "$MODE"')
mode_at=$(line_of "MODE: \${{ github.event_name == 'push' && 'publish' || 'build' }}")
tests_at=$(line_of 'run: fork/dev test experimental')
release_at=$(line_of "needs.build.outputs.publish == 'true'")
recheck_at=$(line_of 'run: fork/experimental/release_plan.sh recheck')
upload_at=$(line_of 'uses: softprops/action-gh-release@')
if [ -n "$plan_at" ] && [ -n "$mode_at" ] && [ -n "$tests_at" ] && [ "$plan_at" -lt "$tests_at" ]; then
    ok "W the build job plans first, in publish mode only for a push"
else not_ok "W build job: no 'release_plan.sh \"\$MODE\"' step before the tests with MODE from the event"; fi
if [ -n "$release_at" ] && [ -n "$recheck_at" ] && [ -n "$upload_at" ] &&
    [ "$release_at" -lt "$recheck_at" ] && [ "$recheck_at" -lt "$upload_at" ]; then
    ok "W the release job runs only on the plan's word and re-checks before softprops"
else not_ok "W release job: no publish condition on the plan output, or no recheck before softprops"; fi
if [ "$(grep -cE '^ +fetch-depth: 0$' "$WORKFLOW")" = 2 ]; then ok "W both jobs check out the full history"
else not_ok "W both checkouts need fetch-depth: 0"; fi

echo "release_plan.sh tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
