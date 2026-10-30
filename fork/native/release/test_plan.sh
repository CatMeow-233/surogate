#!/usr/bin/env bash
# Tests of fork/native/release/plan.sh, the only gate between a run of
# .github/workflows/native-release.yml and a published release. Run with `fork/dev test release`;
# the Fork CI check job runs it too.
#
# Each case runs plan.sh in a clone of a synthetic fork and checks its exit code, its messages and
# the outputs it writes to GITHUB_OUTPUT. The synthetic upstream (vX.Y.Z tags) and fork (origin,
# with main) live in a temporary directory, with their own git configuration; the script needs only
# git and bash, no network, and changes nothing outside that directory. The last section checks
# that native-release.yml still calls plan.sh where publishing depends on it.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PLAN=$HERE/plan.sh
WORKFLOW=$HERE/../../../.github/workflows/native-release.yml

T=$(mktemp -d "${TMPDIR:-/tmp}/fork-plan-test.XXXXXX")
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
# Only the temporary configuration, and no repository or environment from the caller (for example
# GIT_DIR when this runs from a git hook).
# shellcheck disable=SC2046
unset $(git rev-parse --local-env-vars)
export GIT_CONFIG_GLOBAL=$T/gitconfig GIT_CONFIG_NOSYSTEM=1
unset GITHUB_ACTIONS GITHUB_OUTPUT GITHUB_SHA GITHUB_REF RELEASE_TAG REMOTE UPSTREAM_URL UPSTREAM_BRANCH \
    DEFAULT_BRANCH

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

# Upstream: v1.5.9 (lightweight), v1.5.11 (lightweight), v1.5.12 (annotated), then v1.5.13
# (annotated) and v1.6.0rc1, and two untagged commits U4 and U5 on main, that the fork has not merged.
git init -q "$T/up" && cd "$T/up" || exit 1
c() { echo "$1" >"$2" && git add "$2" && git commit -qm "$1"; }
c u0 up.txt && git tag v1.5.9
c u1 up.txt && git tag v1.5.11 && git tag jackalope-v9
c u2 up.txt && git tag -a v1.5.12 -m v1.5.12
c u3 up.txt && git tag -a v1.5.13 -m v1.5.13 && git tag v1.6.0rc1
c u4 up.txt && U4=$(git rev-parse HEAD)
c u5 up.txt && U5=$(git rev-parse HEAD)
git clone -q --bare "$T/up" "$T/up.git"
U2=$(git -C "$T/up" rev-parse 'v1.5.12^{commit}')

# Fork: main = upstream v1.5.12 + a fork commit with VERSION, plan.sh and a workflow.
git init -q "$T/seed" && cd "$T/seed" || exit 1
git fetch -q "$T/up.git" refs/tags/v1.5.12 && git reset -q --hard "$U2"
mkdir -p fork/native/release .github/workflows
echo 1.5.12.post1 >fork/native/release/VERSION
cp "$PLAN" fork/native/release/plan.sh
echo 'on: workflow_dispatch' >.github/workflows/native-release.yml
git add -A && git commit -qm F1
F1=$(git rev-parse HEAD)
git clone -q --bare "$T/seed" "$T/origin.git"
git -C "$T/origin.git" tag -l | xargs -r git -C "$T/origin.git" tag -d >/dev/null # the fork has no v* tags
git clone -q "$T/origin.git" "$T/work"
cd "$T/work" || exit 1

run() { # mode [VAR=value...]: runs plan.sh in $T/work, prints its output, returns its exit code
    local mode=$1
    shift
    env DEFAULT_BRANCH=main UPSTREAM_URL="$T/up.git" GITHUB_OUTPUT="$T/out" "$@" \
        bash fork/native/release/plan.sh "$mode" 2>&1
}
out_of() { grep -E "^$1=" "$T/out" | tail -1 | cut -d= -f2-; }

# A. publish on the main tip: allowed
: >"$T/out"
o=$(run publish GITHUB_SHA="$F1" GITHUB_REF=refs/heads/main)
check "A publish on the main tip" 0 $? "$o" "upstream v1\.5\.12 at $U2, publish true"
if [ "$(out_of publish)" = true ] && [ "$(out_of release_tag)" = native-v1.5.12.post1 ] &&
    [ "$(out_of upstream_base)" = 1.5.12 ] && [ "$(out_of version)" = 1.5.12.post1 ] &&
    [ "$(out_of upstream_commit)" = "$U2" ] && [ "$(out_of upstream_describe)" = v1.5.12 ]; then
    ok "A outputs: $(tr '\n' ' ' <"$T/out")"
else not_ok "A outputs" && cat "$T/out"; fi

# B. a feature branch: refused in publish, reported in build
git checkout -q -b feature && echo f >f.txt && git add f.txt && git commit -qm F2 && git push -q origin feature
F2=$(git rev-parse HEAD)
o=$(run publish GITHUB_SHA="$F2" GITHUB_REF=refs/heads/feature)
check "B publish from a feature branch" 1 $? "$o" "the run is on refs/heads/feature" \
    "not in the history of main" "refusing to publish"
: >"$T/out"
o=$(run build GITHUB_SHA="$F2" GITHUB_REF=refs/heads/feature)
check "B build on a feature branch" 0 $? "$o" \
    "notice: a publish run would refuse: the run is on refs/heads/feature" "publish false"
if [ "$(out_of publish)" = false ] && grep -q '^release_tag=$' "$T/out"; then
    ok "B build outputs: $(tr '\n' ' ' <"$T/out")"
else not_ok "B build outputs" && cat "$T/out"; fi
git checkout -q main

# C. a dispatch on a tag ref is not the default branch
o=$(run publish GITHUB_SHA="$F1" GITHUB_REF=refs/tags/whatever)
check "C publish on a tag ref" 1 $? "$o" "the run is on refs/tags/whatever"

# D. main moved on without touching .github/workflows: the dispatched commit may still publish
echo m >m.txt && git add m.txt && git commit -qm M2 && git push -q origin main
o=$(run publish GITHUB_SHA="$F1" GITHUB_REF=refs/heads/main)
check "D main moved, same workflows" 0 $? "$o" "publish true"

# E. main moved and changed a workflow file: GITHUB_TOKEN could not create the tag at F1 (R2-NR-4)
echo 'on: push' >.github/workflows/other.yml && git add -A && git commit -qm W && git push -q origin main
o=$(run publish GITHUB_SHA="$F1" GITHUB_REF=refs/heads/main)
check "E main changed .github/workflows" 1 $? "$o" "\.github/workflows at $F1 differs from main"
o=$(run recheck GITHUB_SHA="$F1" GITHUB_REF=refs/heads/main RELEASE_TAG=native-v1.5.12.post1)
check "E recheck too" 1 $? "$o" "differs from main"
WTIP=$(git rev-parse HEAD)
o=$(run publish GITHUB_SHA="$WTIP" GITHUB_REF=refs/heads/main)
check "E a new run on the new tip" 0 $? "$o" "publish true"

# F. main was force-pushed and no longer contains the commit
git checkout -q -b rewritten "$U2" && echo x >x.txt && git add x.txt && git commit -qm X &&
    git push -q -f origin rewritten:main
git checkout -q main
o=$(run publish GITHUB_SHA="$WTIP" GITHUB_REF=refs/heads/main)
check "F commit no longer on main" 1 $? "$o" "is not in the history of main"
git push -q -f origin "$WTIP:main"

# G. VERSION checks: format, and the upstream release the commit is based on (NR-3)
for v in 1.5.9.post1 1.5.12 1.5.13.post1 1.5.12.post1+cu130 abc; do
    git checkout -q -B vtest "$WTIP" && echo "$v" >fork/native/release/VERSION && git commit -qam "v $v" &&
        git push -q -f origin vtest:main
    o=$(run publish GITHUB_SHA="$(git rev-parse HEAD)" GITHUB_REF=refs/heads/main)
    rc=$?
    case $v in
        abc | *+*) check "G VERSION $v" 1 $rc "$o" "not a PEP 440 version" ;;
        *) check "G VERSION $v" 1 $rc "$o" "does not name the upstream release .* \(v1\.5\.12\); use 1\.5\.12\.post<N>" ;;
    esac
done
for v in 1.5.12.post2 1.5.12.post3.dev1; do
    git checkout -q -B vtest "$WTIP" && echo "$v" >fork/native/release/VERSION && git commit -qam "v $v" &&
        git push -q -f origin vtest:main
    o=$(run publish GITHUB_SHA="$(git rev-parse HEAD)" GITHUB_REF=refs/heads/main)
    check "G VERSION $v" 0 $? "$o" "tag native-v$v,.*publish true"
done
git push -q -f origin "$WTIP:main" && git checkout -q main

# O. fork/dev sync merges upstream's main, not a tag: the outputs for the release notes name the
# upstream commit the fork merged and how many commits it is past the newest release
for merged in "$U4" "$U5"; do
    git checkout -q -B synced "$WTIP" && git fetch -q "$T/up.git" refs/heads/main &&
        git merge -q --no-ff -m "sync: merge upstream/main" "$merged" &&
        echo 1.5.13.post1 >fork/native/release/VERSION && git commit -qam "v 1.5.13.post1" &&
        git push -q -f origin synced:main
    if [ "$merged" = "$U4" ]; then describe="v1.5.13 + 1 commit"; else describe="v1.5.13 + 2 commits"; fi
    : >"$T/out"
    o=$(run publish GITHUB_SHA="$(git rev-parse HEAD)" GITHUB_REF=refs/heads/main)
    check "O merged upstream $describe" 0 $? "$o" "upstream ${describe//+/\\+} at $merged, publish true"
    if [ "$(out_of upstream_commit)" = "$merged" ] && [ "$(out_of upstream_describe)" = "$describe" ] &&
        [ "$(out_of upstream_base)" = 1.5.13 ]; then ok "O outputs: $(tr '\n' ' ' <"$T/out")"
    else not_ok "O outputs" && cat "$T/out"; fi
done
git push -q -f origin "$WTIP:main" && git checkout -q main

# H. NR-2 / R2-NR-5: two runs planned before either published. The first publishes (creates the
# tag); the second release job's re-check refuses. A re-run of failed jobs reuses plan's outputs
# (RELEASE_TAG) and is refused the same way.
: >"$T/out"
o=$(run publish GITHUB_SHA="$WTIP" GITHUB_REF=refs/heads/main)
check "H plan of run A" 0 $? "$o" "publish true"
o=$(run publish GITHUB_SHA="$WTIP" GITHUB_REF=refs/heads/main)
check "H plan of run B (tag not created yet)" 0 $? "$o" "publish true"
o=$(run recheck GITHUB_SHA="$WTIP" GITHUB_REF=refs/heads/main RELEASE_TAG=native-v1.5.12.post1)
check "H recheck of run A" 0 $? "$o" "publish true"
before=$(wc -l <"$T/out")
git tag native-v1.5.12.post1 "$WTIP" && git push -q origin native-v1.5.12.post1 &&
    git tag -d native-v1.5.12.post1 >/dev/null
o=$(run recheck GITHUB_SHA="$WTIP" GITHUB_REF=refs/heads/main RELEASE_TAG=native-v1.5.12.post1)
check "H recheck of run B after A published" 1 $? "$o" "native-v1.5.12.post1 already exists on origin" \
    "refusing to publish"
if [ "$(wc -l <"$T/out")" = "$before" ]; then ok "H recheck writes no outputs"; else not_ok "H recheck wrote outputs"; fi
o=$(run publish GITHUB_SHA="$WTIP" GITHUB_REF=refs/heads/main)
check "H a later publish run of the same VERSION" 1 $? "$o" "already exists"
o=$(run build GITHUB_SHA="$WTIP" GITHUB_REF=refs/heads/main)
check "H a build-only run is still allowed" 0 $? "$o" "would refuse: native-v1.5.12.post1 already exists" \
    "publish false"
git push -q origin :refs/tags/native-v1.5.12.post1
git tag -a native-v1.5.12.post1 -m x "$WTIP" && git push -q origin native-v1.5.12.post1 &&
    git tag -d native-v1.5.12.post1 >/dev/null
o=$(run recheck GITHUB_SHA="$WTIP" GITHUB_REF=refs/heads/main RELEASE_TAG=native-v1.5.12.post1)
check "H an annotated tag is found too" 1 $? "$o" "already exists"
git push -q origin :refs/tags/native-v1.5.12.post1

# I. recheck: the plan job's tag must match VERSION at the commit
o=$(run recheck GITHUB_SHA="$WTIP" GITHUB_REF=refs/heads/main RELEASE_TAG=native-v1.5.9.post1)
check "I recheck with another plan tag" 1 $? "$o" \
    "the plan job chose 'native-v1.5.9.post1', but VERSION at .* gives native-v1.5.12.post1"
o=$(run recheck GITHUB_SHA="$WTIP" GITHUB_REF=refs/heads/main)
check "I recheck without a plan tag" 1 $? "$o" "the plan job chose ''"

# J. unreachable remotes fail closed
o=$(run publish GITHUB_SHA="$WTIP" GITHUB_REF=refs/heads/main UPSTREAM_URL="$T/missing.git")
check "J upstream unreachable (publish)" 1 $? "$o" "cannot list the tags of $T/missing.git"
o=$(run build GITHUB_SHA="$WTIP" GITHUB_REF=refs/heads/main UPSTREAM_URL="$T/missing.git")
check "J upstream unreachable (build)" 0 $? "$o" "would refuse: cannot list the tags"
o=$(run publish GITHUB_SHA="$WTIP" GITHUB_REF=refs/heads/main REMOTE=nowhere)
check "J origin unreachable (publish)" 1 $? "$o" "cannot list the tags of nowhere" "cannot fetch main from nowhere"
o=$(run publish GITHUB_SHA="$WTIP" GITHUB_REF=refs/heads/main UPSTREAM_BRANCH=gone)
check "J upstream branch missing (publish)" 1 $? "$o" "cannot fetch gone from $T/up.git"

# K. a shallow clone cannot prove ancestry
git clone -q --depth 1 "file://$T/origin.git" "$T/shallow"
cp "$PLAN" "$T/shallow/fork/native/release/plan.sh"
o=$(cd "$T/shallow" && env DEFAULT_BRANCH=main UPSTREAM_URL="$T/up.git" GITHUB_REF=refs/heads/main \
    bash fork/native/release/plan.sh publish 2>&1)
check "K shallow clone" 1 $? "$o" "the clone is shallow"

# L. GitHub Actions annotations, and the usage error
o=$(run publish GITHUB_ACTIONS=true GITHUB_SHA="$F2" GITHUB_REF=refs/heads/feature)
check "L Actions annotations" 1 $? "$o" "^::error::the run is on refs/heads/feature" "^::error::refusing to publish"
o=$(run frobnicate)
check "L usage" 2 $? "$o" "usage: fork/native/release/plan.sh build\|publish\|recheck"

# M. defaults: local HEAD and its branch
git checkout -q main && git reset -q --hard origin/main
o=$(env DEFAULT_BRANCH=main UPSTREAM_URL="$T/up.git" bash fork/native/release/plan.sh publish 2>&1)
check "M local defaults on main" 0 $? "$o" "commit $(git rev-parse HEAD)"

# N. a local preflight moves no ref: origin's main moves behind the clone's back, and plan.sh
# fetches it without updating refs/remotes/origin/main
refs_before=$(git for-each-ref)
git -C "$T/origin.git" update-ref refs/heads/main "$F1"
o=$(run build GITHUB_SHA="$WTIP" GITHUB_REF=refs/heads/main)
check "N build after origin's main moved" 0 $? "$o" "is not in the history of main \(now $F1\)"
if [ "$(git for-each-ref)" = "$refs_before" ]; then ok "N plan.sh moved no ref"; else
    not_ok "N plan.sh moved a ref" && diff <(echo "$refs_before") <(git for-each-ref); fi
git -C "$T/origin.git" update-ref refs/heads/main "$WTIP"

# W. native-release.yml runs plan.sh before the build and again right before publishing, and never
# replaces a published file
line_of() { grep -nF -- "$1" "$WORKFLOW" | head -1 | cut -d: -f1; }
# shellcheck disable=SC2016 # the workflow's literal text
plan_at=$(line_of 'run: fork/native/release/plan.sh "$MODE"')
mode_at=$(line_of "MODE: \${{ inputs.publish && 'publish' || 'build' }}")
release_at=$(line_of "if: needs.plan.outputs.publish == 'true'")
recheck_at=$(line_of 'run: fork/native/release/plan.sh recheck')
publish_at=$(line_of 'uses: softprops/action-gh-release@')
if [ -n "$plan_at" ] && [ -n "$mode_at" ]; then ok "W plan job runs plan.sh in the mode the publish input picks"
else not_ok "W plan job: no 'plan.sh \"\$MODE\"' step with MODE from inputs.publish in $WORKFLOW"; fi
if [ -n "$release_at" ] && [ -n "$recheck_at" ] && [ -n "$publish_at" ] &&
    [ "$release_at" -lt "$recheck_at" ] && [ "$recheck_at" -lt "$publish_at" ]; then
    ok "W the release job re-checks right before softprops"
else not_ok "W release job: no 'plan.sh recheck' step between its publish condition and softprops"; fi
if grep -qE '^ +overwrite_files: false$' "$WORKFLOW"; then ok "W softprops never replaces an asset"
else not_ok "W softprops: overwrite_files is not false"; fi

echo "plan.sh tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
