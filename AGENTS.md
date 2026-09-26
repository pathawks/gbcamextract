# AGENTS.md

## Checks (run before finishing any code change)

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

`cargo fmt --check` must pass with no diff. Fix with `cargo fmt`, do not hand-format.
`cargo clippy --all-targets -- -D warnings` must pass with zero warnings. Do not add `allow`
to silence a new lint without a comment explaining why the lint is a false positive here.
