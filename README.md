# meshopt-rs

[![Crates.io](https://img.shields.io/crates/v/meshopt-rs.svg?label=meshopt-rs)](https://crates.io/crates/meshopt-rs)
[![Docs.rs](https://docs.rs/meshopt-rs/badge.svg)](https://docs.rs/meshopt-rs)
[![Build Status](https://github.com/yzsolt/meshopt-rs/workflows/continuous-integration/badge.svg)](https://github.com/yzsolt/meshopt-rs/actions)

**Work in progress** pure Rust implementation of the awesome [meshoptimizer](https://github.com/zeux/meshoptimizer) library. Will be published on `crates.io` once all v0.15 features are ported.

If you want to use the original C++ implementation from Rust, check out the [meshopt crate](https://crates.io/crates/meshopt).

## Features

`meshoptimizer` v0.15 feature level is the current support target.

Experimental features (hidden behind `MESHOPTIMIZER_EXPERIMENTAL` in the original implementation) can be enabled with the `experimental` Cargo feature:

```toml
[dependencies]
meshopt-rs = { version = "0.1", features = ["experimental"] }
```

## Performance

Depends on the algorithm: some are in the same ballpark as the original, most are slightly (10-20%) and a few are much (50-100%) slower than the original implementation. Only a small amount of performance work has been done so far. Ideally all algorithms should reach at least 90-95% of the original implementation's performance.

Also note that SIMD support (utilized by vertex buffer decoding/filtering) is [currently missing](https://github.com/yzsolt/meshopt-rs/issues/1).

## Contributing

`meshopt-rs` is licensed under MIT, just like `meshoptimizer`. Contributions are welcome!

Since this is a parallel implementation of an existing and actively developed library, the original implementation is followed as closely as possible: similar naming, documentation and code structure; to help porting new features and bug fixes in the future.
