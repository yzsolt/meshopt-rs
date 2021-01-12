# meshopt-rs

**Work in progress** pure Rust implementation of the awesome [meshoptimizer](https://github.com/zeux/meshoptimizer) library. Will be published on `crates.io` once all v0.15 features are ported.

If you want to use the original C++ implementation from Rust, check out the [meshopt crate](https://crates.io/crates/meshopt).

## Features

`meshoptimizer` v0.15 feature level is the current support target.

Experimental features (hidden behind `MESHOPTIMIZER_EXPERIMENTAL` in the original implementation) can be enabled with the `experimental` Cargo feature:

```toml
[dependencies]
meshopt-rs = { version = "0.1", features = ["experimental"] }
```

## Contributing

`meshopt-rs` is licensed under MIT, just like `meshoptimizer`. Contributions are welcome!

Since this is a parallel implementation of an existing and actively developed library, the original implementation is followed as closely as possible: similar naming, documentation and code structure; to help porting new features and bug fixes in the future.
