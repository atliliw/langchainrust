//! `lc run` — locate the nearest Cargo.toml and run the current project via cargo.
//!
//! This deliberately does not re-implement a build system: it forwards to cargo
//! so behavior never diverges from `cargo run`.

use std::path::PathBuf;
use std::process::Command;

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let manifest = locate_manifest()?;
    let dir = manifest
        .parent()
        .ok_or_else(|| "Cargo.toml has no parent directory".to_string())?;

    let mut cmd = Command::new("cargo");
    cmd.arg("run").arg("--manifest-path").arg(&manifest);
    if !args.is_empty() {
        cmd.arg("--").args(args);
    }
    cmd.current_dir(dir);

    let status = cmd.status()?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

/// Walks up the current directory looking for a `Cargo.toml`.
fn locate_manifest() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut dir = std::env::current_dir()?;
    loop {
        let candidate = dir.join("Cargo.toml");
        if candidate.exists() {
            return Ok(candidate);
        }
        if !dir.pop() {
            return Err(
                "no Cargo.toml found in this directory or any parent".into(),
            );
        }
    }
}