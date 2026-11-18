//! Integration tests for the `lc` binary. These are tracked by git (unlike
//! `crates/lc/tests/`, which is gitignored).

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A unique scratch directory under the system temp dir.
fn scratch(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("lc-cli-test-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_lc"))
}

#[test]
fn help_lists_all_subcommands() {
    let out = bin().arg("--help").output().unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    for sub in ["init", "run", "serve", "eval"] {
        assert!(text.contains(sub), "--help should mention `{sub}`");
    }
}

#[test]
fn init_creates_project_files() {
    let dir = scratch("init");
    let out = bin()
        .args(["init", dir.to_str().unwrap(), "--template", "agent"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "init failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    for f in ["Cargo.toml", "src/main.rs", ".env.example", "README.md"] {
        assert!(dir.join(f).exists(), "missing {f}");
    }
}

#[test]
fn init_chat_template_uses_openai_chat() {
    let dir = scratch("chat");
    let out = bin()
        .args(["init", dir.to_str().unwrap(), "--template", "chat"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let main = std::fs::read_to_string(dir.join("src/main.rs")).unwrap();
    assert!(
        main.contains("OpenAIChat"),
        "chat template body should reference OpenAIChat"
    );
}

#[test]
fn init_rejects_non_empty_directory() {
    let dir = scratch("nonempty");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("keep.txt"), "x").unwrap();

    let out = bin().args(["init", dir.to_str().unwrap()]).output().unwrap();
    assert!(
        !out.status.success(),
        "init must refuse to overwrite a non-empty directory"
    );
    assert!(dir.join("keep.txt").exists(), "existing file must not be removed");
}

#[test]
fn eval_writes_report() {
    let dir = scratch("eval");
    std::fs::create_dir_all(&dir).unwrap();
    let ds = dir.join("dataset.jsonl");
    std::fs::write(
        &ds,
        "{\"input\":\"q\",\"expected\":\"42\",\"actual\":\"42\"}\n\
         {\"input\":\"q2\",\"expected\":\"42\",\"actual\":\"43\"}\n",
    )
    .unwrap();

    // A model client is constructed even when every line carries `actual`;
    // a dummy key lets construction succeed without any network call.
    let out = bin()
        .env("OPENAI_API_KEY", "test")
        .args([
            "eval",
            "--dataset",
            ds.to_str().unwrap(),
            "--output",
            dir.join("report.json").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "eval failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let report = std::fs::read_to_string(dir.join("report.json")).unwrap();
    assert!(report.contains("\"total\": 2"));
    assert!(report.contains("\"passed\": 1"), "report: {report}");
    assert!(report.contains("\"input\": \"q\""));
}