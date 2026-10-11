// lc-tools/src/sandbox/wasm.rs
//! WASM interpreter sandbox backend (Track 3 #201).
//!
//! Backend that actually delivers what the `CodeSandbox` trait promises to
//! [`LocalSandbox`](super::LocalSandbox) users: OS isolation *by construction*. A WASI
//! preview1 module is compiled and run inside the [`wasmi`] interpreter — a single program
//! counter, no host OS access — with a **self-contained WASI host** covering the stream the
//! trait's contract needs:
//!
//! | WASI import | purpose |
//! |---|---|
//! | `fd_write` / `fd_close` | capture stdout (fd 1) and stderr (fd 2) into [`RunResult`] |
//! | `proc_exit` | capture the guest exit code |
//! | `args_*` / `environ_*` | expose the (empty) process environment |
//! | `random_get` | seeded PRNG for guests that expect nondeterminism |
//! | `clock_time_get` / `clock_res_get` | wall clock for time-sensitive guests |
//!
//! Everything the guest prints crosses the host boundary only as bytes appended to a buffer
//! — there is no filesystem, network, socket, or process tree to reach, so a hostile module
//! cannot move laterally the way a raw `python`/`node` subprocess could. Construction of the
//! engine enables wasm **fuel metering**, and `run` budgets a deterministic number of fuel
//! units from `timeout_ms`; exhausting fuel raises `wasmi`'s out-of-fuel error, which we map
//! to [`SandboxError::Timeout`]. No wall-clock thread is needed, so the budget is reproducible
//! and offline.
//!
//! # Scope, honestly
//!
//! The supported input is a small, hand-authored (or toolchain-generated) **WAT / wasm
//! module** that stays within the minimal import surface above. A full Rust-toolchain libc
//! program (which `wasm32-wasi` links against dozens of `fd_read`/`path_open`-style syscalls)
//! and Python / JavaScript source are **not** in scope: they report
//! [`SandboxError::UnsupportedLanguage`]. This keeps the backend real and dependency-light
//! ([`wasmi`], pure interpreter — no cranelift/JIT) rather than a hollow promise of "runs
//! anything". [`Language::Wasm`] is the channel for it.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;

use super::{CodeSandbox, Language, RunResult, SandboxError};

/// WASI preview1 import-module namespace.
const WASI_PV1: &str = "wasi_snapshot_preview1";

/// Number of wasm "instructions" a single millisecond of budget buys. Fuel is a coarse
/// meter (charges per executed instruction/branch), so this is a budget, not a precise timer.
const INSTRUCTIONS_PER_MS: u64 = 1_000;

/// WASI errno we emit when a memory access or fd is invalid.
const EIO_ERRNO: i32 = 5;

/// The interpreter-backed WASM sandbox.
///
/// Construct with [`WasmSandbox::new`]; the engine is built with fuel metering enabled so
/// every [`run`](CodeSandbox::run) can budget deterministic execution time.
pub struct WasmSandbox {
    engine: wasmi::Engine,
}

/// Per-execution host state shared between the WASI functions and the runner.
struct WasiState {
    /// Guest arguments; seeded with a single `argv[0]` so `args_sizes_get` is well-formed.
    args: Vec<Vec<u8>>,
    /// Captured stdout (fd 1).
    stdout: Vec<u8>,
    /// Captured stderr (fd 2).
    stderr: Vec<u8>,
    /// Exit code left by `proc_exit` (0 unless the guest sets it).
    exit_code: i32,
    /// PRNG state for `random_get` (xorshift-style).
    random_state: u64,
}

impl WasiState {
    fn new() -> Self {
        Self {
            args: vec![b"wasm".to_vec()],
            stdout: Vec::new(),
            stderr: Vec::new(),
            exit_code: 0,
            random_state: 0x4d595_df4d0f33173,
        }
    }
}

impl WasmSandbox {
    /// Create a sandbox whose engine meters fuel (deterministic budget for [`run`](Self::run)).
    pub fn new() -> Self {
        let mut config = wasmi::Config::default();
        config.consume_fuel(true);
        let engine = wasmi::Engine::new(&config);
        Self { engine }
    }
}

impl Default for WasmSandbox {
    fn default() -> Self {
        Self::new()
    }
}

/// Does a `wasmi` error mean the fuel budget ran out (→ timeout)?
fn is_out_of_fuel(e: &wasmi::Error) -> bool {
    let s = e.to_string();
    s.contains("out of fuel") || s.contains("all fuel consumed")
}

/// Read a `u32` (little-endian) from guest memory at `offset`, else a WASI errno.
fn read_u32(
    mem: &wasmi::Memory,
    ctx: impl wasmi::AsContextMut,
    offset: usize,
) -> Result<u32, i32> {
    let mut b = [0u8; 4];
    mem.read(ctx, offset, &mut b).map_err(|_| EIO_ERRNO)?;
    Ok(u32::from_le_bytes(b))
}

/// Publish the WASI host functions onto `linker`.
///
/// Only the narrow surface the trait contract needs is defined; any other import a module
/// declares fails instantiation with a [`SandboxError::Runtime`] at run time, which is the
/// honest behaviour for the supported subset.
fn register_wasi(linker: &mut wasmi::Linker<WasiState>) -> Result<(), wasmi::errors::LinkerError> {
    // fd_write(fd, iovec_ptr, iovec_len, nwritten_out) -> errno
    linker.func_wrap(
        WASI_PV1,
        "fd_write",
        |mut caller: wasmi::Caller<'_, WasiState>,
         fd: i32,
         iovs: i32,
         iovs_len: i32,
         nwritten: i32|
         -> i32 {
            let Some(mem) = caller.get_export("memory").and_then(wasmi::Extern::into_memory)
            else {
                return EIO_ERRNO;
            };
            let mut bytes: Vec<u8> = Vec::new();
            for i in 0..iovs_len {
                let off = (iovs + i * 8) as usize;
                let ptr = match read_u32(&mem, &mut caller, off) {
                    Ok(v) => v as usize,
                    Err(e) => return e,
                };
                let len = match read_u32(&mem, &mut caller, off + 4) {
                    Ok(v) => v as usize,
                    Err(e) => return e,
                };
                let mut chunk = vec![0u8; len];
                if mem.read(&mut caller, ptr, &mut chunk).is_err() {
                    return EIO_ERRNO;
                }
                bytes.extend_from_slice(&chunk);
            }
            match fd {
                1 => caller.data_mut().stdout.extend_from_slice(&bytes),
                2 => caller.data_mut().stderr.extend_from_slice(&bytes),
                // Only stdout/stderr are exposed; any other fd is a hard error.
                _ => return EIO_ERRNO,
            }
            let count = bytes.len() as u32;
            if mem.write(&mut caller, nwritten as usize, &count.to_le_bytes()).is_err() {
                return EIO_ERRNO;
            }
            0
        },
    )?;

    // fd_close(_fd) -> errno: this host exposes no open fds beyond stdio, so it's a no-op
    // success for the descriptors we hand out.
    linker.func_wrap(WASI_PV1, "fd_close", |_: wasmi::Caller<'_, WasiState>, _fd: i32| -> i32 {
        0
    })?;

    // proc_exit(code): capture the exit code; the guest is expected to return afterwards
    // (a well-formed `_start` ends right after calling it).
    linker.func_wrap(WASI_PV1, "proc_exit", |mut caller: wasmi::Caller<'_, WasiState>, code: i32| {
        caller.data_mut().exit_code = code;
    })?;

    // args_sizes_get(argc_out, argv_buf_size_out) -> errno
    linker.func_wrap(
        WASI_PV1,
        "args_sizes_get",
        |mut caller: wasmi::Caller<'_, WasiState>, argc_out: i32, buf_size_out: i32| -> i32 {
            let Some(mem) =
                caller.get_export("memory").and_then(wasmi::Extern::into_memory)
            else {
                return EIO_ERRNO;
            };
            let args = caller.data().args.clone();
            let argc = args.len() as u32;
            let buf_size: u32 = args.iter().fold(0, |acc, a| acc + a.len() as u32 + 1);
            if mem.write(&mut caller, argc_out as usize, &argc.to_le_bytes()).is_err() {
                return EIO_ERRNO;
            }
            if mem.write(&mut caller, buf_size_out as usize, &buf_size.to_le_bytes()).is_err() {
                return EIO_ERRNO;
            }
            0
        },
    )?;

    // args_get(argv_ptr_out, argv_buf_out) -> errno
    linker.func_wrap(
        WASI_PV1,
        "args_get",
        |mut caller: wasmi::Caller<'_, WasiState>, argv_out: i32, argv_buf: i32| -> i32 {
            let Some(mem) =
                caller.get_export("memory").and_then(wasmi::Extern::into_memory)
            else {
                return EIO_ERRNO;
            };
            let args = caller.data().args.clone();
            // `argv_buf` holds the argument bytes (null-terminated); the array at `argv_out`
            // holds a `u32` pointer per argument.
            let mut data_cursor = argv_buf as usize;
            for (i, a) in args.iter().enumerate() {
                if mem.write(&mut caller, data_cursor, a).is_err()
                    || mem.write(&mut caller, data_cursor + a.len(), &[0u8]).is_err()
                {
                    return EIO_ERRNO;
                }
                if mem
                    .write(&mut caller, argv_out as usize + i * 4, &(data_cursor as u32).to_le_bytes())
                    .is_err()
                {
                    return EIO_ERRNO;
                }
                data_cursor += a.len() + 1;
            }
            0
        },
    )?;

    // environ: empty.
    linker.func_wrap(
        WASI_PV1,
        "environ_sizes_get",
        |mut caller: wasmi::Caller<'_, WasiState>, envc_out: i32, buf_out: i32| -> i32 {
            let Some(mem) =
                caller.get_export("memory").and_then(wasmi::Extern::into_memory)
            else {
                return EIO_ERRNO;
            };
            let _ = mem.write(&mut caller, envc_out as usize, &0u32.to_le_bytes());
            let _ = mem.write(&mut caller, buf_out as usize, &0u32.to_le_bytes());
            0
        },
    )?;
    linker.func_wrap(WASI_PV1, "environ_get", |_: wasmi::Caller<'_, WasiState>, _: i32, _: i32| -> i32 {
        0
    })?;

    // random_get(buf, len) -> errno: deterministic xorshift fill.
    linker.func_wrap(
        WASI_PV1,
        "random_get",
        |mut caller: wasmi::Caller<'_, WasiState>, buf: i32, len: i32| -> i32 {
            let Some(mem) = caller.get_export("memory").and_then(wasmi::Extern::into_memory) else {
                return EIO_ERRNO;
            };
            let mut state = caller.data_mut().random_state;
            for i in 0..len as usize {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let byte = (state & 0xff) as u8;
                if mem.write(&mut caller, buf as usize + i, &[byte]).is_err() {
                    return EIO_ERRNO;
                }
            }
            caller.data_mut().random_state = state;
            0
        },
    )?;

    // clock_time_get(_id, _precision, out) -> errno
    linker.func_wrap(
        WASI_PV1,
        "clock_time_get",
        |mut caller: wasmi::Caller<'_, WasiState>, _id: i32, _precision: i32, out: i32| -> i32 {
            let Some(mem) = caller.get_export("memory").and_then(wasmi::Extern::into_memory) else {
                return EIO_ERRNO;
            };
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            let _ = mem.write(&mut caller, out as usize, &nanos.to_le_bytes());
            0
        },
    )?;

    // clock_res_get(_id, out) -> errno (1ms granularity).
    linker.func_wrap(
        WASI_PV1,
        "clock_res_get",
        |mut caller: wasmi::Caller<'_, WasiState>, _id: i32, out: i32| -> i32 {
            let Some(mem) = caller.get_export("memory").and_then(wasmi::Extern::into_memory) else {
                return EIO_ERRNO;
            };
            let _ = mem.write(&mut caller, out as usize, &1_000_000u64.to_le_bytes());
            0
        },
    )?;

    Ok(())
}

#[async_trait]
impl CodeSandbox for WasmSandbox {
    async fn run(
        &self,
        code: &str,
        language: Language,
        timeout_ms: u64,
    ) -> Result<RunResult, SandboxError> {
        if language != Language::Wasm {
            return Err(SandboxError::UnsupportedLanguage(language.to_string()));
        }

        let start = Instant::now();

        // `wat` accepts both text (`(module ...)`) and already-binary wasm.
        let wasm_bytes =
            wat::parse_str(code).map_err(|e| SandboxError::Runtime(format!("invalid wasm: {e}")))?;
        let module = wasmi::Module::new(&self.engine, &mut &wasm_bytes[..])
            .map_err(|e| SandboxError::Runtime(format!("wasm compile: {e}")))?;

        let mut store = wasmi::Store::new(&self.engine, WasiState::new());
        let mut linker = wasmi::Linker::<WasiState>::new(&self.engine);
        register_wasi(&mut linker)
            .map_err(|e| SandboxError::Runtime(format!("wasi link: {e}")))?;

        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|e| SandboxError::Runtime(format!("wasm instantiation: {e}")))?
            .start(&mut store)
            .map_err(|e| SandboxError::Runtime(format!("wasm start: {e}")))?;

        // Deterministic execution budget derived from the caller's timeout. Fuel is consumed
        // per instruction; exhausting it traps as out-of-fuel → SandboxError::Timeout.
        let fuel = timeout_ms
            .saturating_mul(INSTRUCTIONS_PER_MS)
            .max(INSTRUCTIONS_PER_MS);
        store
            .add_fuel(fuel)
            .map_err(|e| SandboxError::Runtime(format!("fuel budget: {e}")))?;

        // Run the program's `_start`. Missing `_start` (a pure library) is not an error; use
        // the dynamic `Func` path so a `_start` that returns an errno still runs and we can
        // recover its exit code via `proc_exit`.
        if let Some(start) = instance
            .get_export(&store, "_start")
            .and_then(wasmi::Extern::into_func)
        {
            if let Err(e) = start.call(&mut store, &[], &mut []) {
                return if is_out_of_fuel(&e) {
                    Err(SandboxError::Timeout(timeout_ms))
                } else {
                    Err(SandboxError::Runtime(format!("wasm runtime: {e}")))
                };
            }
        }

        let execution_time_ms = start.elapsed().as_millis() as u64;
        let state = store.into_data();
        Ok(RunResult {
            stdout: String::from_utf8_lossy(&state.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&state.stderr).into_owned(),
            exit_code: state.exit_code,
            execution_time_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small WASI module that prints to stdout and exits 0.
    const HELLO_WAT: &str = r#"
        (module
            (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
            (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
            (memory 1)
            (data (i32.const 16) "Hello from WASM!\n")
            (export "memory" (memory 0))
            (func $_start
                ;; iovec: ptr=16, len=17 (incl. trailing newline)
                (i32.store (i32.const 0) (i32.const 16))
                (i32.store (i32.const 4) (i32.const 17))
                (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 32))
                (drop)
                (call $proc_exit (i32.const 0))
            )
            (export "_start" (func $_start))
        )
    "#;

    /// Same, writing fd 2 (stderr) with a non-zero exit code.
    const STDERR_WAT: &str = r#"
        (module
            (import "wasi_snapshot_preview1" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
            (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
            (memory 1)
            (data (i32.const 16) "oops\n")
            (export "memory" (memory 0))
            (func $_start
                (i32.store (i32.const 0) (i32.const 16))
                (i32.store (i32.const 4) (i32.const 5))
                (call $fd_write (i32.const 2) (i32.const 0) (i32.const 1) (i32.const 32))
                (drop)
                (call $proc_exit (i32.const 7))
            )
            (export "_start" (func $_start))
        )
    "#;

    /// Busy-loop that burns fuel forever → must surface as Timeout.
    const LOOP_WAT: &str = r#"(module (func $_start (loop $l (br $l))) (export "_start" (func $_start)))"#;

    #[tokio::test]
    async fn runs_wat_module_and_captures_stdout() {
        let sandbox = WasmSandbox::new();
        let result = sandbox.run(HELLO_WAT, Language::Wasm, 1_000).await;
        let run = result.expect("module should run");
        assert_eq!(run.stdout, "Hello from WASM!\n");
        assert_eq!(run.stderr, "");
        assert_eq!(run.exit_code, 0);
        assert!(run.execution_time_ms < 1_000);
    }

    #[tokio::test]
    async fn captures_stderr_and_exit_code() {
        let sandbox = WasmSandbox::new();
        let result = sandbox.run(STDERR_WAT, Language::Wasm, 1_000).await;
        let run = result.expect("module should run");
        assert_eq!(run.stdout, "");
        assert_eq!(run.stderr, "oops\n");
        assert_eq!(run.exit_code, 7);
    }

    #[tokio::test]
    async fn loops_until_fuel_exhausts_maps_to_timeout() {
        let sandbox = WasmSandbox::new();
        // A tiny budget so the busy-loop burns out immediately.
        let result = sandbox.run(LOOP_WAT, Language::Wasm, 1).await;
        match result {
            Err(SandboxError::Timeout(ms)) => assert_eq!(ms, 1),
            other => panic!("expected Timeout, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn rejects_non_wasm_languages() {
        let sandbox = WasmSandbox::new();
        for (code, lang) in [
            ("print(1)", Language::Python),
            ("console.log(1)", Language::JavaScript),
            ("fn main(){}", Language::Rust),
        ] {
            let result = sandbox.run(code, lang, 1_000).await;
            match result {
                Err(SandboxError::UnsupportedLanguage(msg)) => {
                    assert_eq!(msg, lang.to_string());
                }
                other => panic!("expected UnsupportedLanguage, got: {:?}", other),
            }
        }
    }

    #[tokio::test]
    async fn invalid_wat_reports_runtime_error() {
        let sandbox = WasmSandbox::new();
        let result = sandbox.run("(not a module", Language::Wasm, 1_000).await;
        match result {
            Err(SandboxError::Runtime(msg)) => assert!(msg.contains("invalid wasm")),
            other => panic!("expected Runtime, got: {:?}", other),
        }
    }
}