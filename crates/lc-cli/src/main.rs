//! `lc` — command-line interface for langchainrust.
//!
//! Commands:
//! - `lc init <path>`   scaffold a new project (chat / agent / rag)
//! - `lc run`           run the current project (`cargo run` passthrough)
//! - `lc serve`         serve an OpenAI-compatible endpoint
//! - `lc eval`          run an evaluation over a JSONL dataset

use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod eval;
mod init;
mod run;
mod serve;

#[derive(Parser)]
#[command(
    name = "lc",
    version,
    about = "langchainrust CLI — init, run, serve, eval"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Scaffold a new project and print how to run it.
    Init {
        /// Target directory (must be empty or non-existent).
        path: PathBuf,
        /// Template to scaffold: chat | agent | rag.
        #[arg(long, default_value = "agent")]
        template: String,
        /// Provider to pre-fill in the generated .env.example.
        #[arg(long, default_value = "openai")]
        provider: String,
    },
    /// Run the current project (locates the nearest Cargo.toml, then `cargo run`).
    Run {
        /// Arguments forwarded to the program.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Serve an OpenAI-compatible endpoint.
    Serve {
        /// Bind port (bound to 127.0.0.1).
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Model id advertised by /v1/models and used for completions.
        #[arg(long)]
        model: Option<String>,
        /// Backend to serve: model | agent | chain.
        #[arg(long, default_value = "model")]
        backend: String,
        /// Optional Bearer token; when set, every request must carry `Authorization: Bearer <token>`.
        #[arg(long)]
        auth: Option<String>,
        /// Optional port for a sidecar MCP stateless HTTP server (serves `Calculator`).
        #[arg(long)]
        mcp_port: Option<u16>,
        /// Optional port for a sidecar A2A agent server (serves an `LLMChain`).
        #[arg(long)]
        a2a_port: Option<u16>,
    },
    /// Evaluate a JSONL dataset and write a JSON report.
    Eval {
        /// Dataset path: JSONL with {"input","expected"[,"actual"]} per line.
        #[arg(long)]
        dataset: PathBuf,
        /// Output report path.
        #[arg(long)]
        output: PathBuf,
        /// Scoring metric: exact | distance.
        #[arg(long, default_value = "exact")]
        scorer: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Init {
            path,
            template,
            provider,
        } => init::run(&path, &template, &provider),
        Command::Run { args } => run::run(&args),
        Command::Serve {
            port,
            model,
            backend,
            auth,
            mcp_port,
            a2a_port,
        } => serve::run(
            port,
            model.as_deref(),
            &backend,
            auth.as_deref(),
            mcp_port,
            a2a_port,
        )
        .await,
        Command::Eval {
            dataset,
            output,
            scorer,
        } => eval::run(&dataset, &output, &scorer).await,
    }
}