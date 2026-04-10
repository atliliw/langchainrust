// src/callbacks/handlers/stdout_handler.rs
//! Standard output callback handler for debugging

use async_trait::async_trait;
use std::io::Write;

use crate::callbacks::{CallbackHandler, RunTree};
use crate::schema::Message;

/// Standard output callback handler
/// 
/// Prints trace information to console for debugging.
pub struct StdOutHandler {
    verbose: bool,
}

impl StdOutHandler {
    pub fn new() -> Self {
        Self { verbose: true }
    }
    
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl Default for StdOutHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CallbackHandler for StdOutHandler {
    async fn on_run_start(&self, run: &RunTree) {
        if self.verbose {
            println!("\n{} [{}] START: {}", 
                run.run_type.emoji(),
                run.run_type.as_str().to_uppercase(), 
                run.name
            );
            println!("   ID: {}", run.id);
            if run.inputs != serde_json::Value::Null {
                println!("   Inputs: {}", 
                    serde_json::to_string_pretty(&run.inputs).unwrap_or_default()
                );
            }
        }
    }
    
    async fn on_run_end(&self, run: &RunTree) {
        if self.verbose {
            let duration = run.duration_ms()
                .map(|d| format!("{}ms", d))
                .unwrap_or_default();
            
            println!("\n{} [{}] END: {} ({})", 
                run.run_type.emoji(),
                run.run_type.as_str().to_uppercase(), 
                run.name,
                duration
            );
            
            if let Some(outputs) = &run.outputs {
                println!("   Outputs: {}", 
                    serde_json::to_string_pretty(outputs).unwrap_or_default()
                );
            }
        }
    }
    
    async fn on_run_error(&self, run: &RunTree, error: &str) {
        println!("\n❌ [{}] ERROR: {}", run.run_type.as_str().to_uppercase(), run.name);
        println!("   Error: {}", error);
    }
    
    async fn on_llm_start(&self, run: &RunTree, messages: &[Message]) {
        if self.verbose {
            self.on_run_start(run).await;
            println!("   Messages: {} message(s)", messages.len());
        }
    }
    
    async fn on_llm_new_token(&self, _run: &RunTree, token: &str) {
        print!("{}", token);
        let _ = std::io::stdout().flush();
    }
    
    async fn on_tool_start(&self, run: &RunTree, tool_name: &str, input: &str) {
        if self.verbose {
            self.on_run_start(run).await;
            println!("   Tool: {}", tool_name);
            println!("   Input: {}", input);
        }
    }
    
    async fn on_retriever_start(&self, run: &RunTree, query: &str) {
        if self.verbose {
            self.on_run_start(run).await;
            println!("   Query: {}", query);
        }
    }
    
    async fn on_retriever_end(&self, run: &RunTree, documents: &[serde_json::Value]) {
        if self.verbose {
            self.on_run_end(run).await;
            println!("   Documents: {} result(s)", documents.len());
        }
    }
}