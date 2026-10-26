// crates/lc/src/agents/tests.rs
//! Facade-level smoke tests: lock the "default main path" — a new user builds
//! an agent from the single `langchainrust` crate via the re-exported
//! [`AgentBuilder`](crate::AgentBuilder) with the documented 3-line fluent form.
//!
//! These sit at the facade (`crates/lc`) rather than inside `lc-agents` because
//! they verify the *canonical entry point*: that every piece the 3-line default
//! path needs (builder, an LLM, a tool) is reachable from the unified crate root,
//! not merely from inside the domain crate. If this test breaks, "how do I make
//! an agent?" loses its one obvious answer.

use crate::AgentBuilder;
use lc_tools::Calculator;

#[test]
fn facade_default_main_path_builds_agent() {
    // The documented 3-line main path, straight from the langchainrust crate root.
    let agent = AgentBuilder::new()
        .llm(lc_providers::OpenAIChat::new(
            lc_providers::OpenAIConfig::new("test_key"),
        ))
        .system("You are a concise assistant.")
        .tool(Calculator::new())
        .build()
        .expect("default main path must build a FunctionCallingAgent");

    // The agent is fully wired: it knows its one tool and its system prompt.
    assert_eq!(agent.tools_count(), 1, "the main-path agent should have its one tool");
    assert_eq!(
        agent.system_prompt(),
        Some("You are a concise assistant."),
        "system prompt should be carried through the main path"
    );
    // It must also be usable as an opaque `BaseAgent` — the main path is not a
    // dead end; the built agent slots straight into an executor.
    use crate::agents::BaseAgent;
    assert_eq!(BaseAgent::max_iterations(&agent), Some(10), "default max_iterations=10");
}

#[test]
fn facade_default_main_path_requires_llm() {
    // Omitting `.llm()` must fail loudly, not build a half-wired agent.
    let err = AgentBuilder::new()
        .system("no model")
        .tool(Calculator::new())
        .build()
        .expect_err("main path without an LLM must not build");
    assert!(err.to_string().contains("LLM is required"));
}