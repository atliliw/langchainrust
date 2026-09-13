//! RAGAS-style offline RAG evaluation (B9, v0.22.4)
//!
//! Scores a RAG system against a labelled dataset with retrieved contexts:
//!
//! - `ContextPrecision` — for every retrieved chunk the judge says whether it is relevant;
//!   the score is mean precision@k over the relevant ranks (chunks in retrieval order), so a
//!   relevant chunk buried below irrelevant ones costs points.
//! - `ContextRecall` — the reference answer is split into atomic claims; the score is the
//!   fraction of claims attributable to the retrieved contexts (missing knowledge lowers it).
//! - `AnswerRelevancy` — the model generates N questions the answer could address and the
//!   score is the mean cosine similarity between those questions' embeddings and the
//!   original question's embedding (a fluent non-answer still scores ~0).
//!
//! Every report row carries the pinned `run_id`, and the report is exported as JSONL so the
//! batch scores join back to the traces the system under test emitted.
//!
//! # Run
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p langchainrust --example ragas_eval
//! # optional: OPENAI_BASE_URL, EVAL_JUDGE_MODEL, EVAL_EMBED_MODEL,
//! #           EVAL_RUN_ID, EVAL_EXPORT=rag.jsonl
//! ```

use async_trait::async_trait;
use langchainrust::evaluation::*;
use langchainrust::schema::Message;
use langchainrust::{
    BaseChatModel, OpenAIChat, OpenAIConfig, OpenAIEmbeddings, OpenAIEmbeddingsConfig,
};

/// Predictor wrapping the chat model: in a real evaluation this is the whole RAG pipeline
/// (retrieval + generation); here the contexts come straight from the labelled dataset.
struct ModelPredictor {
    model: OpenAIChat,
}

#[async_trait]
impl Predictor for ModelPredictor {
    async fn predict(&self, input: &str) -> Result<String, EvalError> {
        let system = "You answer the user's question concisely. Use only known facts.";
        let reply = self
            .model
            .chat_with_system(system.to_string(), vec![Message::human(input)])
            .await
            .map_err(|e| EvalError::PredictorError(e.to_string()))?;
        Ok(reply.content)
    }

    async fn begin_run(&self, run_id: &str) {
        // A real system under test stamps the eval run id into its RunnableConfig metadata
        // ("trace_id"); the agent executor then propagates it onto every callback/OTel span,
        // closing the eval-row ↔ trace join. This stateless demo only logs it.
        println!("starting evaluation run {run_id}");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| "please set the OPENAI_API_KEY environment variable")?;
    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".into());
    let judge_model = std::env::var("EVAL_JUDGE_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
    let embed_name =
        std::env::var("EVAL_EMBED_MODEL").unwrap_or_else(|_| "text-embedding-ada-002".into());

    let chat_config = OpenAIConfig {
        api_key: api_key.clone(),
        base_url: base_url.clone(),
        model: judge_model,
        ..Default::default()
    };
    // The answerer and the three judges each own a cheap client (reqwest reuses connections).
    let answerer = OpenAIChat::new(chat_config.clone());
    let judge1 = OpenAIChat::new(chat_config.clone());
    let judge2 = OpenAIChat::new(chat_config.clone());
    let judge3 = OpenAIChat::new(chat_config);
    let embeddings = OpenAIEmbeddings::new(OpenAIEmbeddingsConfig {
        api_key,
        base_url,
        model: embed_name,
        ..Default::default()
    })?;

    // Labelled dataset: input, ground-truth answer, retrieved chunks in rank order.
    let dataset = Dataset::new(vec![Example::with_contexts(
        "Which planet is closest to the Sun?",
        "Mercury",
        vec![
            "Mercury is the smallest planet in the Solar System and the closest to the Sun.".into(),
            "Venus is the second planet from the Sun, between Mercury and Earth.".into(),
        ],
    )]);

    let run_id = std::env::var("EVAL_RUN_ID").unwrap_or_else(|_| "ragas-demo".to_string());
    let runner = EvalRunner::new(vec![])
        .with_rag_evaluators(vec![
            Box::new(ContextPrecision::new(judge1)),
            Box::new(ContextRecall::new(judge2)),
            Box::new(AnswerRelevancy::new(judge3, embeddings)),
        ])
        .with_run_id(run_id);

    let report = runner
        .run(&dataset, &ModelPredictor { model: answerer })
        .await?;

    println!("{}", report.to_table());
    match std::env::var("EVAL_EXPORT") {
        Ok(path) => {
            report.write_jsonl(&path).await?;
            println!("wrote JSONL export to {path}");
        }
        Err(_) => print!("{}", report.to_jsonl()),
    }
    Ok(())
}
