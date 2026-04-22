// src/callbacks/langsmith_client.rs
//! LangSmith API client

use reqwest::Client;
use std::env;

use super::run_tree::{RunCreate, RunTree, RunUpdate};

/// LangSmith configuration
#[derive(Debug, Clone)]
pub struct LangSmithConfig {
    /// API key (starts with "ls_")
    pub api_key: String,
    
    /// API endpoint URL
    pub api_url: String,
    
    /// Workspace ID (required for org accounts)
    pub workspace_id: Option<String>,
    
    /// Project name
    pub project_name: String,
    
    /// Whether tracing is enabled
    pub tracing_enabled: bool,
}

impl LangSmithConfig {
    /// Create config from environment variables
    pub fn from_env() -> Result<Self, String> {
        let api_key = env::var("LANGSMITH_API_KEY")
            .map_err(|_| "LANGSMITH_API_KEY environment variable not set")?;
        
        let tracing_enabled = env::var("LANGSMITH_TRACING")
            .map(|v| v == "true")
            .unwrap_or(true);
        
        let project_name = env::var("LANGSMITH_PROJECT")
            .unwrap_or_else(|_| "default".to_string());
        
        let api_url = env::var("LANGSMITH_ENDPOINT")
            .unwrap_or_else(|_| "https://api.smith.langchain.com".to_string());
        
        let workspace_id = env::var("LANGSMITH_WORKSPACE_ID").ok();
        
        Ok(Self {
            api_key,
            api_url,
            workspace_id,
            project_name,
            tracing_enabled,
        })
    }
    
    /// Create config with API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            api_url: "https://api.smith.langchain.com".to_string(),
            workspace_id: None,
            project_name: "default".to_string(),
            tracing_enabled: true,
        }
    }
    
    /// Set project name
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project_name = project.into();
        self
    }
    
    /// Set workspace ID
    pub fn with_workspace(mut self, workspace_id: impl Into<String>) -> Self {
        self.workspace_id = Some(workspace_id.into());
        self
    }
    
    /// Set API endpoint
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.api_url = endpoint.into();
        self
    }
    
    /// Enable or disable tracing
    pub fn with_tracing(mut self, enabled: bool) -> Self {
        self.tracing_enabled = enabled;
        self
    }
}

/// LangSmith API client
pub struct LangSmithClient {
    /// Configuration
    pub config: LangSmithConfig,
    http_client: Client,
}

impl LangSmithClient {
    /// Create a new client
    pub fn new(config: LangSmithConfig) -> Self {
        Self {
            config,
            http_client: Client::new(),
        }
    }
    
    /// Create client from environment variables
    pub fn from_env() -> Result<Self, String> {
        let config = LangSmithConfig::from_env()?;
        Ok(Self::new(config))
    }
    
    /// Check if tracing is enabled
    pub fn is_tracing_enabled(&self) -> bool {
        self.config.tracing_enabled
    }
    
    /// Get the project name
    pub fn project_name(&self) -> &str {
        &self.config.project_name
    }
    
    /// Create a run (POST /runs)
    pub async fn create_run(&self, run: &RunTree) -> Result<(), LangSmithError> {
        if !self.config.tracing_enabled {
            return Ok(());
        }
        
        let url = format!("{}/runs", self.config.api_url);
        let mut run_create = RunCreate::from(run);
        if run_create.session_name.is_none() {
            run_create.session_name = Some(self.config.project_name.clone());
        }
        
        let mut request = self.http_client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .json(&run_create);
        
        if let Some(workspace_id) = &self.config.workspace_id {
            request = request.header("x-tenant-id", workspace_id);
        }
        
        let response = request
            .send()
            .await
            .map_err(|e| LangSmithError::Http(e.to_string()))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LangSmithError::Api(format!("HTTP {}: {}", status, body)));
        }
        
        Ok(())
    }
    
    /// Update a run (PATCH /runs/{run_id})
    pub async fn update_run(&self, run: &RunTree) -> Result<(), LangSmithError> {
        if !self.config.tracing_enabled {
            return Ok(());
        }
        
        let url = format!("{}/runs/{}", self.config.api_url, run.id);
        let body = RunUpdate::from(run);
        
        let mut request = self.http_client
            .patch(&url)
            .header("x-api-key", &self.config.api_key)
            .json(&body);
        
        if let Some(workspace_id) = &self.config.workspace_id {
            request = request.header("x-tenant-id", workspace_id);
        }
        
        let response = request
            .send()
            .await
            .map_err(|e| LangSmithError::Http(e.to_string()))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LangSmithError::Api(format!("HTTP {}: {}", status, body)));
        }
        
        Ok(())
    }
    
    /// Batch ingest runs (POST /runs/multipart) for high throughput
    pub async fn batch_ingest(&self, runs: &[RunTree]) -> Result<(), LangSmithError> {
        if !self.config.tracing_enabled || runs.is_empty() {
            return Ok(());
        }
        
        let url = format!("{}/runs/multipart", self.config.api_url);
        
        let runs_create: Vec<RunCreate> = runs
            .iter()
            .map(|run| {
                let mut run_create = RunCreate::from(run);
                if run_create.session_name.is_none() {
                    run_create.session_name = Some(self.config.project_name.clone());
                }
                run_create
            })
            .collect();
        
        let runs_json = serde_json::to_string(&runs_create)
            .map_err(|e| LangSmithError::Api(e.to_string()))?;
        
        let form = reqwest::multipart::Form::new()
            .text("runs", runs_json);
        
        let mut request = self.http_client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .multipart(form);
        
        if let Some(workspace_id) = &self.config.workspace_id {
            request = request.header("x-tenant-id", workspace_id);
        }
        
        let response = request
            .send()
            .await
            .map_err(|e| LangSmithError::Http(e.to_string()))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LangSmithError::Api(format!("HTTP {}: {}", status, body)));
        }
        
        Ok(())
    }
    
    /// Batch ingest runs using parallel requests for maximum throughput
    pub async fn batch_ingest_parallel(&self, runs: &[RunTree]) -> Result<(), LangSmithError> {
        if !self.config.tracing_enabled || runs.is_empty() {
            return Ok(());
        }
        
        use futures_util::future::join_all;
        
        let futures: Vec<_> = runs
            .iter()
            .map(|run| self.create_run(run))
            .collect();
        
        let results = join_all(futures).await;
        
        for result in results {
            result?;
        }
        
        Ok(())
    }
}

/// LangSmith error type
#[derive(Debug)]
pub enum LangSmithError {
    Http(String),
    Api(String),
    Config(String),
}

impl std::fmt::Display for LangSmithError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Http(msg) => write!(f, "HTTP error: {}", msg),
            Self::Api(msg) => write!(f, "API error: {}", msg),
            Self::Config(msg) => write!(f, "Config error: {}", msg),
        }
    }
}

impl std::error::Error for LangSmithError {}