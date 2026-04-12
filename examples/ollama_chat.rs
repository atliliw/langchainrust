// examples/ollama_chat.rs

use langchainrust::{OllamaChat, BaseChatModel};
use langchainrust::schema::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = OllamaChat::new("qwen2.5:7b");
    
    let messages = vec![
        Message::system("你是一个友好的助手"),
        Message::human("你好，介绍一下你自己"),
    ];
    
    println!("Sending request to Ollama...");
    
    let response = llm.chat(messages, None).await?;
    
    println!("Response from {}: ", response.model);
    println!("{}", response.content);
    
    if let Some(usage) = response.token_usage {
        println!("\nToken usage:");
        println!("  Prompt: {}", usage.prompt_tokens);
        println!("  Completion: {}", usage.completion_tokens);
        println!("  Total: {}", usage.total_tokens);
    }
    
    Ok(())
}