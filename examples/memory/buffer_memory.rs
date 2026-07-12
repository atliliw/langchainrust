//! Buffer 记忆示例
//!
//! 展示 ConversationBufferMemory 保存多轮对话历史(无需 API Key)。
//!
//! # 运行
//! ```bash
//! cargo run --example memory_buffer_memory
//! ```

use langchainrust::memory::BaseMemory;
use langchainrust::ConversationBufferMemory;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut memory = ConversationBufferMemory::new();

    for i in 1..=3 {
        let inputs = HashMap::from([("input".to_string(), format!("问题{}", i))]);
        let outputs = HashMap::from([("output".to_string(), format!("答案{}", i))]);
        memory.save_context(&inputs, &outputs).await?;
        println!("第 {} 轮已保存,共 {} 条消息", i, memory.chat_memory().len());
    }

    let loaded = memory.load_memory_variables(&HashMap::new()).await?;
    println!("\n加载的历史:\n{}", loaded.get("history").unwrap());
    Ok(())
}
