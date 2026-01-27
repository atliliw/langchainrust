/// Abstraction over a memory store used to inject conversation history.
pub trait Memory :Send + Sync{
    fn add(&mut self, input: &str, output: &str);
    fn context(&self) -> String;
    fn clear(&mut self);
    fn history(&self) -> Vec<&str>;
}

/// Simple in-memory implementation that keeps a rolling window of messages.
pub struct SimpleMemory {
    history: Vec<String>,
    max_turns: usize,
}

impl SimpleMemory {
    /// Create a memory with a given maximum number of turns.
    pub fn new(max_turns: usize) -> Self {
        Self {
            history: Vec::new(),
            max_turns,
        }
    }
}

impl Default for SimpleMemory {
    fn default() -> Self {
        Self::new(5)
    }
}

impl Memory for SimpleMemory {
    fn add(&mut self, _input: &str, output: &str) {
        self.history.push(output.to_string()); // ← 存 output
        if self.history.len() > self.max_turns {
            self.history.remove(0);
        }
    }

    fn context(&self) -> String {
        self.history
            .iter()
            .enumerate()
            .map(|(i, q)| format!("{}. {}", i + 1, q))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn clear(&mut self) {
        self.history.clear();
    }

    fn history(&self) -> Vec<&str> {
        self.history.iter().map(|s| s.as_str()).collect()
    }
}
