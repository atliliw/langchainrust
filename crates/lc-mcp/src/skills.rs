//! MCP Skills extension (SEP-2640): a server ships human-readable capability
//! descriptions ("skills") alongside its tools, and clients discover them via
//! `skills/list` / `skills/get`.
//!
//! Design note (per the SEP): skills are exposed like an indexed directory of
//! documents — `skills/list` returns the manifest (name + description + version),
//! `skills/get` returns the full markdown body. Content is opaque text; execution
//! stays with the caller.

use serde::Deserialize;
use serde_json::{json, Value};

/// One skill a server can ship.
#[derive(Debug, Clone)]
pub struct McpSkill {
    /// Unique skill name (matches the `name` frontmatter in a SKILL.md).
    pub name: String,
    /// One-line description surfaced by `skills/list`.
    pub description: String,
    /// Skill version.
    pub version: String,
    /// Full markdown body, returned only by `skills/get`.
    pub content: String,
}

/// Client-side manifest entry from `skills/list` (no body).
#[derive(Debug, Clone, Deserialize)]
pub struct SkillDescriptor {
    /// Skill name.
    pub name: String,
    /// One-line description.
    pub description: String,
    /// Skill version.
    pub version: String,
}

/// Client-side payload from `skills/get`.
#[derive(Debug, Clone, Deserialize)]
pub struct SkillBody {
    /// Skill name.
    pub name: String,
    /// Skill version.
    pub version: String,
    /// Full markdown body.
    pub content: String,
}

impl McpSkill {
    /// Build a skill with version `"1"`.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            version: "1".to_string(),
            content: content.into(),
        }
    }

    /// Override the version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Manifest entry shown by `skills/list` (no body).
    pub(crate) fn manifest(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "version": self.version,
        })
    }
}

/// Ordered registry of skills exposed by a server.
#[derive(Debug, Default)]
pub(crate) struct SkillsDirectory {
    skills: Vec<McpSkill>,
}

impl SkillsDirectory {
    /// Register a skill; a later duplicate name replaces the earlier one.
    pub(crate) fn add(&mut self, skill: McpSkill) {
        self.skills.retain(|s| s.name != skill.name);
        self.skills.push(skill);
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub(crate) fn list(&self) -> Value {
        json!({ "skills": self.skills.iter().map(|s| s.manifest()).collect::<Vec<_>>() })
    }

    pub(crate) fn get(&self, name: &str) -> Option<Value> {
        self.skills
            .iter()
            .find(|s| s.name == name)
            .map(|s| json!({ "name": s.name, "version": s.version, "content": s.content }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directory_dedupes_by_name() {
        let mut dir = SkillsDirectory::default();
        dir.add(McpSkill::new("a", "desc a", "# A"));
        dir.add(McpSkill::new("a", "desc a v2", "# A v2"));
        let list = dir.list();
        let skills = list["skills"].as_array().unwrap();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0]["version"], json!("1"));
    }

    #[test]
    fn manifest_excludes_body() {
        let skill = McpSkill::new("write", "write prose", "long body here");
        let m = skill.manifest();
        assert!(m.get("content").is_none(), "manifest must not leak the body");
        assert_eq!(m["name"], json!("write"));
    }

    #[test]
    fn get_returns_body_and_unknown_is_none() {
        let mut dir = SkillsDirectory::default();
        dir.add(McpSkill::new("x", "d", "body of x"));
        let got = dir.get("x").unwrap();
        assert_eq!(got["content"], json!("body of x"));
        assert!(dir.get("missing").is_none());
    }
}