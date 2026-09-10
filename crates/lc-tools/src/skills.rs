// lc-tools/src/skills.rs
//! Agent Skills (SKILL.md) loader with progressive disclosure (D1, v0.22.1).
//!
//! Parses the ecosystem-standard Agent Skills skill bundle — a directory whose entry point is a
//! `SKILL.md` file with YAML frontmatter. Exposes two views:
//! - [`Skill::disclosure_view`]: name + one-line description only, safe to show to the model up
//!   front (low token cost, no body leakage).
//! - [`Skill::full_text`]: the complete skill body, loaded only once the model actually selects
//!   the skill (progressive disclosure).
//!
//! Execution sandboxing is intentionally NOT implemented: a skill's auxiliary scripts/templates
//! are surfaced as metadata, leaving execution to the caller (and out of scope here).

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Error while loading or parsing a skill bundle.
#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    /// The directory does not contain a `SKILL.md` entry point.
    #[error("skill directory missing SKILL.md: {0}")]
    MissingSkillFile(String),
    /// The frontmatter is malformed.
    #[error("invalid SKILL.md frontmatter: {0}")]
    Frontmatter(String),
    /// Required frontmatter fields are absent.
    #[error("SKILL.md missing required field: {0}")]
    MissingField(String),
    /// I/O failure reading the skill directory.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Required head matter of a SKILL.md file.
///
/// `name` and `description` are surfaced as `Option` so that a missing required field is
/// reported via [`SkillError::MissingField`] (with a field-specific message) rather than a
/// generic YAML deserialization error.
#[derive(Debug, Clone, Deserialize)]
pub struct SkillFrontmatter {
    /// Skill name (unique identifier).
    pub name: Option<String>,
    /// One-line description shown for progressive disclosure.
    pub description: Option<String>,
    /// Optional files/scripts that belong to the skill bundle.
    #[serde(default)]
    #[allow(dead_code)]
    pub allowed_tools: Option<Vec<String>>,
    /// Optional extra vendor-defined fields are preserved as opaque metadata.
    #[serde(flatten)]
    #[allow(dead_code)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// A parsed, loadable skill unit.
#[derive(Debug, Clone)]
pub struct Skill {
    frontmatter: SkillFrontmatter,
    /// Full markdown body (frontmatter stripped).
    body: String,
    /// Directory containing the bundle (for resolving auxiliary files).
    dir: PathBuf,
}

impl Skill {
    /// Load a skill bundle from a directory containing `SKILL.md`.
    pub fn load(dir: impl Into<PathBuf>) -> Result<Self, SkillError> {
        let dir = dir.into();
        let skill_path = dir.join("SKILL.md");
        if !skill_path.is_file() {
            return Err(SkillError::MissingSkillFile(skill_path.display().to_string()));
        }
        let raw = fs::read_to_string(&skill_path)?;
        let (frontmatter, body) = parse_frontmatter(&raw, &skill_path)?;
        Ok(Self {
            frontmatter,
            body,
            dir,
        })
    }

    /// Scan a root directory for all skill bundle directories (each containing `SKILL.md`).
    pub fn scan(root: impl AsRef<Path>) -> Result<Vec<Self>, SkillError> {
        let mut skills = Vec::new();
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() && path.join("SKILL.md").is_file() {
                skills.push(Self::load(&path)?);
            }
        }
        Ok(skills)
    }

    /// Skill name (unique identifier).
    ///
    /// Safe because [`Skill::load`] rejects a missing/empty `name`; [`Skill`] can therefore
    /// never be constructed with `None` here.
    pub fn name(&self) -> &str {
        self.frontmatter.name.as_deref().unwrap_or_default()
    }

    /// One-line description, safe for up-front listing.
    ///
    /// Safe for the same invariant as [`Skill::name`].
    pub fn description(&self) -> &str {
        self.frontmatter.description.as_deref().unwrap_or_default()
    }

    /// Progressive disclosure: name + description only. Low token cost, no body leakage.
    ///
    /// This is what the agent should see in its system prompt *before* selecting a skill.
    pub fn disclosure_view(&self) -> String {
        format!("{}: {}", self.name(), self.description())
    }

    /// Full skill body, loaded only after the model selects this skill.
    pub fn full_text(&self) -> String {
        format!(
            "# Skill: {}\n\n## Description\n{}\n\n## Instructions\n\n{}",
            self.name(),
            self.description(),
            self.body
        )
    }

    /// The bundle directory (for resolving auxiliary files/scripts).
    pub fn directory(&self) -> &Path {
        &self.dir
    }
}

/// Splits a SKILL.md file into (frontmatter, body). Frontmatter is delimited by `---` lines.
fn parse_frontmatter(raw: &str, path: &Path) -> Result<(SkillFrontmatter, String), SkillError> {
    let stripped = raw.strip_prefix('\u{feff}').unwrap_or(raw); // tolerate a leading BOM
    if !stripped.trim_start().starts_with("---") {
        return Err(SkillError::MissingField(
            "frontmatter block (---...) missing".into(),
        ));
    }
    // Find the closing `---`.
    let after_open = stripped.find('\n').ok_or_else(|| {
        SkillError::Frontmatter(format!("{}: no newline after opening ---", path.display()))
    })?;
    let rest = &stripped[after_open..];
    let close = rest.find("\n---").ok_or_else(|| {
        SkillError::Frontmatter(format!("{}: no closing --- for frontmatter", path.display()))
    })?;
    let yaml = &rest[..close];
    let body = &rest[close + 4..];

    let frontmatter: SkillFrontmatter = serde_yaml::from_str(yaml)
        .map_err(|e| SkillError::Frontmatter(format!("{}: {}", path.display(), e)))?;
    match &frontmatter.name {
        Some(n) if !n.trim().is_empty() => {}
        _ => return Err(SkillError::MissingField("name".into())),
    }
    match &frontmatter.description {
        Some(d) if !d.trim().is_empty() => {}
        _ => return Err(SkillError::MissingField("description".into())),
    }
    Ok((frontmatter, body.trim_matches('\n').to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_skill(dir: &Path, name: &str, description: &str, body: &str) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let frontmatter = format!("---\nname: {name}\ndescription: {description}\n---\n\n");
        let p = dir.join("SKILL.md");
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(format!("{frontmatter}{body}").as_bytes()).unwrap();
        p
    }

    #[test]
    fn parses_frontmatter_and_body() {
        let dir = tempfile::tempdir().unwrap();
        write_skill(dir.path(), "web_search", "Searches the web", "Do a search then summarize.");
        let skill = Skill::load(dir.path()).unwrap();
        assert_eq!(skill.name(), "web_search");
        assert_eq!(skill.description(), "Searches the web");
        assert_eq!(skill.full_text().contains("Do a search then summarize."), true);
    }

    #[test]
    fn disclosure_view_excludes_body() {
        let dir = tempfile::tempdir().unwrap();
        write_skill(dir.path(), "secret", "Just a name", "<!-- SECRET BODY -->");
        let skill = Skill::load(dir.path()).unwrap();
        assert_eq!(skill.disclosure_view(), "secret: Just a name");
        assert_eq!(skill.disclosure_view().contains("SECRET BODY"), false);
        assert_eq!(skill.full_text().contains("SECRET BODY"), true);
    }

    #[test]
    fn missing_skill_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let err = Skill::load(dir.path()).unwrap_err();
        assert!(matches!(err, SkillError::MissingSkillFile(_)));
    }

    #[test]
    fn missing_required_field_errors() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("SKILL.md");
        std::fs::write(&p, "---\ndescription: no name here\n---\n\nbody").unwrap();
        let err = Skill::load(dir.path()).unwrap_err();
        assert!(matches!(err, SkillError::MissingField(_)));
    }

    #[test]
    fn scan_finds_only_skill_dirs() {
        let root = tempfile::tempdir().unwrap();
        write_skill(&root.path().join("a"), "a", "skill A", "body a");
        write_skill(&root.path().join("b"), "b", "skill B", "body b");
        std::fs::create_dir(root.path().join("plain")).unwrap();
        let skills = Skill::scan(root.path()).unwrap();
        assert_eq!(skills.len(), 2);
    }
}