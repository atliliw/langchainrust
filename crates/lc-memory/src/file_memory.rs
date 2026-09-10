// lc-memory/src/file_memory.rs
//! File-based memory (C1, v0.22.1 §S4): memories are real, inspectable files.
//!
//! Many agent memory systems model "memory" as an opaque serialized blob. C1 takes the
//! opposite approach: each memory is a plain `NAME.md` file under a single root
//! directory. This gives the model (and a human auditor) a first-class editing surface —
//! view, create, append, str_replace, rename, delete — mirroring the model editing tools
//! that frontier agents expose, but as a deterministic library primitive, not a
//! tool-calling loop.
//!
//! # Path safety
//!
//! Because the memory name arrives from the model, every operation funnels through
//! `validated_name`, which rejects traversal (`..`), absolute paths, separators
//! (`/`, `\`), drive letters (`:`), leading/trailing whitespace, and Windows reserved
//! device names. The store root is canonicalized at construction, so no name can
//! escape it. The `.md` suffix is appended by the store, never by the caller.
//!
//! Decay / forgetting (TTL + importance) is layered on top in [`super::decay`].

use std::fs;
use std::path::{Path, PathBuf};

/// Windows reserved device names that cannot be used as file names (case-insensitive,
/// with or without an extension).
const WINDOWS_RESERVED: [&str; 22] = [
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// A memory file on disk, ready for inspection.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// Memory name (file stem, no `.md`).
    pub name: String,
    /// Last modification time of the file.
    pub modified: std::time::SystemTime,
}

/// Errors from [`FileMemoryStore`] operations.
#[derive(Debug, thiserror::Error)]
pub enum FileMemoryError {
    /// The memory name is unsafe (traversal, absolute path, separator, reserved name...).
    #[error("unsafe memory name `{name}`: {reason}")]
    UnsafeName {
        /// The offending name.
        name: String,
        /// Why it was rejected.
        reason: String,
    },
    /// A memory with that name already exists.
    #[error("memory `{0}` already exists")]
    AlreadyExists(String),
    /// No memory with that name exists.
    #[error("memory `{0}` not found")]
    NotFound(String),
    /// `str_replace` could not find the old text.
    #[error("old text not found in memory `{0}`")]
    OldTextNotFound(String),
    /// Filesystem I/O failure.
    #[error("I/O error while accessing `{path}`: {source}")]
    Io {
        /// The path involved.
        path: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

impl From<std::io::Error> for FileMemoryError {
    fn from(source: std::io::Error) -> Self {
        FileMemoryError::Io {
            path: "<unknown>".to_string(),
            source,
        }
    }
}

/// Validates a caller-supplied memory name, forbidding anything that could escape the
/// store root. Returns the validated bare name.
fn validated_name(name: &str) -> Result<String, FileMemoryError> {
    if name.is_empty() {
        return Err(unsafe_name(name, "name is empty"));
    }
    if name == "." || name == ".." {
        return Err(unsafe_name(name, "path traversal"));
    }
    if name.contains('/') || name.contains('\\') {
        return Err(unsafe_name(name, "path separators are not allowed"));
    }
    if name.contains(':') {
        return Err(unsafe_name(
            name,
            "drive/stream designators (`:`) are not allowed",
        ));
    }
    if name.trim() != name {
        return Err(unsafe_name(
            name,
            "leading/trailing whitespace is not allowed",
        ));
    }
    if name.contains('\0') {
        return Err(unsafe_name(name, "NUL byte is not allowed"));
    }
    if name.len() > 255 {
        return Err(unsafe_name(name, "name too long"));
    }
    let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
    if WINDOWS_RESERVED.contains(&stem.as_str()) {
        return Err(unsafe_name(name, "Windows reserved device name"));
    }
    Ok(name.to_string())
}

/// Builds an [`FileMemoryError::UnsafeName`] for a rejected name.
fn unsafe_name(name: &str, reason: &str) -> FileMemoryError {
    FileMemoryError::UnsafeName {
        name: name.to_string(),
        reason: reason.to_string(),
    }
}

/// A deterministic, path-sandboxed collection of memory files under one root directory.
pub struct FileMemoryStore {
    root: PathBuf,
}

impl std::fmt::Debug for FileMemoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileMemoryStore")
            .field("root", &self.root)
            .finish()
    }
}

impl FileMemoryStore {
    /// Opens (creating if needed) a memory root and canonicalizes it so every later
    /// path check is against a single absolute base.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, FileMemoryError> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        let root = root.canonicalize().map_err(|source| FileMemoryError::Io {
            path: root.display().to_string(),
            source,
        })?;
        Ok(Self { root })
    }

    /// The canonicalized root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolves a validated name to its `.md` path inside the root.
    fn path_for(&self, name: &str) -> Result<PathBuf, FileMemoryError> {
        let name = validated_name(name)?;
        Ok(self.root.join(format!("{name}.md")))
    }

    /// Creates a new memory. Fails if a memory with the same name already exists.
    pub fn create(&self, name: &str, content: &str) -> Result<(), FileMemoryError> {
        let path = self.path_for(name)?;
        if path.exists() {
            return Err(FileMemoryError::AlreadyExists(name.to_string()));
        }
        fs::write(&path, content).map_err(map_io(&path))?;
        Ok(())
    }

    /// Returns the full content of a memory.
    pub fn view(&self, name: &str) -> Result<String, FileMemoryError> {
        let path = self.path_for(name)?;
        if !path.exists() {
            return Err(FileMemoryError::NotFound(name.to_string()));
        }
        fs::read_to_string(&path).map_err(map_io(&path))
    }

    /// Overwrites a memory's content whether or not it exists.
    ///
    /// Unlike [`Self::create`] this never fails on an existing name; it is the
    /// upsert primitive used by higher-level stacks (e.g. re-remembering).
    pub fn write(&self, name: &str, content: &str) -> Result<(), FileMemoryError> {
        let path = self.path_for(name)?;
        fs::write(&path, content).map_err(map_io(&path))?;
        Ok(())
    }

    /// Appends `content` to an existing memory.
    pub fn append(&self, name: &str, content: &str) -> Result<(), FileMemoryError> {
        let path = self.path_for(name)?;
        if !path.exists() {
            return Err(FileMemoryError::NotFound(name.to_string()));
        }
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .map_err(map_io(&path))?;
        std::io::Write::write_all(&mut file, content.as_bytes()).map_err(map_io(&path))?;
        Ok(())
    }

    /// Replaces the first exact occurrence of `old` with `new` in a memory.
    ///
    /// Fails with [`FileMemoryError::OldTextNotFound`] if `old` is not present, rather
    /// than silently corrupting the file — str_replace must be idempotent and observable.
    pub fn str_replace(&self, name: &str, old: &str, new: &str) -> Result<(), FileMemoryError> {
        let path = self.path_for(name)?;
        if !path.exists() {
            return Err(FileMemoryError::NotFound(name.to_string()));
        }
        let content = fs::read_to_string(&path).map_err(map_io(&path))?;
        if !content.contains(old) {
            return Err(FileMemoryError::OldTextNotFound(name.to_string()));
        }
        // one-shot replace (no `replace_all`): the caller observes exactly one edit.
        let replaced = content.replacen(old, new, 1);
        fs::write(&path, replaced).map_err(map_io(&path))?;
        Ok(())
    }

    /// Renames a memory to a new validated name.
    pub fn rename(&self, name: &str, new_name: &str) -> Result<(), FileMemoryError> {
        let from = self.path_for(name)?;
        let to = self.path_for(new_name)?;
        if !from.exists() {
            return Err(FileMemoryError::NotFound(name.to_string()));
        }
        if to.exists() {
            return Err(FileMemoryError::AlreadyExists(new_name.to_string()));
        }
        fs::rename(&from, &to).map_err(map_io(&from))?;
        Ok(())
    }

    /// Deletes a memory.
    pub fn delete(&self, name: &str) -> Result<(), FileMemoryError> {
        let path = self.path_for(name)?;
        if !path.exists() {
            return Err(FileMemoryError::NotFound(name.to_string()));
        }
        fs::remove_file(&path).map_err(map_io(&path))?;
        Ok(())
    }

    /// Lists all memories, newest-modified first.
    pub fn list(&self) -> Result<Vec<MemoryEntry>, FileMemoryError> {
        let mut entries = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
            let modified = entry
                .metadata()?
                .modified()
                .unwrap_or(std::time::UNIX_EPOCH);
            entries.push(MemoryEntry { name, modified });
        }
        entries.sort_by(|a, b| b.modified.cmp(&a.modified));
        Ok(entries)
    }
}

/// Lifts an I/O error into `FileMemoryError::Io` carrying the path context.
fn map_io(path: &Path) -> impl FnOnce(std::io::Error) -> FileMemoryError + '_ {
    move |source| FileMemoryError::Io {
        path: path.display().to_string(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, FileMemoryStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = FileMemoryStore::new(dir.path()).unwrap();
        (dir, store)
    }

    #[test]
    fn create_view_roundtrip() {
        let (_d, s) = store();
        s.create("facts", "# Facts\n\n- Rust is fast\n").unwrap();
        assert_eq!(s.view("facts").unwrap(), "# Facts\n\n- Rust is fast\n");
    }

    #[test]
    fn create_rejects_duplicate() {
        let (_d, s) = store();
        s.create("a", "one").unwrap();
        assert!(matches!(
            s.create("a", "two"),
            Err(FileMemoryError::AlreadyExists(_))
        ));
    }

    #[test]
    fn append_adds_to_existing() {
        let (_d, s) = store();
        s.create("a", "first\n").unwrap();
        s.append("a", "second\n").unwrap();
        assert_eq!(s.view("a").unwrap(), "first\nsecond\n");
    }

    #[test]
    fn append_requires_existing() {
        let (_d, s) = store();
        assert!(matches!(
            s.append("nope", "x"),
            Err(FileMemoryError::NotFound(_))
        ));
    }

    #[test]
    fn str_replace_is_single_and_explicit() {
        let (_d, s) = store();
        s.create("a", "x x x").unwrap();
        s.str_replace("a", "x", "y").unwrap();
        assert_eq!(s.view("a").unwrap(), "y x x");
        // missing old text is an explicit error, not silent no-op
        assert!(matches!(
            s.str_replace("a", "zzz", "q"),
            Err(FileMemoryError::OldTextNotFound(_))
        ));
    }

    #[test]
    fn rename_moves_content() {
        let (_d, s) = store();
        s.create("a", "data").unwrap();
        s.rename("a", "b").unwrap();
        assert!(matches!(s.view("a"), Err(FileMemoryError::NotFound(_))));
        assert_eq!(s.view("b").unwrap(), "data");
    }

    #[test]
    fn rename_onto_existing_fails() {
        let (_d, s) = store();
        s.create("a", "x").unwrap();
        s.create("b", "y").unwrap();
        assert!(matches!(
            s.rename("a", "b"),
            Err(FileMemoryError::AlreadyExists(_))
        ));
    }

    #[test]
    fn delete_removes() {
        let (_d, s) = store();
        s.create("a", "x").unwrap();
        s.delete("a").unwrap();
        assert!(matches!(s.view("a"), Err(FileMemoryError::NotFound(_))));
        assert!(s.list().unwrap().is_empty());
    }

    #[test]
    fn list_newest_first() {
        let (_d, s) = store();
        s.create("old", "1").unwrap();
        s.create("new", "2").unwrap();
        let names: Vec<String> = s.list().unwrap().into_iter().map(|e| e.name).collect();
        assert_eq!(names, vec!["new", "old"]);
    }

    // ---- path safety ----

    #[test]
    fn traversal_is_rejected() {
        let (_d, s) = store();
        for bad in [
            "../evil",
            "..",
            ".",
            "a/../evil",
            "a\\..\\evil",
            "C:\\evil",
            "/etc/passwd",
            "a:b",
        ] {
            assert!(
                matches!(s.create(bad, "x"), Err(FileMemoryError::UnsafeName { .. })),
                "expected rejection for {bad:?}"
            );
        }
    }

    #[test]
    fn reserved_windows_names_rejected() {
        let (_d, s) = store();
        for bad in ["CON", "con", "PRN", "NUL", "COM1", "LPT9", "CON.txt"] {
            assert!(
                matches!(s.create(bad, "x"), Err(FileMemoryError::UnsafeName { .. })),
                "expected rejection for {bad:?}"
            );
        }
    }

    #[test]
    fn empty_and_padded_names_rejected() {
        let (_d, s) = store();
        for bad in ["", " closed"] {
            assert!(
                matches!(s.create(bad, "x"), Err(FileMemoryError::UnsafeName { .. })),
                "expected rejection for {bad:?}"
            );
        }
    }
}
