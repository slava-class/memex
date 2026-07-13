use crate::analytics::{AnalyticsStore, AnalyticsWriter, analytics_path, backfill_from_index};
use crate::config::{IndexedToolContentLimits, Paths};
use crate::embed::{EmbedRuntimeConfig, EmbedderHandle, ModelChoice};
use crate::index::SearchIndex;
use crate::progress::{Progress, SOURCE_COUNT};
use crate::state::{FileState, IngestState, ScanCache, SourceFileState};
use crate::types::{Record, RecordLinks, SourceKind};
use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use memchr::memchr;
use memmap2::Mmap;
use rayon::prelude::*;
use simd_json::BorrowedValue;
use simd_json::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use walkdir::WalkDir;

const EMBED_BATCH_SIZE: usize = 64;
const EMBED_MAX_CHARS: usize = 8192;
const RETAINED_HEAD_PERCENT: usize = 75;
const INDEX_PROGRESS_BATCH: u64 = 1;
// Keep a small amount of parser/writer overlap without retaining an unbounded transcript backlog.
const RECORD_CHANNEL_CAPACITY: usize = 8;
const CURSOR_SUBAGENT_TURN_BASE: u32 = 1_000_000_000;
const CURSOR_SUBAGENT_TURN_STRIDE: u32 = 50_000;
const CURSOR_SUBAGENT_TURN_BUCKETS: u32 = 65_536;

#[derive(Debug, Clone)]
pub struct IngestOptions {
    pub claude_source: PathBuf,
    pub include_agents: bool,
    pub include_codex: bool,
    pub include_opencode: bool,
    pub include_cursor: bool,
    pub include_pi: bool,
    pub include_copilot: bool,
    pub include_omp: bool,
    pub omp_source: PathBuf,
    pub embeddings: bool,
    pub backfill_embeddings: bool,
    pub model: ModelChoice,
    pub embed_runtime: EmbedRuntimeConfig,
    pub tool_content_limits: IndexedToolContentLimits,
}

#[derive(Debug)]
pub struct IngestReport {
    pub records_added: usize,
    pub records_embedded: usize,
    pub files_scanned: usize,
    pub files_skipped: usize,
}

#[derive(Debug)]
struct FileTask {
    path: PathBuf,
    source: SourceKind,
    offset: u64,
    turn_id: u32,
    size: u64,
    mtime: i64,
    delete_first: bool,
}

#[derive(Debug)]
struct FileUpdate {
    path: String,
    state: FileState,
    session_id: Option<String>,
}

struct FileTaskSet<'a> {
    state: &'a IngestState,
    tasks: Vec<FileTask>,
    files_scanned: usize,
    files_skipped: usize,
    total_bytes: u64,
}

impl<'a> FileTaskSet<'a> {
    fn new(state: &'a IngestState) -> Self {
        Self {
            state,
            tasks: Vec::new(),
            files_scanned: 0,
            files_skipped: 0,
            total_bytes: 0,
        }
    }

    fn add_paths<I>(&mut self, paths: I, source: SourceKind) -> Result<()>
    where
        I: IntoIterator<Item = PathBuf>,
    {
        for path in paths {
            let meta = path.metadata()?;
            let size = meta.len();
            let mtime = meta
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|duration| duration.as_secs() as i64)
                .unwrap_or(0);
            self.files_scanned += 1;
            self.total_bytes += size;

            let key = path.to_string_lossy().to_string();
            let previous = self.state.files.get(&key);
            let source_rewritten = if let Some(previous) = previous {
                size >= previous.size
                    && (size != previous.size || mtime != previous.mtime)
                    && source_file_requires_full_reindex(source, &path, previous)?
            } else {
                false
            };
            let (offset, turn_id, delete_first, skip) = match previous {
                None => (0, 0, false, false),
                Some(previous)
                    if source_rewritten || size < previous.size || mtime < previous.mtime =>
                {
                    (0, 0, true, false)
                }
                Some(previous) if size == previous.size && mtime == previous.mtime => {
                    (previous.offset, previous.turn_id, false, true)
                }
                Some(previous) => (previous.offset, previous.turn_id, false, false),
            };
            if skip {
                self.files_skipped += 1;
                continue;
            }
            self.tasks.push(FileTask {
                path,
                source,
                offset,
                turn_id,
                size,
                mtime,
                delete_first,
            });
        }
        Ok(())
    }
}

#[derive(Clone)]
struct RecordSender {
    sender: Sender<Record>,
    limits: IndexedToolContentLimits,
}

impl RecordSender {
    fn new(sender: Sender<Record>, limits: IndexedToolContentLimits) -> Self {
        Self { sender, limits }
    }

    fn send(&self, mut record: Record) -> Result<()> {
        limit_record_tool_content(&mut record, self.limits);
        self.sender.send(record)?;
        Ok(())
    }
}

struct WriterContext {
    embeddings: bool,
    do_backfill_embeddings: bool,
    vector_dir: PathBuf,
    analytics_path: PathBuf,
    progress: Arc<Progress>,
    model: ModelChoice,
    embed_runtime: EmbedRuntimeConfig,
    tool_content_limits: IndexedToolContentLimits,
}

fn record_channel() -> (Sender<Record>, Receiver<Record>) {
    bounded(RECORD_CHANNEL_CAPACITY)
}

/// Check if scan cache is fresh and vector state is usable; if so, skip indexing entirely.
/// Returns Ok(None) if skipped due to fresh cache, Ok(Some(report)) if indexing ran.
pub fn ingest_if_stale(
    paths: &Paths,
    index: &SearchIndex,
    options: &IngestOptions,
    ttl_seconds: u64,
) -> Result<Option<IngestReport>> {
    let cache_path = paths.state.join("scan_cache.json");
    let cache = ScanCache::load(&cache_path)?;

    if can_skip_fresh_scan(&cache, paths, index, options, ttl_seconds)? {
        return Ok(None);
    }

    let report = ingest_all(paths, index, options)?;
    Ok(Some(report))
}

pub fn ingest_all(
    paths: &Paths,
    index: &SearchIndex,
    options: &IngestOptions,
) -> Result<IngestReport> {
    let state_path = paths.state.join("ingest.json");
    let mut state = IngestState::load(&state_path)?;
    if index.doc_count()? == 0 && !state.files.is_empty() {
        state = IngestState::default();
        if paths.vectors.exists() {
            std::fs::remove_dir_all(&paths.vectors)?;
            std::fs::create_dir_all(&paths.vectors)?;
        }
    }
    let next_doc_id = Arc::new(AtomicU64::new(state.next_doc_id));

    let mut task_set = FileTaskSet::new(&state);
    if options.claude_source.exists() {
        task_set.add_paths(
            collect_claude_files(&options.claude_source, options.include_agents)?,
            SourceKind::Claude,
        )?;
    }

    let mut session_ids = HashSet::new();
    if options.include_codex {
        let codex_files = collect_codex_session_files()?;
        for path in &codex_files {
            if let Some(id) = session_id_from_filename(path) {
                session_ids.insert(id);
            }
        }
        task_set.add_paths(codex_files, SourceKind::CodexSession)?;
    }

    if options.include_codex {
        let history_path = codex_history_path();
        if history_path.exists() {
            task_set.add_paths([history_path], SourceKind::CodexHistory)?;
        }
    }

    if options.include_opencode {
        task_set.add_paths(collect_opencode_files()?, SourceKind::Opencode)?;
    }

    if options.include_cursor {
        task_set.add_paths(collect_cursor_files()?, SourceKind::Cursor)?;
    }

    if options.include_pi {
        task_set.add_paths(collect_pi_files()?, SourceKind::Pi)?;
    }

    if options.include_copilot {
        task_set.add_paths(collect_copilot_files()?, SourceKind::Copilot)?;
    }

    if options.include_omp {
        task_set.add_paths(
            collect_omp_files_from_root(&options.omp_source)?,
            SourceKind::Omp,
        )?;
    }

    let FileTaskSet {
        tasks,
        files_scanned,
        files_skipped,
        total_bytes,
        ..
    } = task_set;

    let opencode_session_links = if tasks.iter().any(|task| task.source == SourceKind::Opencode) {
        opencode_session_links_by_id()
    } else {
        HashMap::new()
    };

    let totals = compute_totals(&tasks);
    let file_totals = compute_file_totals(&tasks);
    let analytics_db = analytics_path(&paths.state);
    let analytics_needs_backfill =
        !AnalyticsStore::is_complete(&analytics_db) && index.doc_count()? > 0;
    if tasks.is_empty() && can_skip_noop_index(paths, index, options)? {
        if analytics_needs_backfill {
            backfill_from_index(&analytics_db, index)?;
        }
        update_scan_cache(paths, files_scanned, total_bytes);
        return Ok(IngestReport {
            records_added: 0,
            records_embedded: 0,
            files_scanned,
            files_skipped,
        });
    }

    let progress = Arc::new(Progress::new(totals, file_totals, options.embeddings));

    let (raw_tx_record, rx_record) = record_channel();
    let tx_record = RecordSender::new(raw_tx_record, options.tool_content_limits);
    let (tx_update, rx_update) = unbounded::<FileUpdate>();

    let delete_paths: Vec<String> = tasks
        .iter()
        .filter(|t| t.delete_first)
        .map(|t| t.path.to_string_lossy().to_string())
        .collect();

    let writer_index = index.clone();
    let writer_ctx = WriterContext {
        embeddings: options.embeddings,
        do_backfill_embeddings: options.backfill_embeddings,
        vector_dir: paths.vectors.clone(),
        analytics_path: analytics_db.clone(),
        progress: progress.clone(),
        model: options.model,
        embed_runtime: options.embed_runtime.clone(),
        tool_content_limits: options.tool_content_limits,
    };
    let writer_handle =
        std::thread::spawn(move || writer_loop(writer_index, rx_record, delete_paths, writer_ctx));

    let tasks_arc = Arc::new(tasks);
    tasks_arc.par_iter().try_for_each(|task| -> Result<()> {
        match task.source {
            SourceKind::Claude => {
                parse_claude_file(task, &tx_record, &tx_update, &next_doc_id, &progress)?
            }
            SourceKind::CodexSession => {
                parse_codex_session(task, &tx_record, &tx_update, &next_doc_id, &progress)?
            }
            SourceKind::CodexHistory => parse_codex_history(
                task,
                &tx_record,
                &tx_update,
                &next_doc_id,
                &session_ids,
                &progress,
            )?,
            SourceKind::Opencode => parse_opencode_file(
                task,
                &tx_record,
                &tx_update,
                &next_doc_id,
                &progress,
                &opencode_session_links,
            )?,
            SourceKind::Cursor => {
                parse_cursor_file(task, &tx_record, &tx_update, &next_doc_id, &progress)?
            }
            SourceKind::Pi => parse_pi_file(task, &tx_record, &tx_update, &next_doc_id, &progress)?,
            SourceKind::Copilot => {
                parse_copilot_session(task, &tx_record, &tx_update, &next_doc_id, &progress)?
            }
            SourceKind::Omp => {
                parse_omp_file(task, &tx_record, &tx_update, &next_doc_id, &progress)?
            }
        }
        Ok(())
    })?;

    drop(tx_record);
    drop(tx_update);

    let writer_result = writer_handle
        .join()
        .map_err(|_| anyhow!("writer thread panicked"))?;
    progress.finish();
    let (records_added, records_embedded) = writer_result?;
    if analytics_needs_backfill {
        backfill_from_index(&analytics_db, index)?;
    } else {
        AnalyticsStore::open(&analytics_db)?.mark_complete()?;
    }

    let mut updated_files = HashMap::new();
    while let Ok(update) = rx_update.recv() {
        updated_files.insert(update.path.clone(), update.state.clone());
        let _ = update.session_id;
    }

    for (path, update) in updated_files {
        state.files.insert(path, update);
    }
    state.next_doc_id = next_doc_id.load(Ordering::SeqCst);
    state.save(&state_path)?;

    update_scan_cache(paths, files_scanned, total_bytes);

    Ok(IngestReport {
        records_added,
        records_embedded,
        files_scanned,
        files_skipped,
    })
}

fn update_scan_cache(paths: &Paths, files_scanned: usize, total_bytes: u64) {
    let cache_path = paths.state.join("scan_cache.json");
    let mut cache = ScanCache::load(&cache_path).unwrap_or_default();
    cache.update(files_scanned, total_bytes);
    let _ = cache.save(&cache_path);
}

fn can_skip_fresh_scan(
    cache: &ScanCache,
    paths: &Paths,
    index: &SearchIndex,
    options: &IngestOptions,
    ttl_seconds: u64,
) -> Result<bool> {
    if index.doc_count()? == 0 {
        return Ok(false);
    }
    if !cache.is_fresh(ttl_seconds) {
        return Ok(false);
    }
    if !AnalyticsStore::is_complete(analytics_path(&paths.state)) && index.doc_count()? > 0 {
        return Ok(false);
    }
    can_skip_noop_index(paths, index, options)
}

fn can_skip_noop_index(
    paths: &Paths,
    index: &SearchIndex,
    options: &IngestOptions,
) -> Result<bool> {
    if !options.embeddings {
        return Ok(true);
    }
    let Some(dimensions) = options.model.known_dimensions() else {
        return Ok(false);
    };
    if !paths.vectors.join("usearch.index").exists() {
        return Ok(false);
    }
    let vector_index = crate::vector::VectorIndex::open(&paths.vectors)?;
    if vector_index.model() != Some(options.model.as_str())
        || vector_index.dimensions() != dimensions
    {
        return Ok(false);
    }
    vector_index_covers_embeddable_records(index, &vector_index)
}

fn vector_index_covers_embeddable_records(
    index: &SearchIndex,
    vector_index: &crate::vector::VectorIndex,
) -> Result<bool> {
    let mut covers_all = true;
    index.for_each_record(|record| {
        if record_needs_embedding(&record) && !vector_index.contains(record.doc_id) {
            covers_all = false;
        }
        Ok(())
    })?;
    Ok(covers_all)
}

fn record_needs_embedding(record: &Record) -> bool {
    is_embedding_role(&record.role) && !record.text.is_empty()
}

fn writer_loop(
    index: SearchIndex,
    rx: Receiver<Record>,
    delete_paths: Vec<String>,
    ctx: WriterContext,
) -> Result<(usize, usize)> {
    let WriterContext {
        embeddings,
        do_backfill_embeddings,
        vector_dir,
        analytics_path,
        progress,
        model,
        embed_runtime,
        tool_content_limits,
    } = ctx;
    let mut writer = index.writer()?;
    let mut analytics = AnalyticsWriter::open(&analytics_path)?;
    for path in delete_paths {
        index.delete_by_source_path(&mut writer, &path);
        analytics.delete_source_path(&path)?;
    }

    let mut count = 0usize;
    let mut embedded_count = 0usize;
    let mut vector_index = None;
    let mut embedder: Option<EmbedderHandle> = None;
    let mut embed_buffer: Vec<(u64, String, SourceKind)> = Vec::new();
    let mut index_pending = [0u64; SOURCE_COUNT];
    if embeddings {
        let handle = EmbedderHandle::with_model_and_runtime(model, &embed_runtime)?;
        let dims = handle.dims;
        vector_index = Some(crate::vector::VectorIndex::open_or_create(
            &vector_dir,
            dims,
            Some(model.as_str()),
        )?);
        embedder = Some(handle);
        progress.set_embed_ready();
    }

    for mut record in rx.iter() {
        // Parsers apply the limit before queueing; enforce it here as a defensive boundary too.
        limit_record_tool_content(&mut record, tool_content_limits);
        analytics.record(&record)?;
        index.add_record(&mut writer, &record)?;
        let source_idx = record.source.idx();
        index_pending[source_idx] += 1;
        if index_pending[source_idx] >= INDEX_PROGRESS_BATCH {
            progress.add_indexed(record.source, index_pending[source_idx]);
            index_pending[source_idx] = 0;
        }
        if embeddings && is_embedding_role(&record.role) && !record.text.is_empty() {
            let text = truncate_for_embedding(std::mem::take(&mut record.text));
            if let Some(vindex) = vector_index.as_ref()
                && !vindex.contains(record.doc_id)
            {
                progress.add_embed_total(record.source, 1);
                progress.add_embed_pending(record.source, 1);
                embed_buffer.push((record.doc_id, text, record.source));
            }
            if let Some(emb) = embedder.as_mut()
                && embed_buffer.len() >= EMBED_BATCH_SIZE
            {
                embedded_count += flush_embeddings(
                    &mut embed_buffer,
                    emb,
                    vector_index.as_mut().unwrap(),
                    &progress,
                )?;
            }
        }
        count += 1;
    }

    // Flush any remaining index progress
    for (idx, &pending) in index_pending.iter().enumerate() {
        if pending > 0
            && let Some(source) = SourceKind::from_idx(idx)
        {
            progress.add_indexed(source, pending);
        }
    }

    analytics.flush()?;
    writer.commit()?;
    if embeddings {
        if !embed_buffer.is_empty() {
            embedded_count += flush_embeddings(
                &mut embed_buffer,
                embedder.as_mut().unwrap(),
                vector_index.as_mut().unwrap(),
                &progress,
            )?;
        }

        let needs_vector_backfill = match vector_index.as_ref() {
            Some(vindex) => {
                vindex.needs_backfill() || !vector_index_covers_embeddable_records(&index, vindex)?
            }
            None => false,
        };
        if do_backfill_embeddings || needs_vector_backfill {
            embedded_count += backfill_embeddings(
                &index,
                embedder.as_mut().unwrap(),
                vector_index.as_mut().unwrap(),
                &progress,
            )?;
        }
        if let Some(vindex) = vector_index.as_mut() {
            vindex.save()?;
        }
        if let Some(handle) = embedder.take() {
            std::mem::forget(handle);
        }
    }
    Ok((count, embedded_count))
}

fn backfill_embeddings(
    index: &SearchIndex,
    embedder: &mut EmbedderHandle,
    vector_index: &mut crate::vector::VectorIndex,
    progress: &Arc<Progress>,
) -> Result<usize> {
    use std::cell::Cell;
    let embedded_count = Cell::new(0usize);
    let mut embed_buffer: Vec<(u64, String, SourceKind)> = Vec::new();
    index.for_each_record(|record| {
        if record.text.is_empty()
            || !is_embedding_role(&record.role)
            || vector_index.contains(record.doc_id)
        {
            return Ok(());
        }
        progress.add_embed_total(record.source, 1);
        progress.add_embed_pending(record.source, 1);
        embed_buffer.push((
            record.doc_id,
            truncate_for_embedding(record.text),
            record.source,
        ));
        if embed_buffer.len() >= EMBED_BATCH_SIZE {
            let n = flush_embeddings(&mut embed_buffer, embedder, vector_index, progress)?;
            embedded_count.set(embedded_count.get() + n);
        }
        Ok(())
    })?;
    if !embed_buffer.is_empty() {
        let n = flush_embeddings(&mut embed_buffer, embedder, vector_index, progress)?;
        embedded_count.set(embedded_count.get() + n);
    }
    Ok(embedded_count.get())
}

fn collect_claude_files(source: &Path, include_agents: bool) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in WalkDir::new(source).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        if !include_agents
            && let Some(name) = path.file_name().and_then(|n| n.to_str())
            && name.starts_with("agent-")
        {
            continue;
        }
        files.push(path.to_path_buf());
    }
    Ok(files)
}

fn collect_codex_session_files() -> Result<Vec<PathBuf>> {
    collect_codex_session_files_from_roots(&codex_session_roots())
}

fn collect_codex_session_files_from_roots(roots: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for root in roots {
        if !root.exists() {
            continue;
        }
        for entry in WalkDir::new(root).into_iter().filter_map(Result::ok) {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            files.push(path.to_path_buf());
        }
    }
    Ok(files)
}

fn collect_opencode_files() -> Result<Vec<PathBuf>> {
    let root = opencode_root();
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    // In opencode, "sessions" are directories inside storage/message/
    // e.g. storage/message/ses_.../
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir()
            && let Some(name) = path.file_name().and_then(|n| n.to_str())
            && name.starts_with("ses_")
        {
            files.push(path);
        }
    }
    Ok(files)
}

fn collect_cursor_files() -> Result<Vec<PathBuf>> {
    let root = cursor_projects_root();
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in WalkDir::new(root).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Some(path_str) = path.to_str() else {
            continue;
        };
        if path_str.contains("/agent-transcripts/") || path_str.contains("\\agent-transcripts\\") {
            files.push(path.to_path_buf());
        }
    }
    files.sort();
    Ok(files)
}

fn collect_pi_files() -> Result<Vec<PathBuf>> {
    collect_pi_files_from_root(&pi_sessions_root())
}

fn collect_pi_files_from_root(root: &Path) -> Result<Vec<PathBuf>> {
    collect_jsonl_files_from_root(root)
}

fn collect_omp_files_from_root(root: &Path) -> Result<Vec<PathBuf>> {
    collect_jsonl_files_from_root(root)
}

fn collect_jsonl_files_from_root(root: &Path) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in WalkDir::new(root).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        files.push(path.to_path_buf());
    }
    files.sort();
    Ok(files)
}

fn collect_copilot_files() -> Result<Vec<PathBuf>> {
    collect_copilot_files_from_root(&copilot_session_root())
}

fn collect_copilot_files_from_root(root: &Path) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in WalkDir::new(root).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.file_name().and_then(|n| n.to_str()) == Some("events.jsonl") {
            files.push(path.to_path_buf());
        }
    }
    Ok(files)
}

fn codex_root() -> PathBuf {
    let home = directories::BaseDirs::new()
        .map(|b| b.home_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("/"));
    home.join(".codex")
}

fn codex_session_roots() -> Vec<PathBuf> {
    let codex_root = codex_root();
    vec![
        codex_root.join("sessions"),
        codex_root.join("archived_sessions"),
    ]
}

fn opencode_root() -> PathBuf {
    let home = directories::BaseDirs::new()
        .map(|b| b.home_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("/"));
    home.join(".local")
        .join("share")
        .join("opencode")
        .join("storage")
        .join("message")
}

fn copilot_root() -> PathBuf {
    if let Some(path) = std::env::var_os("COPILOT_HOME") {
        return PathBuf::from(path);
    }
    let home = directories::BaseDirs::new()
        .map(|b| b.home_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("/"));
    home.join(".copilot")
}

fn copilot_session_root() -> PathBuf {
    copilot_root().join("session-state")
}

fn opencode_parts_root() -> PathBuf {
    let home = directories::BaseDirs::new()
        .map(|b| b.home_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("/"));
    home.join(".local")
        .join("share")
        .join("opencode")
        .join("storage")
        .join("part")
}

fn opencode_default_session_links() -> SessionLinks {
    SessionLinks {
        conversation_kind: Some("main".to_string()),
        ..SessionLinks::default()
    }
}

fn opencode_session_links_by_id() -> HashMap<String, SessionLinks> {
    opencode_session_links_by_id_from_root(&opencode_storage_root().join("session"))
}

fn opencode_session_links_by_id_from_root(root: &Path) -> HashMap<String, SessionLinks> {
    let mut links_by_id = HashMap::new();
    if !root.exists() {
        return links_by_id;
    }

    for entry in WalkDir::new(root).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Some(session_id) = entry
            .path()
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
        else {
            continue;
        };
        let Ok(file) = File::open(entry.path()) else {
            continue;
        };
        let reader = std::io::BufReader::new(file);
        let Ok(value) = serde_json::from_reader::<_, serde_json::Value>(reader) else {
            continue;
        };

        links_by_id.insert(session_id, opencode_session_links_from_value(&value));
    }

    links_by_id
}

#[cfg(test)]
fn opencode_session_links_from_root(root: &Path, session_id: &str) -> SessionLinks {
    opencode_session_links_by_id_from_root(root)
        .remove(session_id)
        .unwrap_or_else(opencode_default_session_links)
}

fn opencode_session_links_from_value(value: &serde_json::Value) -> SessionLinks {
    let parent_session_id = value
        .get("parentID")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    SessionLinks {
        conversation_kind: Some(if parent_session_id.is_some() {
            "fork".to_string()
        } else {
            "main".to_string()
        }),
        thread_source: parent_session_id.as_ref().map(|_| "fork".to_string()),
        parent_session_id,
    }
}

fn opencode_storage_root() -> PathBuf {
    let home = directories::BaseDirs::new()
        .map(|b| b.home_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("/"));
    home.join(".local")
        .join("share")
        .join("opencode")
        .join("storage")
}

fn codex_history_path() -> PathBuf {
    codex_root().join("history.jsonl")
}

fn cursor_projects_root() -> PathBuf {
    let home = directories::BaseDirs::new()
        .map(|b| b.home_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("/"));
    home.join(".cursor").join("projects")
}

fn pi_agent_root() -> PathBuf {
    if let Some(root) = std::env::var_os("PI_CODING_AGENT_DIR") {
        return PathBuf::from(root);
    }
    let home = directories::BaseDirs::new()
        .map(|b| b.home_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("/"));
    home.join(".pi").join("agent")
}

fn pi_sessions_root() -> PathBuf {
    if let Some(root) = std::env::var_os("PI_CODING_AGENT_SESSION_DIR") {
        return PathBuf::from(root);
    }
    let agent_root = pi_agent_root();
    if let Some(root) = pi_configured_session_root(&agent_root) {
        return root;
    }
    agent_root.join("sessions")
}

fn pi_configured_session_root(agent_root: &Path) -> Option<PathBuf> {
    let settings_path = agent_root.join("settings.json");
    let contents = std::fs::read_to_string(settings_path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&contents).ok()?;
    let session_dir = value.get("sessionDir")?.as_str()?.trim();
    if session_dir.is_empty() {
        return None;
    }
    Some(resolve_pi_settings_path(session_dir, agent_root))
}

fn resolve_pi_settings_path(raw: &str, base: &Path) -> PathBuf {
    if raw == "~" {
        return directories::BaseDirs::new()
            .map(|b| b.home_dir().to_path_buf())
            .unwrap_or_else(|| PathBuf::from(raw));
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        return directories::BaseDirs::new()
            .map(|b| b.home_dir().join(rest))
            .unwrap_or_else(|| PathBuf::from(raw));
    }
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

#[derive(Clone, Default)]
struct SessionLinks {
    parent_session_id: Option<String>,
    thread_source: Option<String>,
    conversation_kind: Option<String>,
}

impl SessionLinks {
    fn record_links(&self) -> RecordLinks {
        RecordLinks {
            parent_session_id: self.parent_session_id.clone(),
            thread_source: self.thread_source.clone(),
            conversation_kind: self.conversation_kind.clone(),
            ..RecordLinks::default()
        }
    }
}

#[derive(Clone)]
struct CodexSessionMeta {
    session_id: String,
    project: String,
    links: SessionLinks,
}

fn codex_session_meta_from_path(path: &Path) -> CodexSessionMeta {
    CodexSessionMeta {
        session_id: session_id_from_filename(path).unwrap_or_else(|| "unknown".to_string()),
        project: "codex".to_string(),
        links: SessionLinks {
            conversation_kind: Some("main".to_string()),
            ..SessionLinks::default()
        },
    }
}

fn read_codex_session_meta_until(path: &Path, limit: u64) -> Result<CodexSessionMeta> {
    let mut meta = codex_session_meta_from_path(path);
    if limit == 0 {
        return Ok(meta);
    }
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut start = 0usize;
    let limit = (limit as usize).min(mmap.len());
    let mut buf = Vec::new();
    while start < limit {
        let slice = &mmap[start..limit];
        let rel = memchr(b'\n', slice).unwrap_or(slice.len());
        let line = &slice[..rel];
        start += rel + 1;
        if line.is_empty() {
            continue;
        }
        buf.clear();
        buf.extend_from_slice(line);
        let value: BorrowedValue = match simd_json::to_borrowed_value(&mut buf) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if value.get("type").and_then(|v| v.as_str()) == Some("session_meta")
            && let Some(payload) = value.get("payload").and_then(|v| v.as_object())
        {
            apply_codex_session_meta(payload, &mut meta);
        }
    }
    Ok(meta)
}

fn apply_codex_session_meta(payload: &simd_json::borrowed::Object, meta: &mut CodexSessionMeta) {
    if let Some(id) = payload.get("id").and_then(|v| v.as_str()) {
        meta.session_id = id.to_string();
    }
    if let Some(cwd) = payload.get("cwd").and_then(|v| v.as_str()) {
        meta.project = project_from_path(cwd);
    }

    let forked_from_id = payload
        .get("forked_from_id")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let parent_thread_id = payload
        .get("source")
        .and_then(|v| v.as_object())
        .and_then(|source| source.get("subagent"))
        .and_then(|v| v.as_object())
        .and_then(|subagent| subagent.get("thread_spawn"))
        .and_then(|v| v.as_object())
        .and_then(|spawn| spawn.get("parent_thread_id"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let thread_source = payload
        .get("thread_source")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .or_else(|| parent_thread_id.as_ref().map(|_| "subagent".to_string()))
        .or_else(|| forked_from_id.as_ref().map(|_| "fork".to_string()));

    meta.links.parent_session_id = forked_from_id.clone().or(parent_thread_id);
    meta.links.thread_source = thread_source.clone();
    meta.links.conversation_kind = Some(if thread_source.as_deref() == Some("subagent") {
        "subagent".to_string()
    } else if forked_from_id.is_some() {
        "fork".to_string()
    } else {
        "main".to_string()
    });
}

fn opt_str(obj: &simd_json::borrowed::Object, key: &str) -> Option<String> {
    obj.get(key).and_then(|v| v.as_str()).map(str::to_string)
}

fn pi_base_links(obj: &simd_json::borrowed::Object, conversation_kind: &str) -> RecordLinks {
    RecordLinks {
        event_id: opt_str(obj, "id"),
        parent_event_id: opt_str(obj, "parentId"),
        logical_parent_event_id: opt_str(obj, "fromId"),
        thread_source: (conversation_kind != "main").then(|| conversation_kind.to_string()),
        conversation_kind: Some(conversation_kind.to_string()),
        ..RecordLinks::default()
    }
}

fn parse_claude_file(
    task: &FileTask,
    tx_record: &RecordSender,
    tx_update: &Sender<FileUpdate>,
    next_doc_id: &AtomicU64,
    progress: &Arc<Progress>,
) -> Result<()> {
    let file = File::open(&task.path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut start = task.offset as usize;
    let mut turn_id = task.turn_id;

    let project = project_from_claude_path(&task.path);
    let session_id = task
        .path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    let is_agent_file = session_id.starts_with("agent-")
        || task
            .path
            .components()
            .any(|component| component.as_os_str().to_str() == Some("subagents"));
    let source_path = task.path.to_string_lossy().to_string();
    let mut tool_id_to_name: HashMap<String, String> = HashMap::new();

    let mut buf = Vec::new();
    let mut parsed_bytes = 0u64;
    while start < mmap.len() {
        let slice = &mmap[start..];
        let rel = memchr(b'\n', slice).unwrap_or(slice.len());
        let line = &slice[..rel];
        let advanced = rel + 1;
        start += advanced;
        parsed_bytes += advanced as u64;
        if parsed_bytes >= 64 * 1024 {
            progress.add_parsed_bytes(SourceKind::Claude, parsed_bytes);
            parsed_bytes = 0;
        }
        if line.is_empty() {
            continue;
        }
        buf.clear();
        buf.extend_from_slice(line);
        let value: BorrowedValue = match simd_json::to_borrowed_value(&mut buf) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let obj = match value.as_object() {
            Some(o) => o,
            None => continue,
        };
        let entry_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if entry_type != "user" && entry_type != "assistant" {
            continue;
        }
        let entry_uuid = opt_str(obj, "uuid");
        let entry_parent_uuid = opt_str(obj, "parentUuid");
        let is_sidechain = obj
            .get("isSidechain")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let conversation_kind = if is_agent_file {
            "subagent"
        } else if is_sidechain {
            "sidechain"
        } else {
            "main"
        };
        let thread_source = if is_agent_file {
            Some("subagent".to_string())
        } else if is_sidechain {
            Some("sidechain".to_string())
        } else {
            None
        };
        let entry_links = RecordLinks {
            event_id: entry_uuid.clone(),
            parent_event_id: entry_parent_uuid,
            logical_parent_event_id: opt_str(obj, "logicalParentUuid"),
            parent_session_id: is_agent_file.then(|| opt_str(obj, "sessionId")).flatten(),
            thread_source,
            conversation_kind: Some(conversation_kind.to_string()),
            parent_tool_use_id: opt_str(obj, "parentToolUseID"),
            source_tool_use_id: opt_str(obj, "sourceToolUseID"),
            source_tool_assistant_uuid: opt_str(obj, "sourceToolAssistantUUID"),
        };
        let timestamp = obj
            .get("timestamp")
            .and_then(|v| v.as_str())
            .and_then(parse_iso_millis)
            .unwrap_or(0);
        let message = match obj.get("message").and_then(|v| v.as_object()) {
            Some(m) => m,
            None => continue,
        };
        let content = message.get("content");
        let mut text_parts = Vec::new();
        if let Some(content) = content {
            if let Some(text) = content.as_str() {
                text_parts.push(text);
            } else if let Some(arr) = content.as_array() {
                for block in arr {
                    let block_obj = match block.as_object() {
                        Some(b) => b,
                        None => continue,
                    };
                    let block_type = block_obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if block_type == "text" {
                        if let Some(text) = block_obj.get("text").and_then(|v| v.as_str()) {
                            text_parts.push(text);
                        }
                    } else if block_type == "tool_use" {
                        let tool_name = block_obj
                            .get("name")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        if let (Some(id), Some(name)) = (
                            block_obj.get("id").and_then(|v| v.as_str()),
                            tool_name.clone(),
                        ) {
                            tool_id_to_name.insert(id.to_string(), name);
                        }
                        let tool_input = block_obj.get("input").map(|v| v.to_string());
                        let text = tool_input.clone().unwrap_or_default();
                        let mut links = entry_links.clone();
                        if let Some(tool_id) = block_obj.get("id").and_then(|v| v.as_str()) {
                            links.event_id = Some(tool_id.to_string());
                            links.parent_event_id = entry_uuid.clone();
                        }
                        let record = Record {
                            source: SourceKind::Claude,
                            doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                            ts: timestamp,
                            project: project.clone(),
                            session_id: session_id.clone(),
                            turn_id,
                            role: "tool_use".to_string(),
                            text,
                            tool_name,
                            tool_input,
                            tool_output: None,
                            links,
                            source_path: source_path.clone(),
                        };
                        progress.add_produced(SourceKind::Claude, 1);
                        tx_record.send(record)?;
                        turn_id += 1;
                    }
                }
            }
        }

        if entry_type == "user"
            && let Some(content) = content
            && let Some(arr) = content.as_array()
        {
            for block in arr {
                let block_obj = match block.as_object() {
                    Some(b) => b,
                    None => continue,
                };
                if block_obj.get("type").and_then(|v| v.as_str()) == Some("tool_result") {
                    let tool_output = block_obj.get("content").map(|v| v.to_string());
                    let mut text = extract_text_from_tool_result(block).unwrap_or_default();
                    if text.is_empty()
                        && let Some(content) = block_obj.get("content")
                    {
                        text = content.to_string();
                    }
                    let tool_name = block_obj
                        .get("tool_use_id")
                        .and_then(|v| v.as_str())
                        .and_then(|id| tool_id_to_name.get(id))
                        .cloned();
                    let tool_use_id = block_obj
                        .get("tool_use_id")
                        .and_then(|v| v.as_str())
                        .map(str::to_string);
                    let mut links = entry_links.clone();
                    if let Some(tool_use_id) = &tool_use_id {
                        links.event_id = entry_uuid
                            .as_ref()
                            .map(|uuid| format!("{uuid}:tool_result:{tool_use_id}"));
                        links.parent_event_id = Some(tool_use_id.clone());
                        links.parent_tool_use_id = Some(tool_use_id.clone());
                    }
                    let record = Record {
                        source: SourceKind::Claude,
                        doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                        ts: timestamp,
                        project: project.clone(),
                        session_id: session_id.clone(),
                        turn_id,
                        role: "tool_result".to_string(),
                        text,
                        tool_name,
                        tool_input: None,
                        tool_output,
                        links,
                        source_path: source_path.clone(),
                    };
                    progress.add_produced(SourceKind::Claude, 1);
                    tx_record.send(record)?;
                    turn_id += 1;
                }
            }
        }

        let text = text_parts.join(" ").trim().to_string();
        if !text.is_empty() {
            let record = Record {
                source: SourceKind::Claude,
                doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                ts: timestamp,
                project: project.clone(),
                session_id: session_id.clone(),
                turn_id,
                role: entry_type.to_string(),
                text,
                tool_name: None,
                tool_input: None,
                tool_output: None,
                links: entry_links,
                source_path: source_path.clone(),
            };
            progress.add_produced(SourceKind::Claude, 1);
            tx_record.send(record)?;
            turn_id += 1;
        }
    }

    if parsed_bytes > 0 {
        progress.add_parsed_bytes(SourceKind::Claude, parsed_bytes);
    }
    progress.add_files_done(SourceKind::Claude, 1);
    let state = FileState {
        size: task.size,
        mtime: task.mtime,
        offset: mmap.len() as u64,
        turn_id,
        source_state: None,
    };
    tx_update.send(FileUpdate {
        path: source_path,
        state,
        session_id: Some(session_id),
    })?;
    Ok(())
}

fn parse_codex_session(
    task: &FileTask,
    tx_record: &RecordSender,
    tx_update: &Sender<FileUpdate>,
    next_doc_id: &AtomicU64,
    progress: &Arc<Progress>,
) -> Result<()> {
    let file = File::open(&task.path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut start = task.offset as usize;
    let mut turn_id = task.turn_id;

    let source_path = task.path.to_string_lossy().to_string();
    let mut meta = read_codex_session_meta_until(&task.path, task.offset)?;
    let mut call_id_to_name: HashMap<String, String> = HashMap::new();

    let mut buf = Vec::new();
    let mut parsed_bytes = 0u64;
    while start < mmap.len() {
        let slice = &mmap[start..];
        let rel = memchr(b'\n', slice).unwrap_or(slice.len());
        let line = &slice[..rel];
        let advanced = rel + 1;
        start += advanced;
        parsed_bytes += advanced as u64;
        if parsed_bytes >= 64 * 1024 {
            progress.add_parsed_bytes(SourceKind::CodexSession, parsed_bytes);
            parsed_bytes = 0;
        }
        if line.is_empty() {
            continue;
        }
        buf.clear();
        buf.extend_from_slice(line);
        let value: BorrowedValue = match simd_json::to_borrowed_value(&mut buf) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let obj = match value.as_object() {
            Some(o) => o,
            None => continue,
        };
        let entry_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let timestamp = obj
            .get("timestamp")
            .and_then(|v| v.as_str())
            .and_then(parse_iso_millis)
            .unwrap_or(0);
        if entry_type == "session_meta" {
            if let Some(payload) = obj.get("payload").and_then(|v| v.as_object()) {
                apply_codex_session_meta(payload, &mut meta);
            }
            continue;
        }
        if entry_type != "response_item" {
            continue;
        }
        let payload = match obj.get("payload").and_then(|v| v.as_object()) {
            Some(p) => p,
            None => continue,
        };
        let payload_type = payload.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let mut base_links = meta.links.record_links();
        base_links.event_id = opt_str(payload, "id");
        if payload_type == "message" {
            let role = payload.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let content = payload.get("content");
            let mut text_parts = Vec::new();
            if let Some(content) = content {
                if let Some(text) = content.as_str() {
                    text_parts.push(text);
                } else if let Some(arr) = content.as_array() {
                    for block in arr {
                        if let Some(block_obj) = block.as_object()
                            && let Some(text) = block_obj.get("text").and_then(|v| v.as_str())
                        {
                            text_parts.push(text);
                        }
                    }
                }
            }
            let text = text_parts.join("\n").trim().to_string();
            if text.is_empty() {
                continue;
            }
            if is_system_instruction(&text) {
                continue;
            }
            let record = Record {
                source: SourceKind::CodexSession,
                doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                ts: timestamp,
                project: meta.project.clone(),
                session_id: meta.session_id.clone(),
                turn_id,
                role: role.to_string(),
                text,
                tool_name: None,
                tool_input: None,
                tool_output: None,
                links: base_links,
                source_path: source_path.clone(),
            };
            progress.add_produced(SourceKind::CodexSession, 1);
            tx_record.send(record)?;
            turn_id += 1;
        } else if payload_type == "function_call" {
            let tool_name = payload
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let tool_input = payload
                .get("arguments")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            if let Some(call_id) = payload.get("call_id").and_then(|v| v.as_str())
                && let Some(name) = tool_name.clone()
            {
                call_id_to_name.insert(call_id.to_string(), name);
            }
            let text = tool_input.clone().unwrap_or_default();
            let mut links = base_links;
            if let Some(call_id) = payload.get("call_id").and_then(|v| v.as_str()) {
                links.event_id = Some(call_id.to_string());
            }
            let record = Record {
                source: SourceKind::CodexSession,
                doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                ts: timestamp,
                project: meta.project.clone(),
                session_id: meta.session_id.clone(),
                turn_id,
                role: "tool_use".to_string(),
                text,
                tool_name,
                tool_input,
                tool_output: None,
                links,
                source_path: source_path.clone(),
            };
            progress.add_produced(SourceKind::CodexSession, 1);
            tx_record.send(record)?;
            turn_id += 1;
        } else if payload_type == "function_call_output" {
            let call_id = payload
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let tool_name = call_id_to_name.get(call_id).cloned();
            let tool_output = payload
                .get("output")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let text = tool_output.clone().unwrap_or_default();
            if text.is_empty() {
                continue;
            }
            let mut links = base_links;
            if !call_id.is_empty() {
                links.parent_event_id = Some(call_id.to_string());
                links.parent_tool_use_id = Some(call_id.to_string());
            }
            let record = Record {
                source: SourceKind::CodexSession,
                doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                ts: timestamp,
                project: meta.project.clone(),
                session_id: meta.session_id.clone(),
                turn_id,
                role: "tool_result".to_string(),
                text,
                tool_name,
                tool_input: None,
                tool_output,
                links,
                source_path: source_path.clone(),
            };
            progress.add_produced(SourceKind::CodexSession, 1);
            tx_record.send(record)?;
            turn_id += 1;
        }
    }

    if parsed_bytes > 0 {
        progress.add_parsed_bytes(SourceKind::CodexSession, parsed_bytes);
    }
    progress.add_files_done(SourceKind::CodexSession, 1);
    let state = FileState {
        size: task.size,
        mtime: task.mtime,
        offset: mmap.len() as u64,
        turn_id,
        source_state: None,
    };
    tx_update.send(FileUpdate {
        path: source_path,
        state,
        session_id: Some(meta.session_id),
    })?;
    Ok(())
}

fn parse_codex_history(
    task: &FileTask,
    tx_record: &RecordSender,
    tx_update: &Sender<FileUpdate>,
    next_doc_id: &AtomicU64,
    session_ids: &HashSet<String>,
    progress: &Arc<Progress>,
) -> Result<()> {
    let file = File::open(&task.path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut start = task.offset as usize;
    let mut turn_id = task.turn_id;
    let source_path = task.path.to_string_lossy().to_string();

    let mut buf = Vec::new();
    let mut parsed_bytes = 0u64;
    while start < mmap.len() {
        let slice = &mmap[start..];
        let rel = memchr(b'\n', slice).unwrap_or(slice.len());
        let line = &slice[..rel];
        let advanced = rel + 1;
        start += advanced;
        parsed_bytes += advanced as u64;
        if parsed_bytes >= 64 * 1024 {
            progress.add_parsed_bytes(SourceKind::CodexHistory, parsed_bytes);
            parsed_bytes = 0;
        }
        if line.is_empty() {
            continue;
        }
        buf.clear();
        buf.extend_from_slice(line);
        let value: BorrowedValue = match simd_json::to_borrowed_value(&mut buf) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let obj = match value.as_object() {
            Some(o) => o,
            None => continue,
        };
        let session_id = obj.get("session_id").and_then(|v| v.as_str()).unwrap_or("");
        if session_id.is_empty() || session_ids.contains(session_id) {
            continue;
        }
        let ts = obj.get("ts").and_then(|v| v.as_i64()).unwrap_or(0);
        let ts_ms = (ts.max(0) as u64) * 1000;
        let text = obj
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if text.is_empty() {
            continue;
        }
        let record = Record {
            source: SourceKind::CodexHistory,
            doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
            ts: ts_ms,
            project: "codex".to_string(),
            session_id: session_id.to_string(),
            turn_id,
            role: "user".to_string(),
            text,
            tool_name: None,
            tool_input: None,
            tool_output: None,
            links: RecordLinks {
                conversation_kind: Some("main".to_string()),
                ..RecordLinks::default()
            },
            source_path: source_path.clone(),
        };
        progress.add_produced(SourceKind::CodexHistory, 1);
        tx_record.send(record)?;
        turn_id += 1;
    }

    if parsed_bytes > 0 {
        progress.add_parsed_bytes(SourceKind::CodexHistory, parsed_bytes);
    }
    progress.add_files_done(SourceKind::CodexHistory, 1);
    let state = FileState {
        size: task.size,
        mtime: task.mtime,
        offset: mmap.len() as u64,
        turn_id,
        source_state: None,
    };
    tx_update.send(FileUpdate {
        path: source_path,
        state,
        session_id: None,
    })?;
    Ok(())
}

fn parse_opencode_file(
    task: &FileTask,
    tx_record: &RecordSender,
    tx_update: &Sender<FileUpdate>,
    next_doc_id: &AtomicU64,
    progress: &Arc<Progress>,
    opencode_session_links: &HashMap<String, SessionLinks>,
) -> Result<()> {
    let session_dir = &task.path;
    let session_id = session_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();
    let project = "opencode".to_string();
    let session_links = opencode_session_links
        .get(&session_id)
        .cloned()
        .unwrap_or_else(opencode_default_session_links);

    let mut messages = Vec::new();
    for entry in std::fs::read_dir(session_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let file = File::open(&path)?;
        let reader = std::io::BufReader::new(file);
        let msg: serde_json::Value = match serde_json::from_reader(reader) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let msg_id = msg.get("id").and_then(|v| v.as_str()).unwrap_or_default();
        if msg_id.is_empty() {
            continue;
        }
        let timestamp = msg
            .get("time")
            .and_then(|t| t.get("created"))
            .and_then(|c| c.as_u64())
            .unwrap_or(0);
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");

        messages.push((msg_id.to_string(), timestamp, role.to_string()));
    }

    messages.sort_by_key(|k| k.1);

    let parts_root = opencode_parts_root();
    let mut turn_id = task.turn_id;

    for (msg_id, timestamp, role) in messages {
        let part_dir = parts_root.join(&msg_id);
        if !part_dir.exists() {
            continue;
        }

        let mut part_files: Vec<_> = std::fs::read_dir(&part_dir)?
            .flatten()
            .map(|e| e.path())
            .collect();
        // Ensure deterministic order for message parts
        part_files.sort();

        let mut text_parts = Vec::new();
        for path in part_files {
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let file = File::open(&path)?;
            let reader = std::io::BufReader::new(file);
            let part: serde_json::Value = match serde_json::from_reader(reader) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                text_parts.push(text.to_string());
            }
        }

        if text_parts.is_empty() {
            continue;
        }

        let text = text_parts.join("\n");
        let mut links = session_links.record_links();
        links.event_id = Some(msg_id.clone());
        let record = Record {
            source: SourceKind::Opencode,
            doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
            ts: timestamp,
            project: project.clone(),
            session_id: session_id.clone(),
            turn_id,
            role: role.clone(),
            text,
            tool_name: None,
            tool_input: None,
            tool_output: None,
            links,
            source_path: session_dir.to_string_lossy().to_string(),
        };
        progress.add_produced(SourceKind::Opencode, 1);
        tx_record.send(record)?;
        turn_id += 1;
    }

    progress.add_files_done(SourceKind::Opencode, 1);
    let state = FileState {
        size: task.size,
        mtime: task.mtime,
        offset: 0,
        turn_id,
        source_state: None,
    };
    tx_update.send(FileUpdate {
        path: session_dir.to_string_lossy().to_string(),
        state,
        session_id: Some(session_id),
    })?;
    Ok(())
}

fn parse_cursor_file(
    task: &FileTask,
    tx_record: &RecordSender,
    tx_update: &Sender<FileUpdate>,
    next_doc_id: &AtomicU64,
    progress: &Arc<Progress>,
) -> Result<()> {
    let file = File::open(&task.path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut start = task.offset as usize;
    let mut turn_id = cursor_initial_turn_id(&task.path, task.turn_id);

    let source_path = task.path.to_string_lossy().to_string();
    let session_id = cursor_session_id_from_path(&task.path);
    let project = project_from_cursor_path(&task.path);
    let timestamp = task.mtime.max(0) as u64 * 1000;

    let mut buf = Vec::new();
    let mut parsed_bytes = 0u64;
    while start < mmap.len() {
        let slice = &mmap[start..];
        let rel = memchr(b'\n', slice).unwrap_or(slice.len());
        let line = &slice[..rel];
        let advanced = rel + 1;
        start += advanced;
        parsed_bytes += advanced as u64;
        if parsed_bytes >= 64 * 1024 {
            progress.add_parsed_bytes(SourceKind::Cursor, parsed_bytes);
            parsed_bytes = 0;
        }
        if line.is_empty() {
            continue;
        }
        buf.clear();
        buf.extend_from_slice(line);
        let value: BorrowedValue = match simd_json::to_borrowed_value(&mut buf) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let obj = match value.as_object() {
            Some(o) => o,
            None => continue,
        };
        let role = obj.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if role != "user" && role != "assistant" {
            continue;
        }
        let message = match obj.get("message").and_then(|v| v.as_object()) {
            Some(m) => m,
            None => continue,
        };
        let mut text_parts = Vec::new();
        if let Some(content) = message.get("content") {
            if let Some(text) = content.as_str() {
                text_parts.push(text);
            } else if let Some(arr) = content.as_array() {
                for block in arr {
                    let Some(block_obj) = block.as_object() else {
                        continue;
                    };
                    let block_type = block_obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if block_type == "text" {
                        if let Some(text) = block_obj.get("text").and_then(|v| v.as_str()) {
                            text_parts.push(text);
                        }
                    } else if block_type == "tool_use" {
                        let tool_name = block_obj
                            .get("name")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let tool_input = block_obj.get("input").map(|v| v.to_string());
                        let text = tool_input.clone().unwrap_or_default();
                        let record = Record {
                            source: SourceKind::Cursor,
                            doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                            ts: timestamp,
                            project: project.clone(),
                            session_id: session_id.clone(),
                            turn_id,
                            role: "tool_use".to_string(),
                            text,
                            tool_name,
                            tool_input,
                            tool_output: None,
                            links: cursor_record_links(&task.path, &session_id, turn_id),
                            source_path: source_path.clone(),
                        };
                        progress.add_produced(SourceKind::Cursor, 1);
                        tx_record.send(record)?;
                        turn_id += 1;
                    } else if block_type == "tool_result" {
                        let tool_output = block_obj.get("content").map(|v| v.to_string());
                        let mut text = extract_text_from_tool_result(block).unwrap_or_default();
                        if text.is_empty()
                            && let Some(content) = block_obj.get("content")
                        {
                            text = content.to_string();
                        }
                        if text.is_empty() {
                            continue;
                        }
                        let record = Record {
                            source: SourceKind::Cursor,
                            doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                            ts: timestamp,
                            project: project.clone(),
                            session_id: session_id.clone(),
                            turn_id,
                            role: "tool_result".to_string(),
                            text,
                            tool_name: None,
                            tool_input: None,
                            tool_output,
                            links: cursor_record_links(&task.path, &session_id, turn_id),
                            source_path: source_path.clone(),
                        };
                        progress.add_produced(SourceKind::Cursor, 1);
                        tx_record.send(record)?;
                        turn_id += 1;
                    }
                }
            }
        }

        let text = text_parts.join("\n").trim().to_string();
        if !text.is_empty() {
            let record = Record {
                source: SourceKind::Cursor,
                doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                ts: timestamp,
                project: project.clone(),
                session_id: session_id.clone(),
                turn_id,
                role: role.to_string(),
                text,
                tool_name: None,
                tool_input: None,
                tool_output: None,
                links: cursor_record_links(&task.path, &session_id, turn_id),
                source_path: source_path.clone(),
            };
            progress.add_produced(SourceKind::Cursor, 1);
            tx_record.send(record)?;
            turn_id += 1;
        }
    }

    if parsed_bytes > 0 {
        progress.add_parsed_bytes(SourceKind::Cursor, parsed_bytes);
    }
    progress.add_files_done(SourceKind::Cursor, 1);
    let state = FileState {
        size: task.size,
        mtime: task.mtime,
        offset: mmap.len() as u64,
        turn_id,
        source_state: None,
    };
    tx_update.send(FileUpdate {
        path: source_path,
        state,
        session_id: Some(session_id),
    })?;
    Ok(())
}

fn parse_pi_file(
    task: &FileTask,
    tx_record: &RecordSender,
    tx_update: &Sender<FileUpdate>,
    next_doc_id: &AtomicU64,
    progress: &Arc<Progress>,
) -> Result<()> {
    let file = File::open(&task.path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut start = task.offset as usize;
    let mut turn_id = task.turn_id;

    let source_path = task.path.to_string_lossy().to_string();
    let mut session_id =
        session_id_from_filename(&task.path).unwrap_or_else(|| source_path.clone());
    let mut project = project_from_pi_session_path(&task.path);
    let mut tool_id_to_name: HashMap<String, String> = HashMap::new();

    let mut buf = Vec::new();
    if start > 0 && !mmap.is_empty() {
        let rel = memchr(b'\n', &mmap).unwrap_or(mmap.len());
        let line = &mmap[..rel];
        if !line.is_empty() {
            buf.extend_from_slice(line);
            if let Ok(value) = simd_json::to_borrowed_value(&mut buf)
                && let Some(obj) = value.as_object()
                && obj.get("type").and_then(|v| v.as_str()) == Some("session")
            {
                apply_pi_session_header(obj, &mut session_id, &mut project);
            }
            buf.clear();
        }
    }
    let mut parsed_bytes = 0u64;
    while start < mmap.len() {
        let slice = &mmap[start..];
        let rel = memchr(b'\n', slice).unwrap_or(slice.len());
        let line = &slice[..rel];
        let advanced = rel + 1;
        start += advanced;
        parsed_bytes += advanced as u64;
        if parsed_bytes >= 64 * 1024 {
            progress.add_parsed_bytes(SourceKind::Pi, parsed_bytes);
            parsed_bytes = 0;
        }
        if line.is_empty() {
            continue;
        }
        buf.clear();
        buf.extend_from_slice(line);
        let value: BorrowedValue = match simd_json::to_borrowed_value(&mut buf) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let obj = match value.as_object() {
            Some(o) => o,
            None => continue,
        };
        let entry_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let timestamp = obj
            .get("timestamp")
            .and_then(|v| v.as_str())
            .and_then(parse_iso_millis)
            .unwrap_or(0);
        let conversation_kind = match entry_type {
            "branch_summary" => "branch",
            "compaction" => "compaction",
            _ => "main",
        };
        let mut base_links = pi_base_links(obj, conversation_kind);

        if entry_type == "session" {
            apply_pi_session_header(obj, &mut session_id, &mut project);
            continue;
        }

        if entry_type == "compaction" || entry_type == "branch_summary" {
            let summary = obj
                .get("summary")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            if summary.is_empty() {
                continue;
            }
            let record = Record {
                source: SourceKind::Pi,
                doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                ts: timestamp,
                project: project.clone(),
                session_id: session_id.clone(),
                turn_id,
                role: "assistant".to_string(),
                text: format!("{entry_type}: {summary}"),
                tool_name: None,
                tool_input: None,
                tool_output: None,
                links: base_links,
                source_path: source_path.clone(),
            };
            progress.add_produced(SourceKind::Pi, 1);
            tx_record.send(record)?;
            turn_id += 1;
            continue;
        }

        if entry_type == "custom_message" {
            let text = pi_content_text(obj.get("content")).trim().to_string();
            if text.is_empty() {
                continue;
            }
            let custom_type = obj.get("customType").and_then(|v| v.as_str()).unwrap_or("");
            let prefix = if custom_type.is_empty() {
                "custom_message".to_string()
            } else {
                format!("custom_message({custom_type})")
            };
            let record = Record {
                source: SourceKind::Pi,
                doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                ts: timestamp,
                project: project.clone(),
                session_id: session_id.clone(),
                turn_id,
                role: "assistant".to_string(),
                text: format!("{prefix}: {text}"),
                tool_name: None,
                tool_input: None,
                tool_output: None,
                links: base_links,
                source_path: source_path.clone(),
            };
            progress.add_produced(SourceKind::Pi, 1);
            tx_record.send(record)?;
            turn_id += 1;
            continue;
        }

        if entry_type != "message" {
            continue;
        }
        let message = match obj.get("message").and_then(|v| v.as_object()) {
            Some(m) => m,
            None => continue,
        };
        let timestamp = if timestamp == 0 {
            message
                .get("timestamp")
                .and_then(|v| v.as_str())
                .and_then(parse_iso_millis)
                .unwrap_or(0)
        } else {
            timestamp
        };
        let role = message.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if conversation_kind == "main" {
            match role {
                "branchSummary" => {
                    base_links.thread_source = Some("branch".to_string());
                    base_links.conversation_kind = Some("branch".to_string());
                }
                "compactionSummary" => {
                    base_links.thread_source = Some("compaction".to_string());
                    base_links.conversation_kind = Some("compaction".to_string());
                }
                _ => {}
            }
        }

        match role {
            "user" | "assistant" => {
                let content = message.get("content");
                if role == "assistant"
                    && let Some(arr) = content.and_then(|v| v.as_array())
                {
                    for block in arr {
                        let Some(block_obj) = block.as_object() else {
                            continue;
                        };
                        if block_obj.get("type").and_then(|v| v.as_str()) != Some("toolCall") {
                            continue;
                        }
                        let tool_name = block_obj
                            .get("name")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        if let (Some(id), Some(name)) = (
                            block_obj.get("id").and_then(|v| v.as_str()),
                            tool_name.clone(),
                        ) {
                            tool_id_to_name.insert(id.to_string(), name);
                        }
                        let tool_input = block_obj.get("arguments").map(|v| v.to_string());
                        let mut links = base_links.clone();
                        if let Some(tool_call_id) = block_obj.get("id").and_then(|v| v.as_str()) {
                            links.event_id = Some(tool_call_id.to_string());
                            links.parent_event_id = base_links.event_id.clone();
                        }
                        let record = Record {
                            source: SourceKind::Pi,
                            doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                            ts: timestamp,
                            project: project.clone(),
                            session_id: session_id.clone(),
                            turn_id,
                            role: "tool_use".to_string(),
                            text: tool_input.clone().unwrap_or_default(),
                            tool_name,
                            tool_input,
                            tool_output: None,
                            links,
                            source_path: source_path.clone(),
                        };
                        progress.add_produced(SourceKind::Pi, 1);
                        tx_record.send(record)?;
                        turn_id += 1;
                    }
                }

                let text = pi_content_text(content).trim().to_string();
                if text.is_empty() {
                    continue;
                }
                let record = Record {
                    source: SourceKind::Pi,
                    doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                    ts: timestamp,
                    project: project.clone(),
                    session_id: session_id.clone(),
                    turn_id,
                    role: role.to_string(),
                    text,
                    tool_name: None,
                    tool_input: None,
                    tool_output: None,
                    links: base_links,
                    source_path: source_path.clone(),
                };
                progress.add_produced(SourceKind::Pi, 1);
                tx_record.send(record)?;
                turn_id += 1;
            }
            "toolResult" => {
                let tool_call_id = message
                    .get("toolCallId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let tool_name = message
                    .get("toolName")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| tool_id_to_name.get(tool_call_id).cloned());
                let tool_output = Some(pi_content_text(message.get("content")));
                let text = tool_output.clone().unwrap_or_default();
                if text.trim().is_empty() {
                    continue;
                }
                let mut links = base_links;
                if !tool_call_id.is_empty() {
                    links.parent_tool_use_id = Some(tool_call_id.to_string());
                }
                let record = Record {
                    source: SourceKind::Pi,
                    doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                    ts: timestamp,
                    project: project.clone(),
                    session_id: session_id.clone(),
                    turn_id,
                    role: "tool_result".to_string(),
                    text,
                    tool_name,
                    tool_input: None,
                    tool_output,
                    links,
                    source_path: source_path.clone(),
                };
                progress.add_produced(SourceKind::Pi, 1);
                tx_record.send(record)?;
                turn_id += 1;
            }
            "bashExecution" => {
                if message
                    .get("excludeFromContext")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    continue;
                }
                let command = message
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let output = message
                    .get("output")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let exit_code = message.get("exitCode").and_then(|v| v.as_i64());
                let text = pi_bash_text(&command, &output, exit_code);
                if text.trim().is_empty() {
                    continue;
                }
                let record = Record {
                    source: SourceKind::Pi,
                    doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                    ts: timestamp,
                    project: project.clone(),
                    session_id: session_id.clone(),
                    turn_id,
                    role: "tool_result".to_string(),
                    text,
                    tool_name: Some("Bash".to_string()),
                    tool_input: if command.is_empty() {
                        None
                    } else {
                        Some(command)
                    },
                    tool_output: if output.is_empty() {
                        None
                    } else {
                        Some(output)
                    },
                    links: base_links,
                    source_path: source_path.clone(),
                };
                progress.add_produced(SourceKind::Pi, 1);
                tx_record.send(record)?;
                turn_id += 1;
            }
            "custom" | "branchSummary" | "compactionSummary" => {
                let text = pi_summary_message_text(message, role);
                if text.is_empty() {
                    continue;
                }
                let record = Record {
                    source: SourceKind::Pi,
                    doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                    ts: timestamp,
                    project: project.clone(),
                    session_id: session_id.clone(),
                    turn_id,
                    role: "assistant".to_string(),
                    text: format!("{role}: {text}"),
                    tool_name: None,
                    tool_input: None,
                    tool_output: None,
                    links: base_links,
                    source_path: source_path.clone(),
                };
                progress.add_produced(SourceKind::Pi, 1);
                tx_record.send(record)?;
                turn_id += 1;
            }
            _ => {}
        }
    }

    if parsed_bytes > 0 {
        progress.add_parsed_bytes(SourceKind::Pi, parsed_bytes);
    }
    progress.add_files_done(SourceKind::Pi, 1);
    let state = FileState {
        size: task.size,
        mtime: task.mtime,
        offset: mmap.len() as u64,
        turn_id,
        source_state: None,
    };
    tx_update.send(FileUpdate {
        path: source_path,
        state,
        session_id: Some(session_id),
    })?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OmpSessionVersion {
    V1,
    V2,
    V3,
}

impl OmpSessionVersion {
    fn has_entry_tree(self) -> bool {
        matches!(self, Self::V2 | Self::V3)
    }

    fn as_u8(self) -> u8 {
        match self {
            Self::V1 => 1,
            Self::V2 => 2,
            Self::V3 => 3,
        }
    }
}

#[derive(Debug)]
struct OmpSessionInfo {
    version: OmpSessionVersion,
    session_id: String,
    project: String,
    parent_session_id: Option<String>,
    thread_source: Option<String>,
    conversation_kind: String,
    title_slot_bytes: usize,
}

impl OmpSessionInfo {
    fn source_file_state(&self, mmap: &[u8], offset: u64) -> SourceFileState {
        SourceFileState::Omp {
            version: self.version.as_u8(),
            title_slot_bytes: u32::try_from(self.title_slot_bytes).unwrap_or(u32::MAX),
            append_checksum: omp_append_checksum(
                mmap,
                offset,
                self.title_slot_bytes.min(mmap.len()),
            ),
        }
    }
}

struct OmpRecordSink<'a> {
    tx_record: &'a RecordSender,
    next_doc_id: &'a AtomicU64,
    progress: &'a Progress,
    source_path: &'a str,
    session: &'a OmpSessionInfo,
    turn_id: u32,
}

struct OmpEmission<'a> {
    ts: u64,
    role: &'a str,
    text: String,
    tool_name: Option<String>,
    tool_input: Option<String>,
    tool_output: Option<String>,
    links: RecordLinks,
}

impl OmpRecordSink<'_> {
    fn emit(&mut self, emission: OmpEmission<'_>) -> Result<()> {
        let record = Record {
            source: SourceKind::Omp,
            doc_id: self.next_doc_id.fetch_add(1, Ordering::SeqCst),
            ts: emission.ts,
            project: self.session.project.clone(),
            session_id: self.session.session_id.clone(),
            turn_id: self.turn_id,
            role: emission.role.to_string(),
            text: emission.text,
            tool_name: emission.tool_name,
            tool_input: emission.tool_input,
            tool_output: emission.tool_output,
            links: emission.links,
            source_path: self.source_path.to_string(),
        };
        self.progress.add_produced(SourceKind::Omp, 1);
        self.tx_record.send(record)?;
        self.turn_id += 1;
        Ok(())
    }
}

fn parse_omp_file(
    task: &FileTask,
    tx_record: &RecordSender,
    tx_update: &Sender<FileUpdate>,
    next_doc_id: &AtomicU64,
    progress: &Arc<Progress>,
) -> Result<()> {
    let file = File::open(&task.path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let session = read_omp_session_header(&mmap, &task.path)?;
    let source_path = task.path.to_string_lossy().to_string();
    let mut sink = OmpRecordSink {
        tx_record,
        next_doc_id,
        progress,
        source_path: &source_path,
        session: &session,
        turn_id: task.turn_id,
    };
    let mut start = task.offset as usize;
    let mut parsed_bytes = 0u64;
    let mut buf = Vec::new();
    let mut tool_id_to_name = HashMap::new();

    while start < mmap.len() {
        let slice = &mmap[start..];
        let relative_end = memchr(b'\n', slice).unwrap_or(slice.len());
        let line = &slice[..relative_end];
        let advanced = relative_end + usize::from(relative_end < slice.len());
        start += advanced;
        parsed_bytes += advanced as u64;
        if parsed_bytes >= 64 * 1024 {
            progress.add_parsed_bytes(SourceKind::Omp, parsed_bytes);
            parsed_bytes = 0;
        }
        if line.is_empty() {
            continue;
        }

        buf.clear();
        buf.extend_from_slice(line);
        let value: BorrowedValue = match simd_json::to_borrowed_value(&mut buf) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let Some(obj) = value.as_object() else {
            continue;
        };
        let entry_type = obj
            .get("type")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        if matches!(entry_type, "title" | "session") {
            continue;
        }

        let timestamp = obj
            .get("timestamp")
            .and_then(omp_timestamp_millis)
            .unwrap_or(0);
        let entry_kind = match entry_type {
            "branch_summary" => "branch",
            "compaction" => "compaction",
            _ => session.conversation_kind.as_str(),
        };
        let base_links = omp_record_links(obj, &session, entry_kind);

        if matches!(entry_type, "compaction" | "branch_summary") {
            let summary = obj
                .get("summary")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .trim();
            if !summary.is_empty() {
                sink.emit(OmpEmission {
                    ts: timestamp,
                    role: "assistant",
                    text: format!("{entry_type}: {summary}"),
                    tool_name: None,
                    tool_input: None,
                    tool_output: None,
                    links: base_links,
                })?;
            }
            continue;
        }

        if entry_type == "custom_message" {
            let text = omp_content_text(obj.get("content")).trim().to_string();
            if text.is_empty() {
                continue;
            }
            let custom_type = obj
                .get("customType")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let prefix = if custom_type.is_empty() {
                "custom_message".to_string()
            } else {
                format!("custom_message({custom_type})")
            };
            let role = if obj.get("attribution").and_then(|value| value.as_str()) == Some("user") {
                "user"
            } else {
                "assistant"
            };
            sink.emit(OmpEmission {
                ts: timestamp,
                role,
                text: format!("{prefix}: {text}"),
                tool_name: None,
                tool_input: None,
                tool_output: None,
                links: base_links,
            })?;
            continue;
        }

        if entry_type != "message" {
            continue;
        }
        let Some(message) = obj.get("message").and_then(|value| value.as_object()) else {
            continue;
        };
        let timestamp = if timestamp == 0 {
            message
                .get("timestamp")
                .and_then(omp_timestamp_millis)
                .unwrap_or(0)
        } else {
            timestamp
        };
        let role = message
            .get("role")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        let mut message_links = base_links;
        match role {
            "branchSummary" => {
                message_links.thread_source = Some("branch".to_string());
                message_links.conversation_kind = Some("branch".to_string());
            }
            "compactionSummary" => {
                message_links.thread_source = Some("compaction".to_string());
                message_links.conversation_kind = Some("compaction".to_string());
            }
            _ => {}
        }

        match role {
            "user" | "assistant" => {
                let content = message.get("content");
                if role == "assistant"
                    && let Some(blocks) = content.and_then(|value| value.as_array())
                {
                    for block in blocks {
                        let Some(block) = block.as_object() else {
                            continue;
                        };
                        if block.get("type").and_then(|value| value.as_str()) != Some("toolCall") {
                            continue;
                        }
                        let tool_name = block
                            .get("name")
                            .and_then(|value| value.as_str())
                            .map(str::to_string);
                        if let (Some(tool_call_id), Some(tool_name)) = (
                            block.get("id").and_then(|value| value.as_str()),
                            tool_name.as_ref(),
                        ) {
                            tool_id_to_name.insert(tool_call_id.to_string(), tool_name.clone());
                        }
                        let tool_input = block.get("arguments").map(ToString::to_string);
                        let mut links = message_links.clone();
                        if let Some(tool_call_id) = block.get("id").and_then(|value| value.as_str())
                        {
                            links.event_id = Some(tool_call_id.to_string());
                            links.parent_event_id = message_links.event_id.clone();
                        }
                        sink.emit(OmpEmission {
                            ts: timestamp,
                            role: "tool_use",
                            text: tool_input.clone().unwrap_or_default(),
                            tool_name,
                            tool_input,
                            tool_output: None,
                            links,
                        })?;
                    }
                }

                let text = omp_content_text(content).trim().to_string();
                if !text.is_empty() {
                    sink.emit(OmpEmission {
                        ts: timestamp,
                        role,
                        text,
                        tool_name: None,
                        tool_input: None,
                        tool_output: None,
                        links: message_links,
                    })?;
                }
            }
            "toolResult" => {
                let tool_call_id = message
                    .get("toolCallId")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                let tool_name = message
                    .get("toolName")
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
                    .or_else(|| tool_id_to_name.get(tool_call_id).cloned());
                let tool_output = omp_content_text(message.get("content"));
                if tool_output.trim().is_empty() {
                    continue;
                }
                let mut links = message_links;
                if !tool_call_id.is_empty() {
                    links.parent_tool_use_id = Some(tool_call_id.to_string());
                }
                sink.emit(OmpEmission {
                    ts: timestamp,
                    role: "tool_result",
                    text: tool_output.clone(),
                    tool_name,
                    tool_input: None,
                    tool_output: Some(tool_output),
                    links,
                })?;
            }
            "bashExecution" => {
                if message
                    .get("excludeFromContext")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false)
                {
                    continue;
                }
                let command = message
                    .get("command")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_string();
                let output = message
                    .get("output")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_string();
                let text = pi_bash_text(
                    &command,
                    &output,
                    message.get("exitCode").and_then(|value| value.as_i64()),
                );
                if !text.trim().is_empty() {
                    sink.emit(OmpEmission {
                        ts: timestamp,
                        role: "tool_result",
                        text,
                        tool_name: Some("Bash".to_string()),
                        tool_input: (!command.is_empty()).then_some(command),
                        tool_output: (!output.is_empty()).then_some(output),
                        links: message_links,
                    })?;
                }
            }
            "custom" | "hookMessage" | "branchSummary" | "compactionSummary" => {
                let text = pi_summary_message_text(message, role);
                if !text.is_empty() {
                    sink.emit(OmpEmission {
                        ts: timestamp,
                        role: "assistant",
                        text: format!("{role}: {text}"),
                        tool_name: None,
                        tool_input: None,
                        tool_output: None,
                        links: message_links,
                    })?;
                }
            }
            _ => {}
        }
    }

    if parsed_bytes > 0 {
        progress.add_parsed_bytes(SourceKind::Omp, parsed_bytes);
    }
    progress.add_files_done(SourceKind::Omp, 1);
    let state = FileState {
        size: task.size,
        mtime: task.mtime,
        offset: mmap.len() as u64,
        turn_id: sink.turn_id,
        source_state: Some(session.source_file_state(&mmap, mmap.len() as u64)),
    };
    tx_update.send(FileUpdate {
        path: source_path,
        state,
        session_id: Some(session.session_id),
    })?;
    Ok(())
}

fn read_omp_session_header(mmap: &[u8], path: &Path) -> Result<OmpSessionInfo> {
    let mut start = 0usize;
    let mut buf = Vec::new();
    let mut title_slot_bytes = 0usize;
    for line_number in 1..=2 {
        if start >= mmap.len() {
            break;
        }
        let slice = &mmap[start..];
        let relative_end = memchr(b'\n', slice).unwrap_or(slice.len());
        let line = &slice[..relative_end];
        start += relative_end + usize::from(relative_end < slice.len());
        if line.is_empty() {
            continue;
        }
        buf.clear();
        buf.extend_from_slice(line);
        let value: BorrowedValue = simd_json::to_borrowed_value(&mut buf).map_err(|error| {
            anyhow!(
                "invalid OMP transcript header at {}:{line_number}: {error}",
                path.display()
            )
        })?;
        let obj = value.as_object().ok_or_else(|| {
            anyhow!(
                "invalid OMP transcript header at {}:{line_number}: expected object",
                path.display()
            )
        })?;
        let entry_type = obj.get("type").and_then(|value| value.as_str());
        if entry_type == Some("title") && line_number == 1 {
            title_slot_bytes = start;
            continue;
        }
        if entry_type != Some("session") {
            return Err(anyhow!(
                "invalid OMP transcript header at {}:{line_number}: expected session entry",
                path.display()
            ));
        }

        let version = match obj.get("version") {
            None => OmpSessionVersion::V1,
            Some(value) => match value.as_u64() {
                Some(1) => OmpSessionVersion::V1,
                Some(2) => OmpSessionVersion::V2,
                Some(3) => OmpSessionVersion::V3,
                Some(version) => {
                    return Err(anyhow!(
                        "unsupported OMP session version {version} in {}; memex supports versions 1, 2, and 3",
                        path.display()
                    ));
                }
                None => {
                    return Err(anyhow!(
                        "invalid OMP session version in {}: expected integer",
                        path.display()
                    ));
                }
            },
        };
        let session_id = obj
            .get("id")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("OMP session header missing id in {}", path.display()))?
            .to_string();
        let cwd = obj
            .get("cwd")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("OMP session header missing cwd in {}", path.display()))?;
        let explicit_parent = obj
            .get("parentSession")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let nested_parent = omp_nested_parent_session_id(path);
        let (parent_session_id, thread_source, conversation_kind) =
            if let Some(parent) = nested_parent {
                (Some(parent), Some("subagent".to_string()), "subagent")
            } else if let Some(parent) = explicit_parent {
                (Some(parent), Some("fork".to_string()), "fork")
            } else {
                (None, None, "main")
            };
        return Ok(OmpSessionInfo {
            version,
            session_id,
            project: project_from_path(cwd),
            parent_session_id,
            thread_source,
            conversation_kind: conversation_kind.to_string(),
            title_slot_bytes,
        });
    }
    Err(anyhow!(
        "OMP transcript {} does not contain a session header in its first two lines",
        path.display()
    ))
}

fn source_file_requires_full_reindex(
    source: SourceKind,
    path: &Path,
    previous: &FileState,
) -> Result<bool> {
    if source != SourceKind::Omp {
        return Ok(false);
    }
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let session = read_omp_session_header(&mmap, path)?;
    let current = session.source_file_state(&mmap, previous.offset);
    Ok(previous.source_state.as_ref() != Some(&current))
}

fn omp_append_checksum(mmap: &[u8], offset: u64, mutable_prefix_bytes: usize) -> u64 {
    const WINDOW_BYTES: usize = 64 * 1024;
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let end = usize::try_from(offset)
        .unwrap_or(usize::MAX)
        .min(mmap.len());
    let start = end
        .saturating_sub(WINDOW_BYTES)
        .max(mutable_prefix_bytes.min(end));
    mmap[start..end]
        .iter()
        .fold(FNV_OFFSET_BASIS, |checksum, byte| {
            (checksum ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
        })
}

fn omp_nested_parent_session_id(path: &Path) -> Option<String> {
    let parent_dir = path.parent()?;
    if !parent_dir.with_extension("jsonl").is_file() {
        return None;
    }
    session_id_from_filename(parent_dir)
}

fn omp_record_links(
    obj: &simd_json::borrowed::Object,
    session: &OmpSessionInfo,
    conversation_kind: &str,
) -> RecordLinks {
    let has_entry_tree = session.version.has_entry_tree();
    RecordLinks {
        event_id: has_entry_tree.then(|| opt_str(obj, "id")).flatten(),
        parent_event_id: has_entry_tree.then(|| opt_str(obj, "parentId")).flatten(),
        logical_parent_event_id: has_entry_tree.then(|| opt_str(obj, "fromId")).flatten(),
        parent_session_id: session.parent_session_id.clone(),
        thread_source: if conversation_kind == session.conversation_kind {
            session.thread_source.clone()
        } else {
            Some(conversation_kind.to_string())
        },
        conversation_kind: Some(conversation_kind.to_string()),
        ..RecordLinks::default()
    }
}

fn omp_timestamp_millis(value: &BorrowedValue) -> Option<u64> {
    if let Some(value) = value.as_str() {
        return parse_iso_millis(value).or_else(|| value.parse::<u64>().ok());
    }
    value.as_u64().map(|value| {
        if value < 10_000_000_000 {
            value * 1000
        } else {
            value
        }
    })
}

fn omp_content_text(content: Option<&BorrowedValue>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    if let Some(text) = content.as_str() {
        return text.to_string();
    }
    let Some(blocks) = content.as_array() else {
        return String::new();
    };
    let mut parts = Vec::new();
    for block in blocks {
        if let Some(text) = block.as_str() {
            parts.push(text.to_string());
            continue;
        }
        let Some(block) = block.as_object() else {
            continue;
        };
        let block_type = block
            .get("type")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        if matches!(block_type, "thinking" | "toolCall" | "image") {
            continue;
        }
        if let Some(text) = block
            .get("text")
            .or_else(|| block.get("content"))
            .and_then(|value| value.as_str())
        {
            parts.push(text.to_string());
        }
    }
    parts.join("\n")
}

fn parse_iso_millis(input: &str) -> Option<u64> {
    DateTime::parse_from_rfc3339(input)
        .ok()
        .map(|dt| dt.with_timezone(&Utc).timestamp_millis() as u64)
}

fn project_from_claude_path(path: &Path) -> String {
    let Some(parent) = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
    else {
        return "unknown".to_string();
    };
    decode_project_name(parent)
}

fn project_from_path(path: &str) -> String {
    let p = Path::new(path);
    if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
        return name.to_string();
    }
    "codex".to_string()
}

fn apply_pi_session_header(
    obj: &simd_json::borrowed::Object,
    session_id: &mut String,
    project: &mut String,
) {
    if let Some(id) = obj.get("id").and_then(|v| v.as_str())
        && !id.is_empty()
    {
        *session_id = id.to_string();
    }
    if let Some(cwd) = obj.get("cwd").and_then(|v| v.as_str())
        && !cwd.is_empty()
    {
        *project = project_from_path(cwd);
    }
}

#[derive(Debug, Default)]
struct CopilotWorkspace {
    cwd: Option<String>,
    git_root: Option<String>,
    repository: Option<String>,
    branch: Option<String>,
}

fn parse_copilot_session(
    task: &FileTask,
    tx_record: &RecordSender,
    tx_update: &Sender<FileUpdate>,
    next_doc_id: &AtomicU64,
    progress: &Arc<Progress>,
) -> Result<()> {
    let file = File::open(&task.path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mut start = task.offset as usize;
    let mut turn_id = task.turn_id;

    let source_path = task.path.to_string_lossy().to_string();
    let mut session_id =
        session_id_from_copilot_path(&task.path).unwrap_or_else(|| "unknown".to_string());
    let mut workspace = read_copilot_workspace(&task.path);
    let mut project = copilot_project(&workspace);
    let mut call_id_to_name: HashMap<String, String> = HashMap::new();

    let mut parsed_bytes = 0u64;
    while start < mmap.len() {
        let slice = &mmap[start..];
        let rel = memchr(b'\n', slice).unwrap_or(slice.len());
        let line = &slice[..rel];
        let advanced = rel + 1;
        start += advanced;
        parsed_bytes += advanced as u64;
        if parsed_bytes >= 64 * 1024 {
            progress.add_parsed_bytes(SourceKind::Copilot, parsed_bytes);
            parsed_bytes = 0;
        }
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value = match serde_json::from_slice(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let Some(obj) = value.as_object() else {
            continue;
        };
        let entry_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let data = obj
            .get("data")
            .or_else(|| obj.get("payload"))
            .unwrap_or(&serde_json::Value::Null);
        let timestamp = obj
            .get("timestamp")
            .and_then(json_timestamp_millis)
            .or_else(|| data.get("timestamp").and_then(json_timestamp_millis))
            .unwrap_or(0);

        match entry_type {
            "session.start" | "session.resume" => {
                if let Some(id) = data
                    .get("sessionId")
                    .or_else(|| data.get("session_id"))
                    .and_then(|v| v.as_str())
                {
                    session_id = id.to_string();
                }
                merge_copilot_workspace(&mut workspace, data.get("context").unwrap_or(data));
                project = copilot_project(&workspace);
            }
            "session.context_changed" => {
                merge_copilot_workspace(&mut workspace, data);
                project = copilot_project(&workspace);
            }
            "user.message" => {
                let text = data
                    .get("content")
                    .or_else(|| data.get("message"))
                    .or_else(|| data.get("prompt"))
                    .and_then(text_from_json)
                    .unwrap_or_default();
                if text.trim().is_empty() {
                    continue;
                }
                let links = copilot_record_links(obj, data, &session_id, turn_id);
                let record = Record {
                    source: SourceKind::Copilot,
                    doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                    ts: timestamp,
                    project: project.clone(),
                    session_id: session_id.clone(),
                    turn_id,
                    role: "user".to_string(),
                    text,
                    tool_name: None,
                    tool_input: None,
                    tool_output: None,
                    links,
                    source_path: source_path.clone(),
                };
                progress.add_produced(SourceKind::Copilot, 1);
                tx_record.send(record)?;
                turn_id += 1;
            }
            "assistant.message" => {
                let text = data
                    .get("content")
                    .and_then(text_from_json)
                    .unwrap_or_default();
                if text.trim().is_empty() {
                    continue;
                }
                let links = copilot_record_links(obj, data, &session_id, turn_id);
                let record = Record {
                    source: SourceKind::Copilot,
                    doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                    ts: timestamp,
                    project: project.clone(),
                    session_id: session_id.clone(),
                    turn_id,
                    role: "assistant".to_string(),
                    text,
                    tool_name: None,
                    tool_input: None,
                    tool_output: None,
                    links,
                    source_path: source_path.clone(),
                };
                progress.add_produced(SourceKind::Copilot, 1);
                tx_record.send(record)?;
                turn_id += 1;
            }
            "tool.execution_start" | "tool.user_requested" => {
                let tool_name = data
                    .get("toolName")
                    .or_else(|| data.get("name"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                if let Some(call_id) = data.get("toolCallId").and_then(|v| v.as_str())
                    && let Some(name) = tool_name.clone()
                {
                    call_id_to_name.insert(call_id.to_string(), name);
                }
                let tool_input = data
                    .get("arguments")
                    .map(json_to_text)
                    .filter(|s| !s.is_empty());
                let text = tool_input.clone().unwrap_or_default();
                let mut links = copilot_record_links(obj, data, &session_id, turn_id);
                if let Some(call_id) = data.get("toolCallId").and_then(|v| v.as_str()) {
                    links.event_id = Some(call_id.to_string());
                }
                let record = Record {
                    source: SourceKind::Copilot,
                    doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                    ts: timestamp,
                    project: project.clone(),
                    session_id: session_id.clone(),
                    turn_id,
                    role: "tool_use".to_string(),
                    text,
                    tool_name,
                    tool_input,
                    tool_output: None,
                    links,
                    source_path: source_path.clone(),
                };
                progress.add_produced(SourceKind::Copilot, 1);
                tx_record.send(record)?;
                turn_id += 1;
            }
            "tool.execution_complete" => {
                let call_id = data
                    .get("toolCallId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let tool_name = data
                    .get("toolName")
                    .or_else(|| data.get("name"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| call_id_to_name.get(call_id).cloned());
                let tool_output = copilot_tool_output(data);
                let text = tool_output.clone().unwrap_or_default();
                if text.trim().is_empty() {
                    continue;
                }
                let mut links = copilot_record_links(obj, data, &session_id, turn_id);
                if !call_id.is_empty() {
                    links.parent_event_id = Some(call_id.to_string());
                    links.parent_tool_use_id = Some(call_id.to_string());
                }
                let record = Record {
                    source: SourceKind::Copilot,
                    doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                    ts: timestamp,
                    project: project.clone(),
                    session_id: session_id.clone(),
                    turn_id,
                    role: "tool_result".to_string(),
                    text,
                    tool_name,
                    tool_input: None,
                    tool_output,
                    links,
                    source_path: source_path.clone(),
                };
                progress.add_produced(SourceKind::Copilot, 1);
                tx_record.send(record)?;
                turn_id += 1;
            }
            "session.task_complete" => {
                let text = data
                    .get("summary")
                    .and_then(text_from_json)
                    .unwrap_or_default();
                if text.trim().is_empty() {
                    continue;
                }
                let links = copilot_record_links(obj, data, &session_id, turn_id);
                let record = Record {
                    source: SourceKind::Copilot,
                    doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                    ts: timestamp,
                    project: project.clone(),
                    session_id: session_id.clone(),
                    turn_id,
                    role: "assistant".to_string(),
                    text,
                    tool_name: None,
                    tool_input: None,
                    tool_output: None,
                    links,
                    source_path: source_path.clone(),
                };
                progress.add_produced(SourceKind::Copilot, 1);
                tx_record.send(record)?;
                turn_id += 1;
            }
            _ => {}
        }
    }

    if parsed_bytes > 0 {
        progress.add_parsed_bytes(SourceKind::Copilot, parsed_bytes);
    }
    progress.add_files_done(SourceKind::Copilot, 1);
    let state = FileState {
        size: task.size,
        mtime: task.mtime,
        offset: mmap.len() as u64,
        turn_id,
        source_state: None,
    };
    tx_update.send(FileUpdate {
        path: source_path,
        state,
        session_id: Some(session_id),
    })?;
    Ok(())
}

fn read_copilot_workspace(events_path: &Path) -> CopilotWorkspace {
    let Some(dir) = events_path.parent() else {
        return CopilotWorkspace::default();
    };
    let path = dir.join("workspace.yaml");
    let Ok(contents) = std::fs::read_to_string(path) else {
        return CopilotWorkspace::default();
    };
    parse_copilot_workspace_yaml(&contents)
}

fn parse_copilot_workspace_yaml(contents: &str) -> CopilotWorkspace {
    let mut workspace = CopilotWorkspace::default();
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || line.chars().next().is_some_and(|c| c.is_whitespace())
        {
            continue;
        }
        let Some((key, value)) = trimmed.split_once(':') else {
            continue;
        };
        let value = value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
        if value.is_empty() {
            continue;
        }
        match key.trim() {
            "cwd" => workspace.cwd = Some(value),
            "gitRoot" | "git_root" => workspace.git_root = Some(value),
            "repository" => workspace.repository = Some(value),
            "branch" => workspace.branch = Some(value),
            _ => {}
        }
    }
    workspace
}

fn merge_copilot_workspace(workspace: &mut CopilotWorkspace, value: &serde_json::Value) {
    if let Some(cwd) = value.get("cwd").and_then(|v| v.as_str()) {
        workspace.cwd = Some(cwd.to_string());
    }
    if let Some(git_root) = value
        .get("gitRoot")
        .or_else(|| value.get("git_root"))
        .and_then(|v| v.as_str())
    {
        workspace.git_root = Some(git_root.to_string());
    }
    if let Some(repository) = value.get("repository").and_then(|v| v.as_str()) {
        workspace.repository = Some(repository.to_string());
    }
    if let Some(branch) = value.get("branch").and_then(|v| v.as_str()) {
        workspace.branch = Some(branch.to_string());
    }
}

fn copilot_project(workspace: &CopilotWorkspace) -> String {
    if let Some(repo) = workspace.repository.as_deref() {
        if let Some((_, name)) = repo.rsplit_once('/')
            && !name.is_empty()
        {
            return name.to_string();
        }
        if !repo.is_empty() {
            return repo.to_string();
        }
    }
    if let Some(git_root) = workspace.git_root.as_deref() {
        return project_from_path(git_root);
    }
    if let Some(cwd) = workspace.cwd.as_deref() {
        return project_from_path(cwd);
    }
    "copilot".to_string()
}

fn session_id_from_copilot_path(path: &Path) -> Option<String> {
    path.parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

fn copilot_record_links(
    obj: &serde_json::Map<String, serde_json::Value>,
    data: &serde_json::Value,
    session_id: &str,
    turn_id: u32,
) -> RecordLinks {
    let parent_session_id = copilot_string_field(data, obj, COPILOT_PARENT_SESSION_KEYS);
    let explicit_thread_source =
        copilot_string_field(data, obj, &["threadSource", "thread_source", "source"]);
    let thread_source = explicit_thread_source.or_else(|| {
        parent_session_id
            .as_ref()
            .filter(|parent| parent.as_str() != session_id)
            .map(|_| "fork".to_string())
    });
    let conversation_kind = match thread_source.as_deref() {
        Some("subagent") => "subagent",
        Some("sidechain") => "sidechain",
        Some("branch") => "branch",
        Some("fork") => "fork",
        _ => "main",
    };

    RecordLinks {
        event_id: copilot_string_field(data, obj, COPILOT_EVENT_KEYS)
            .or_else(|| Some(format!("{session_id}:{turn_id}"))),
        parent_event_id: copilot_string_field(data, obj, COPILOT_PARENT_EVENT_KEYS),
        logical_parent_event_id: copilot_string_field(data, obj, COPILOT_LOGICAL_PARENT_KEYS),
        parent_session_id,
        thread_source,
        conversation_kind: Some(conversation_kind.to_string()),
        ..RecordLinks::default()
    }
}

const COPILOT_EVENT_KEYS: &[&str] = &[
    "id",
    "eventId",
    "event_id",
    "messageId",
    "message_id",
    "requestId",
    "request_id",
    "responseId",
    "response_id",
];
const COPILOT_PARENT_EVENT_KEYS: &[&str] = &[
    "parentId",
    "parent_id",
    "parentMessageId",
    "parent_message_id",
    "parentEventId",
    "parent_event_id",
];
const COPILOT_LOGICAL_PARENT_KEYS: &[&str] = &["fromId", "from_id", "rootId", "root_id"];
const COPILOT_PARENT_SESSION_KEYS: &[&str] = &[
    "parentSessionId",
    "parent_session_id",
    "forkedFromSessionId",
    "forked_from_session_id",
];

fn copilot_string_field(
    data: &serde_json::Value,
    obj: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Option<String> {
    for key in keys {
        if let Some(value) = data
            .get(*key)
            .or_else(|| obj.get(*key))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
        {
            return Some(value.to_string());
        }
    }
    None
}

fn json_timestamp_millis(value: &serde_json::Value) -> Option<u64> {
    if let Some(text) = value.as_str() {
        if let Some(ts) = parse_iso_millis(text) {
            return Some(ts);
        }
        return text.parse::<u64>().ok();
    }
    if let Some(n) = value.as_u64() {
        return Some(if n < 10_000_000_000 { n * 1000 } else { n });
    }
    None
}

fn text_from_json(value: &serde_json::Value) -> Option<String> {
    if let Some(text) = value.as_str() {
        return Some(text.trim().to_string());
    }
    if let Some(arr) = value.as_array() {
        let mut parts = Vec::new();
        for item in arr {
            if let Some(text) = item.as_str() {
                parts.push(text);
                continue;
            }
            if let Some(obj) = item.as_object()
                && let Some(text) = obj
                    .get("text")
                    .or_else(|| obj.get("content"))
                    .and_then(|v| v.as_str())
            {
                parts.push(text);
            }
        }
        if !parts.is_empty() {
            return Some(parts.join("\n").trim().to_string());
        }
    }
    None
}

fn json_to_text(value: &serde_json::Value) -> String {
    if let Some(text) = text_from_json(value) {
        return text;
    }
    serde_json::to_string(value).unwrap_or_default()
}

fn copilot_tool_output(data: &serde_json::Value) -> Option<String> {
    if let Some(result) = data.get("result") {
        for key in ["detailedContent", "content"] {
            if let Some(text) = result.get(key).and_then(text_from_json)
                && !text.trim().is_empty()
            {
                return Some(text);
            }
        }
        if let Some(contents) = result.get("contents").and_then(|v| v.as_array()) {
            let mut parts = Vec::new();
            for item in contents {
                if let Some(text) = text_from_json(item) {
                    parts.push(text);
                }
            }
            if !parts.is_empty() {
                return Some(parts.join("\n"));
            }
        }
    }
    if let Some(error) = data.get("error") {
        if let Some(message) = error.get("message").and_then(|v| v.as_str()) {
            return Some(message.to_string());
        }
        return Some(json_to_text(error));
    }
    None
}

fn project_from_cursor_path(path: &Path) -> String {
    let root = cursor_projects_root();
    let Ok(relative) = path.strip_prefix(root) else {
        return "cursor".to_string();
    };
    let Some(project_folder) = relative
        .components()
        .next()
        .and_then(|c| c.as_os_str().to_str())
    else {
        return "cursor".to_string();
    };
    if project_folder == "empty-window" {
        return project_folder.to_string();
    }
    decode_project_name(project_folder)
}

fn cursor_session_id_from_path(path: &Path) -> String {
    let components: Vec<_> = path.components().collect();
    for (idx, component) in components.iter().enumerate() {
        if component.as_os_str().to_str() == Some("agent-transcripts")
            && let Some(session_id) = components
                .get(idx + 1)
                .and_then(|c| Path::new(c.as_os_str()).file_stem())
                .and_then(|s| s.to_str())
                .filter(|s| !s.is_empty())
        {
            return session_id.to_string();
        }
    }

    session_id_from_filename(path).unwrap_or_else(|| {
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string()
    })
}

fn cursor_initial_turn_id(path: &Path, cached_turn_id: u32) -> u32 {
    if cached_turn_id != 0 {
        return cached_turn_id;
    }
    cursor_subagent_turn_base(path).unwrap_or(cached_turn_id)
}

fn cursor_subagent_turn_base(path: &Path) -> Option<u32> {
    if !path
        .components()
        .any(|component| component.as_os_str().to_str() == Some("subagents"))
    {
        return None;
    }

    let agent_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())?;
    let bucket = stable_cursor_turn_bucket(agent_id);
    Some(CURSOR_SUBAGENT_TURN_BASE + bucket.saturating_mul(CURSOR_SUBAGENT_TURN_STRIDE))
}

fn cursor_is_subagent_transcript(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str().to_str() == Some("subagents"))
}

fn cursor_transcript_id(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown")
        .to_string()
}

fn cursor_record_links(path: &Path, session_id: &str, turn_id: u32) -> RecordLinks {
    let is_subagent = cursor_is_subagent_transcript(path);
    RecordLinks {
        event_id: Some(format!("{}:{turn_id}", cursor_transcript_id(path))),
        parent_session_id: is_subagent.then(|| session_id.to_string()),
        thread_source: is_subagent.then(|| "subagent".to_string()),
        conversation_kind: Some(if is_subagent { "subagent" } else { "main" }.to_string()),
        ..RecordLinks::default()
    }
}

fn stable_cursor_turn_bucket(value: &str) -> u32 {
    let mut hash = 2_166_136_261u32;
    for byte in value.as_bytes() {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    hash % CURSOR_SUBAGENT_TURN_BUCKETS
}

fn project_from_pi_session_path(path: &Path) -> String {
    path.parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .map(|name| decode_pi_project_name(pi_session_dir_project_key(name)))
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "pi".to_string())
}

fn pi_session_dir_project_key(name: &str) -> &str {
    if name.starts_with("--") && name.ends_with("--") && name.len() > 4 {
        &name[1..name.len() - 1]
    } else {
        name
    }
}

fn pi_content_text(content: Option<&BorrowedValue>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    if let Some(text) = content.as_str() {
        return text.to_string();
    }
    if let Some(arr) = content.as_array() {
        let mut parts = Vec::new();
        for item in arr {
            if let Some(text) = item.as_str() {
                parts.push(text.to_string());
                continue;
            }
            let Some(obj) = item.as_object() else {
                continue;
            };
            match obj.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                "text" => {
                    if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                        parts.push(text.to_string());
                    }
                }
                "thinking" => {
                    if let Some(text) = obj.get("thinking").and_then(|v| v.as_str()) {
                        parts.push(format!("Thinking:\n{text}"));
                    }
                }
                "toolCall" => {}
                _ => {
                    if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                        parts.push(text.to_string());
                    } else if let Some(text) = obj.get("content").and_then(|v| v.as_str()) {
                        parts.push(text.to_string());
                    }
                }
            }
        }
        return parts.join("\n");
    }
    content.to_string()
}

fn pi_summary_message_text(message: &simd_json::borrowed::Object, role: &str) -> String {
    let summary = if role == "branchSummary" || role == "compactionSummary" {
        message.get("summary").and_then(|v| v.as_str())
    } else {
        None
    };
    summary
        .map(str::to_string)
        .unwrap_or_else(|| pi_content_text(message.get("content")))
        .trim()
        .to_string()
}

fn pi_bash_text(command: &str, output: &str, exit_code: Option<i64>) -> String {
    let mut parts = Vec::new();
    if !command.is_empty() {
        parts.push(format!("$ {command}"));
    }
    if !output.is_empty() {
        parts.push(output.to_string());
    }
    if let Some(code) = exit_code {
        parts.push(format!("exit code: {code}"));
    }
    parts.join("\n")
}

fn decode_pi_project_name(folder_name: &str) -> String {
    let decoded = decode_project_name(folder_name);
    decoded
        .rsplit('-')
        .find(|part| !part.is_empty())
        .unwrap_or(&decoded)
        .to_string()
}

fn decode_project_name(folder_name: &str) -> String {
    let prefixes_to_strip = ["-home-", "-mnt-c-Users-", "-mnt-c-users-", "-Users-"];
    let mut name = folder_name;
    if name.len() > 10 {
        let bytes = name.as_bytes();
        if bytes[0] == b'-'
            && bytes[2] == b'-'
            && bytes[3] == b'-'
            && bytes[1].is_ascii_alphabetic()
            && name[4..].to_lowercase().starts_with("users-")
        {
            name = &name[10..];
        }
    }
    for prefix in prefixes_to_strip {
        if name.to_lowercase().starts_with(&prefix.to_lowercase()) {
            name = &name[prefix.len()..];
            break;
        }
    }
    let parts: Vec<&str> = name.split('-').filter(|p| !p.is_empty()).collect();
    let skip_dirs = [
        "projects",
        "code",
        "repos",
        "src",
        "dev",
        "work",
        "documents",
    ];
    let mut meaningful = Vec::new();
    let mut found_project = false;

    for (i, part) in parts.iter().enumerate() {
        if i == 0 && !found_project {
            let remaining: Vec<String> = parts[i + 1..].iter().map(|p| p.to_lowercase()).collect();
            if remaining.iter().any(|d| skip_dirs.contains(&d.as_str())) {
                continue;
            }
        }
        if skip_dirs.contains(&part.to_lowercase().as_str()) {
            found_project = true;
            continue;
        }
        meaningful.push(*part);
        found_project = true;
    }
    if meaningful.is_empty() {
        return folder_name.to_string();
    }
    meaningful.join("-")
}

fn extract_text_from_tool_result(block: &simd_json::BorrowedValue) -> Option<String> {
    let obj = block.as_object()?;
    let content = obj.get("content")?;
    if let Some(text) = content.as_str() {
        return Some(text.to_string());
    }
    if let Some(arr) = content.as_array() {
        let mut parts = Vec::new();
        for item in arr {
            if let Some(obj) = item.as_object()
                && obj.get("type").and_then(|v| v.as_str()) == Some("text")
                && let Some(text) = obj.get("text").and_then(|v| v.as_str())
            {
                parts.push(text);
            }
        }
        if !parts.is_empty() {
            return Some(parts.join(" "));
        }
    }
    None
}

fn session_id_from_filename(path: &Path) -> Option<String> {
    static UUID_RE: once_cell::sync::Lazy<regex::Regex> = once_cell::sync::Lazy::new(|| {
        regex::Regex::new(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
            .expect("uuid regex")
    });
    let name = path.file_stem()?.to_string_lossy();
    UUID_RE
        .captures(&name)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
}

fn is_system_instruction(text: &str) -> bool {
    let t = text.trim_start();
    t.starts_with("<system_instruction>") || t.starts_with("<system-instruction>")
}

fn flush_embeddings(
    buffer: &mut Vec<(u64, String, SourceKind)>,
    embedder: &mut EmbedderHandle,
    vindex: &mut crate::vector::VectorIndex,
    progress: &Arc<Progress>,
) -> Result<usize> {
    if buffer.is_empty() {
        return Ok(0);
    }

    // Prepare texts for batch embedding
    let items: Vec<(u64, String, SourceKind)> = buffer
        .drain(..)
        .map(|(doc_id, text, source)| (doc_id, truncate_for_embedding(text), source))
        .filter(|(_, text, _)| !text.is_empty())
        .collect();

    if items.is_empty() {
        return Ok(0);
    }

    // Batch embed all texts at once (ONNX Runtime handles internal parallelism)
    let texts: Vec<&str> = items.iter().map(|(_, text, _)| text.as_str()).collect();
    let embeddings = embedder.embed_texts(&texts)?;

    // Add embeddings to index
    let mut count = 0;
    for ((doc_id, _, source), vec) in items.iter().zip(embeddings.iter()) {
        vindex.add(*doc_id, vec)?;
        progress.sub_embed_pending(*source, 1);
        progress.add_embedded(*source, 1);
        count += 1;
    }
    Ok(count)
}

fn compute_totals(tasks: &[FileTask]) -> [u64; SOURCE_COUNT] {
    let mut totals = [0u64; SOURCE_COUNT];
    for task in tasks {
        let remaining = task.size.saturating_sub(task.offset);
        totals[task.source.idx()] += remaining;
    }
    totals
}

fn compute_file_totals(tasks: &[FileTask]) -> [u64; SOURCE_COUNT] {
    let mut totals = [0u64; SOURCE_COUNT];
    for task in tasks {
        totals[task.source.idx()] += 1;
    }
    totals
}

fn truncate_for_embedding(mut text: String) -> String {
    if text.len() <= EMBED_MAX_CHARS {
        return text;
    }
    let mut end = EMBED_MAX_CHARS.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    text
}

fn limit_record_tool_content(record: &mut Record, limits: IndexedToolContentLimits) {
    let text_limit = match record.role.as_str() {
        "tool_use" => Some(limits.input_bytes),
        "tool_result" => Some(limits.output_bytes),
        _ if record.tool_output.is_some() => Some(limits.output_bytes),
        _ if record.tool_input.is_some() => Some(limits.input_bytes),
        _ => None,
    };
    if let Some(max_bytes) = text_limit {
        truncate_for_index(&mut record.text, max_bytes);
    }
    if let Some(tool_input) = record.tool_input.as_mut() {
        truncate_for_index(tool_input, limits.input_bytes);
    }
    if let Some(tool_output) = record.tool_output.as_mut() {
        truncate_for_index(tool_output, limits.output_bytes);
    }
}

fn truncate_for_index(text: &mut String, max_bytes: usize) {
    if text.len() <= max_bytes {
        return;
    }

    let original_len = text.len();
    let mut marker = truncation_marker(original_len);
    let (head_end, tail_start) = loop {
        let retained_bytes = max_bytes.saturating_sub(marker.len());
        let head_target = retained_bytes.saturating_mul(RETAINED_HEAD_PERCENT) / 100;
        let head_end = char_boundary_at_or_before(text, head_target);
        let tail_target = retained_bytes.saturating_sub(head_end);
        let tail_start = char_boundary_at_or_after(text, original_len.saturating_sub(tail_target));
        let omitted_bytes = tail_start.saturating_sub(head_end);
        let updated_marker = truncation_marker(omitted_bytes);
        if updated_marker.len() == marker.len() {
            marker = updated_marker;
            break (head_end, tail_start);
        }
        marker = updated_marker;
    };

    let tail = text[tail_start..].to_string();
    text.truncate(head_end);
    text.push_str(&marker);
    text.push_str(&tail);
}

fn char_boundary_at_or_before(text: &str, mut position: usize) -> usize {
    position = position.min(text.len());
    while position > 0 && !text.is_char_boundary(position) {
        position -= 1;
    }
    position
}

fn char_boundary_at_or_after(text: &str, mut position: usize) -> usize {
    position = position.min(text.len());
    while position < text.len() && !text.is_char_boundary(position) {
        position += 1;
    }
    position
}

fn truncation_marker(omitted_bytes: usize) -> String {
    format!("\n\n[... {omitted_bytes} bytes truncated ...]\n\n")
}

fn is_embedding_role(role: &str) -> bool {
    role == "user" || role == "assistant"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{IndexedToolContentLimits, Paths};
    use crate::embed::{EmbedRuntimeConfig, ModelChoice};
    use crate::index::SearchIndex;
    use crate::test_support::{EnvVarGuard, env_lock};
    use crate::vector::VectorIndex;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn ingest_options(embeddings: bool, model: ModelChoice) -> IngestOptions {
        IngestOptions {
            claude_source: PathBuf::from("/does/not/exist"),
            include_agents: false,
            include_codex: false,
            include_opencode: false,
            include_cursor: false,
            include_pi: false,
            include_copilot: false,
            include_omp: false,
            omp_source: PathBuf::from("/does/not/exist"),
            embeddings,
            backfill_embeddings: false,
            model,
            embed_runtime: EmbedRuntimeConfig::default(),
            tool_content_limits: IndexedToolContentLimits::default(),
        }
    }

    fn save_vector_store(paths: &Paths, model: &str, dimensions: usize) {
        let mut vector = VectorIndex::open_or_create(&paths.vectors, dimensions, Some(model))
            .expect("open vector store");
        vector.add(1, &vec![0.0; dimensions]).expect("add vector");
        vector.save().unwrap();
    }

    fn open_search_index(paths: &Paths) -> SearchIndex {
        fs::create_dir_all(&paths.index).expect("create index dir");
        SearchIndex::open_or_create(&paths.index).expect("open search index")
    }

    fn save_search_records(paths: &Paths, records: &[Record]) -> SearchIndex {
        let index = open_search_index(paths);
        let mut writer = index.writer().expect("open index writer");
        for record in records {
            index.add_record(&mut writer, record).expect("add record");
        }
        writer.commit().expect("commit records");
        index
    }

    fn mark_analytics_complete(paths: &Paths) {
        AnalyticsStore::open(analytics_path(&paths.state))
            .expect("open analytics")
            .mark_complete()
            .expect("mark analytics complete");
    }

    fn record(doc_id: u64, role: &str, text: &str) -> Record {
        Record {
            source: SourceKind::Claude,
            doc_id,
            ts: doc_id,
            project: "project".to_string(),
            session_id: "session".to_string(),
            turn_id: doc_id as u32,
            role: role.to_string(),
            text: text.to_string(),
            tool_name: None,
            tool_input: None,
            tool_output: None,
            links: RecordLinks::default(),
            source_path: format!("source-{doc_id}.jsonl"),
        }
    }

    fn parse_omp_records(path: &Path, offset: u64, turn_id: u32) -> Result<Vec<Record>> {
        let metadata = path.metadata()?;
        let task = FileTask {
            path: path.to_path_buf(),
            source: SourceKind::Omp,
            offset,
            turn_id,
            size: metadata.len(),
            mtime: 0,
            delete_first: false,
        };
        let (raw_tx_record, rx_record) = unbounded();
        let tx_record = RecordSender::new(raw_tx_record, IndexedToolContentLimits::default());
        let (tx_update, _rx_update) = unbounded();
        let next_doc_id = AtomicU64::new(1);
        let mut byte_totals = [0; SOURCE_COUNT];
        byte_totals[SourceKind::Omp.idx()] = metadata.len();
        let mut file_totals = [0; SOURCE_COUNT];
        file_totals[SourceKind::Omp.idx()] = 1;
        let progress = Arc::new(Progress::new(byte_totals, file_totals, false));

        parse_omp_file(&task, &tx_record, &tx_update, &next_doc_id, &progress)?;
        drop(tx_record);
        Ok(rx_record.try_iter().collect())
    }

    #[test]
    fn record_channel_applies_backpressure_at_capacity() {
        let (tx_record, _rx_record) = record_channel();
        for doc_id in 0..RECORD_CHANNEL_CAPACITY {
            tx_record
                .try_send(record(doc_id as u64, "assistant", "text"))
                .expect("record within channel capacity");
        }

        let result =
            tx_record.try_send(record(RECORD_CHANNEL_CAPACITY as u64, "assistant", "text"));
        assert!(matches!(
            result,
            Err(crossbeam_channel::TrySendError::Full(_))
        ));
    }

    #[test]
    fn record_sender_caps_tool_payloads_but_keeps_plain_text() {
        let limits = IndexedToolContentLimits {
            input_bytes: 1024,
            output_bytes: 2048,
        };
        let plain_text = format!("plain-begin{}plain-end", "w".repeat(4096));
        let plain = record(1, "assistant", &plain_text);

        let mut tool_use = record(
            2,
            "tool_use",
            &format!("input-begin{}input-end", "🦀".repeat(2048)),
        );
        tool_use.tool_input = Some(tool_use.text.clone());
        let mut tool_result = record(
            3,
            "tool_result",
            &format!("output-begin{}output-end", "y".repeat(4096)),
        );
        tool_result.tool_output = Some(tool_result.text.clone());
        let role_only_tool_result = record(
            4,
            "tool_result",
            &format!("role-output-begin{}role-output-end", "z".repeat(4096)),
        );

        let (raw_tx, rx) = unbounded();
        let tx = RecordSender::new(raw_tx, limits);
        tx.send(plain).expect("queue plain record");
        tx.send(tool_use).expect("queue tool-use record");
        tx.send(tool_result).expect("queue tool-result record");
        tx.send(role_only_tool_result)
            .expect("queue role-only tool-result record");
        drop(tx);
        let records = rx.iter().collect::<Vec<_>>();

        assert_eq!(records[0].text, plain_text);
        assert_truncated_content(
            &records[1].text,
            limits.input_bytes,
            "input-begin",
            "input-end",
        );
        assert_truncated_content(
            records[1].tool_input.as_deref().expect("tool input"),
            limits.input_bytes,
            "input-begin",
            "input-end",
        );
        assert_truncated_content(
            &records[2].text,
            limits.output_bytes,
            "output-begin",
            "output-end",
        );
        assert_truncated_content(
            records[2].tool_output.as_deref().expect("tool output"),
            limits.output_bytes,
            "output-begin",
            "output-end",
        );
        assert_truncated_content(
            &records[3].text,
            limits.output_bytes,
            "role-output-begin",
            "role-output-end",
        );
    }

    fn assert_truncated_content(content: &str, max_bytes: usize, prefix: &str, suffix: &str) {
        assert!(content.len() <= max_bytes);
        assert!(content.starts_with(prefix));
        assert!(content.contains("bytes truncated"));
        assert!(content.ends_with(suffix));
    }

    fn fresh_scan_cache() -> ScanCache {
        let last_scan_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_secs();
        ScanCache {
            last_scan_ts,
            file_count: 0,
            total_bytes: 0,
        }
    }

    #[test]
    fn cursor_session_id_uses_agent_transcripts_session_directory() {
        let path = Path::new(
            "/Users/nico/.cursor/projects/-Users-nico-Code-memex/agent-transcripts/\
             11111111-1111-1111-1111-111111111111/\
             11111111-1111-1111-1111-111111111111.jsonl",
        );

        assert_eq!(
            cursor_session_id_from_path(path),
            "11111111-1111-1111-1111-111111111111"
        );
    }

    #[test]
    fn cursor_session_id_strips_direct_transcript_extension() {
        let path = Path::new(
            "/Users/nico/.cursor/projects/-Users-nico-Code-memex/agent-transcripts/\
             11111111-1111-1111-1111-111111111111.jsonl",
        );

        assert_eq!(
            cursor_session_id_from_path(path),
            "11111111-1111-1111-1111-111111111111"
        );
    }

    #[test]
    fn cursor_session_id_uses_parent_session_for_subagent_transcripts() {
        let path = Path::new(
            "/Users/nico/.cursor/projects/-Users-nico-Code-memex/agent-transcripts/\
             11111111-1111-1111-1111-111111111111/subagents/\
             22222222-2222-2222-2222-222222222222.jsonl",
        );

        assert_eq!(
            cursor_session_id_from_path(path),
            "11111111-1111-1111-1111-111111111111"
        );
    }

    #[test]
    fn cursor_parent_transcripts_start_at_cached_turn_id() {
        let path = Path::new(
            "/Users/nico/.cursor/projects/-Users-nico-Code-memex/agent-transcripts/\
             11111111-1111-1111-1111-111111111111/\
             11111111-1111-1111-1111-111111111111.jsonl",
        );

        assert_eq!(cursor_initial_turn_id(path, 0), 0);
        assert_eq!(cursor_initial_turn_id(path, 42), 42);
    }

    #[test]
    fn cursor_subagent_transcripts_use_reserved_turn_range() {
        let path = Path::new(
            "/Users/nico/.cursor/projects/-Users-nico-Code-memex/agent-transcripts/\
             11111111-1111-1111-1111-111111111111/subagents/\
             22222222-2222-2222-2222-222222222222.jsonl",
        );

        let initial = cursor_initial_turn_id(path, 0);
        assert!(initial >= CURSOR_SUBAGENT_TURN_BASE);
        assert_eq!(cursor_initial_turn_id(path, initial + 3), initial + 3);
    }

    #[test]
    fn cursor_record_links_mark_subagent_parent_session() {
        let path = Path::new(
            "/Users/nico/.cursor/projects/-Users-nico-Code-memex/agent-transcripts/\
             11111111-1111-1111-1111-111111111111/subagents/\
             22222222-2222-2222-2222-222222222222.jsonl",
        );

        let links = cursor_record_links(path, "11111111-1111-1111-1111-111111111111", 42);

        assert_eq!(
            links.event_id.as_deref(),
            Some("22222222-2222-2222-2222-222222222222:42")
        );
        assert_eq!(
            links.parent_session_id.as_deref(),
            Some("11111111-1111-1111-1111-111111111111")
        );
        assert_eq!(links.thread_source.as_deref(), Some("subagent"));
        assert_eq!(links.conversation_kind.as_deref(), Some("subagent"));
    }

    #[test]
    fn opencode_session_links_preserve_parent_id() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let project = tmp.path().join("project");
        fs::create_dir_all(&project).expect("create opencode project");
        fs::write(
            project.join("ses_child.json"),
            r#"{"id":"ses_child","parentID":"ses_parent","projectID":"global"}"#,
        )
        .expect("write opencode session");

        let links = opencode_session_links_from_root(tmp.path(), "ses_child");

        assert_eq!(links.parent_session_id.as_deref(), Some("ses_parent"));
        assert_eq!(links.thread_source.as_deref(), Some("fork"));
        assert_eq!(links.conversation_kind.as_deref(), Some("fork"));
    }

    #[test]
    fn opencode_session_links_by_id_caches_metadata_tree() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let project = tmp.path().join("project");
        fs::create_dir_all(&project).expect("create opencode project");
        fs::write(
            project.join("ses_child.json"),
            r#"{"id":"ses_child","parentID":"ses_parent","projectID":"global"}"#,
        )
        .expect("write child session");
        fs::write(
            project.join("ses_main.json"),
            r#"{"id":"ses_main","projectID":"global"}"#,
        )
        .expect("write main session");

        let links_by_id = opencode_session_links_by_id_from_root(tmp.path());
        let child_links = links_by_id.get("ses_child").expect("child links");
        let main_links = links_by_id.get("ses_main").expect("main links");

        assert_eq!(links_by_id.len(), 2);
        assert_eq!(child_links.parent_session_id.as_deref(), Some("ses_parent"));
        assert_eq!(child_links.thread_source.as_deref(), Some("fork"));
        assert_eq!(child_links.conversation_kind.as_deref(), Some("fork"));
        assert_eq!(main_links.parent_session_id, None);
        assert_eq!(main_links.thread_source, None);
        assert_eq!(main_links.conversation_kind.as_deref(), Some("main"));
    }

    #[test]
    fn codex_session_meta_preserves_fork_and_subagent_links() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp
            .path()
            .join("rollout-2026-05-22T13-17-11-019e5155-b507-7d83-8c3d-9ecee5f93f12.jsonl");
        fs::write(
            &path,
            r#"{"timestamp":"2026-05-22T20:17:12.595Z","type":"session_meta","payload":{"id":"019e5155-b507-7d83-8c3d-9ecee5f93f12","forked_from_id":"019e5117-c673-7660-b218-af0489416e0f","cwd":"/tmp/project","source":{"subagent":{"thread_spawn":{"parent_thread_id":"019e5117-c673-7660-b218-af0489416e0f","depth":1}}},"thread_source":"subagent"}}"#
                .to_string()
                + "\n",
        )
        .expect("write codex session");

        let meta = read_codex_session_meta_until(&path, fs::metadata(&path).unwrap().len())
            .expect("read codex meta");

        assert_eq!(meta.session_id, "019e5155-b507-7d83-8c3d-9ecee5f93f12");
        assert_eq!(meta.project, "project");
        assert_eq!(
            meta.links.parent_session_id.as_deref(),
            Some("019e5117-c673-7660-b218-af0489416e0f")
        );
        assert_eq!(meta.links.thread_source.as_deref(), Some("subagent"));
        assert_eq!(meta.links.conversation_kind.as_deref(), Some("subagent"));
    }

    #[test]
    fn ingest_claude_records_preserve_sidechain_and_tool_links() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let claude_root = tmp.path().join("claude-projects");
        let project_root = claude_root.join("-Users-nico-Code-memex");
        fs::create_dir_all(&project_root).expect("create claude project");
        let session_file = project_root.join("sess-claude.jsonl");
        fs::write(
            &session_file,
            r#"{"type":"user","uuid":"u1","parentUuid":null,"sessionId":"sess-claude","isSidechain":false,"timestamp":"2026-03-11T01:23:43.844Z","message":{"content":"question"}}
{"type":"assistant","uuid":"a1","parentUuid":"u1","logicalParentUuid":"u0","sessionId":"sess-claude","isSidechain":true,"sourceToolUseID":"source-tool","sourceToolAssistantUUID":"source-assistant","timestamp":"2026-03-11T01:23:44.844Z","message":{"content":[{"type":"text","text":"answer"},{"type":"tool_use","id":"tool-claude","name":"Read","input":{"file_path":"Cargo.toml"}}]}}
{"type":"user","uuid":"r1","parentUuid":"a1","sessionId":"sess-claude","isSidechain":true,"timestamp":"2026-03-11T01:23:45.844Z","message":{"content":[{"type":"tool_result","tool_use_id":"tool-claude","content":"ok"}]}}
"#,
        )
        .expect("write claude fixture");

        let paths = Paths::new(Some(tmp.path().join("memex"))).expect("paths");
        paths.ensure_dirs().expect("ensure dirs");
        let index = SearchIndex::open_or_create(&paths.index).expect("index");
        let options = IngestOptions {
            claude_source: claude_root,
            include_agents: false,
            include_codex: false,
            include_opencode: false,
            include_cursor: false,
            include_pi: false,
            include_copilot: false,
            include_omp: false,
            omp_source: tmp.path().join("missing-omp"),
            embeddings: false,
            backfill_embeddings: false,
            model: ModelChoice::default(),
            embed_runtime: EmbedRuntimeConfig::default(),
            tool_content_limits: IndexedToolContentLimits::default(),
        };

        let report = ingest_all(&paths, &index, &options).expect("ingest");
        assert_eq!(report.records_added, 4);

        let mut records = index
            .records_by_session_id("sess-claude")
            .expect("records by session");
        records.sort_by_key(|record| record.turn_id);

        assert_eq!(records.len(), 4);
        assert_eq!(records[0].role, "user");
        assert_eq!(records[0].links.event_id.as_deref(), Some("u1"));
        assert_eq!(records[0].links.conversation_kind.as_deref(), Some("main"));
        assert_eq!(records[1].role, "tool_use");
        assert_eq!(records[1].links.event_id.as_deref(), Some("tool-claude"));
        assert_eq!(records[1].links.parent_event_id.as_deref(), Some("a1"));
        assert_eq!(
            records[1].links.logical_parent_event_id.as_deref(),
            Some("u0")
        );
        assert_eq!(
            records[1].links.source_tool_use_id.as_deref(),
            Some("source-tool")
        );
        assert_eq!(
            records[1].links.source_tool_assistant_uuid.as_deref(),
            Some("source-assistant")
        );
        assert_eq!(records[1].links.thread_source.as_deref(), Some("sidechain"));
        assert_eq!(
            records[1].links.conversation_kind.as_deref(),
            Some("sidechain")
        );
        assert_eq!(records[2].role, "assistant");
        assert_eq!(records[2].links.event_id.as_deref(), Some("a1"));
        assert_eq!(records[2].links.parent_event_id.as_deref(), Some("u1"));
        assert_eq!(records[2].links.thread_source.as_deref(), Some("sidechain"));
        assert_eq!(
            records[2].links.conversation_kind.as_deref(),
            Some("sidechain")
        );
        assert_eq!(records[3].role, "tool_result");
        assert_eq!(
            records[3].links.event_id.as_deref(),
            Some("r1:tool_result:tool-claude")
        );
        assert_eq!(
            records[3].links.parent_event_id.as_deref(),
            Some("tool-claude")
        );
        assert_eq!(
            records[3].links.parent_tool_use_id.as_deref(),
            Some("tool-claude")
        );
    }

    #[test]
    fn collect_codex_session_files_includes_archived_sessions() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let sessions_root = tmp.path().join("sessions");
        let archived_root = tmp.path().join("archived_sessions");

        let sessions_day = sessions_root.join("2026/02/11");
        fs::create_dir_all(&sessions_day).expect("create sessions day");
        fs::create_dir_all(archived_root.join("state")).expect("create archived state");

        let live = sessions_day.join("session-live.jsonl");
        let archived = archived_root.join("rollout-archive.jsonl");
        let ignored = archived_root.join("state/ingest.json");

        fs::write(&live, "{}\n").expect("write live");
        fs::write(&archived, "{}\n").expect("write archived");
        fs::write(&ignored, "{}\n").expect("write ignored");

        let mut files = collect_codex_session_files_from_roots(&[sessions_root, archived_root])
            .expect("collect codex sessions");
        files.sort();

        assert_eq!(files, vec![archived, live]);
    }

    #[test]
    fn can_skip_noop_index_when_embeddings_are_disabled() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(tmp.path().to_path_buf())).expect("paths");
        let index = open_search_index(&paths);
        let options = ingest_options(false, ModelChoice::BGESmall);

        assert!(can_skip_noop_index(&paths, &index, &options).unwrap());
    }

    #[test]
    fn can_skip_fresh_scan_when_embeddings_are_disabled() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(tmp.path().to_path_buf())).expect("paths");
        let index = save_search_records(&paths, &[record(1, "user", "hello")]);
        let options = ingest_options(false, ModelChoice::BGESmall);
        let cache = fresh_scan_cache();
        mark_analytics_complete(&paths);

        assert!(can_skip_fresh_scan(&cache, &paths, &index, &options, 60).unwrap());
    }

    #[test]
    fn can_skip_fresh_scan_with_compatible_vectors() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(tmp.path().to_path_buf())).expect("paths");
        save_vector_store(&paths, "bge", 384);
        let index = save_search_records(&paths, &[record(1, "user", "hello")]);
        let options = ingest_options(true, ModelChoice::BGESmall);
        let cache = fresh_scan_cache();
        mark_analytics_complete(&paths);

        assert!(can_skip_fresh_scan(&cache, &paths, &index, &options, 60).unwrap());
    }

    #[test]
    fn cannot_skip_fresh_scan_when_vectors_are_missing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(tmp.path().to_path_buf())).expect("paths");
        let index = open_search_index(&paths);
        let options = ingest_options(true, ModelChoice::BGESmall);
        let cache = fresh_scan_cache();

        assert!(!can_skip_fresh_scan(&cache, &paths, &index, &options, 60).unwrap());
    }

    #[test]
    fn cannot_skip_fresh_scan_with_incompatible_vectors() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(tmp.path().to_path_buf())).expect("paths");
        save_vector_store(&paths, "minilm", 384);
        let index = open_search_index(&paths);
        let options = ingest_options(true, ModelChoice::BGESmall);
        let cache = fresh_scan_cache();

        assert!(!can_skip_fresh_scan(&cache, &paths, &index, &options, 60).unwrap());
    }

    #[test]
    fn cannot_skip_fresh_scan_when_cache_is_stale() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(tmp.path().to_path_buf())).expect("paths");
        let index = open_search_index(&paths);
        let options = ingest_options(false, ModelChoice::BGESmall);
        let cache = ScanCache {
            last_scan_ts: 0,
            file_count: 0,
            total_bytes: 0,
        };

        assert!(!can_skip_fresh_scan(&cache, &paths, &index, &options, 60).unwrap());
    }

    #[test]
    fn can_skip_noop_index_with_compatible_vectors() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(tmp.path().to_path_buf())).expect("paths");
        save_vector_store(&paths, "bge", 384);
        let index = save_search_records(&paths, &[record(1, "user", "hello")]);
        let options = ingest_options(true, ModelChoice::BGESmall);

        assert!(can_skip_noop_index(&paths, &index, &options).unwrap());
    }

    #[test]
    fn cannot_skip_noop_index_with_partial_compatible_vectors() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(tmp.path().to_path_buf())).expect("paths");
        save_vector_store(&paths, "bge", 384);
        let index = save_search_records(
            &paths,
            &[
                record(1, "user", "embedded"),
                record(2, "assistant", "missing vector"),
            ],
        );
        let options = ingest_options(true, ModelChoice::BGESmall);

        assert!(!can_skip_noop_index(&paths, &index, &options).unwrap());
    }

    #[test]
    fn can_skip_noop_index_ignores_records_that_do_not_need_embeddings() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(tmp.path().to_path_buf())).expect("paths");
        save_vector_store(&paths, "bge", 384);
        let index = save_search_records(
            &paths,
            &[
                record(1, "user", "embedded"),
                record(2, "tool_result", "not embedded"),
                record(3, "assistant", ""),
            ],
        );
        let options = ingest_options(true, ModelChoice::BGESmall);

        assert!(can_skip_noop_index(&paths, &index, &options).unwrap());
    }

    #[test]
    fn cannot_skip_noop_index_when_vectors_are_missing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(tmp.path().to_path_buf())).expect("paths");
        let index = open_search_index(&paths);
        let options = ingest_options(true, ModelChoice::BGESmall);

        assert!(!can_skip_noop_index(&paths, &index, &options).unwrap());
    }

    #[test]
    fn cannot_skip_noop_index_with_incompatible_vectors() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(tmp.path().to_path_buf())).expect("paths");
        save_vector_store(&paths, "minilm", 384);
        let index = open_search_index(&paths);
        let options = ingest_options(true, ModelChoice::BGESmall);

        assert!(!can_skip_noop_index(&paths, &index, &options).unwrap());
    }

    #[test]
    fn cannot_skip_noop_index_with_wrong_vector_dimensions() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(tmp.path().to_path_buf())).expect("paths");
        save_vector_store(&paths, "bge", 768);
        let index = open_search_index(&paths);
        let options = ingest_options(true, ModelChoice::BGESmall);

        assert!(!can_skip_noop_index(&paths, &index, &options).unwrap());
    }

    #[test]
    fn cannot_skip_noop_index_when_model_dimensions_are_dynamic() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(tmp.path().to_path_buf())).expect("paths");
        save_vector_store(&paths, "potion", 256);
        let index = open_search_index(&paths);
        let options = ingest_options(true, ModelChoice::Potion);

        assert!(!can_skip_noop_index(&paths, &index, &options).unwrap());
    }
    #[test]
    fn collect_pi_files_recurses_under_sessions_root() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let sessions_root = tmp.path().join("sessions");
        let project_root = sessions_root.join("--Users-nico-Code-memex--");
        fs::create_dir_all(&project_root).expect("create pi session dir");

        let session =
            project_root.join("20260703T010203Z_11111111-1111-1111-1111-111111111111.jsonl");
        let ignored = project_root.join("notes.json");
        fs::write(&session, "{}\n").expect("write pi session");
        fs::write(&ignored, "{}\n").expect("write ignored");

        let files = collect_pi_files_from_root(&sessions_root).expect("collect pi files");

        assert_eq!(files, vec![session]);
    }

    #[test]
    fn pi_sessions_root_honors_session_dir_override() {
        let _guard = env_lock();
        let tmp = tempfile::tempdir().expect("tempdir");
        let custom_sessions = tmp.path().join("custom-sessions");
        let _env = EnvVarGuard::set_os(&[
            (
                "PI_CODING_AGENT_SESSION_DIR",
                Some(custom_sessions.as_os_str()),
            ),
            ("PI_CODING_AGENT_DIR", None),
        ]);

        assert_eq!(pi_sessions_root(), custom_sessions);
    }

    #[test]
    fn pi_sessions_root_honors_settings_session_dir() {
        let _guard = env_lock();
        let tmp = tempfile::tempdir().expect("tempdir");
        let pi_root = tmp.path().join("pi-agent");
        fs::create_dir_all(&pi_root).expect("create pi root");
        fs::write(
            pi_root.join("settings.json"),
            r#"{ "sessionDir": ".pi/sessions" }"#,
        )
        .expect("write settings");
        let _env = EnvVarGuard::set_os(&[
            ("PI_CODING_AGENT_SESSION_DIR", None),
            ("PI_CODING_AGENT_DIR", Some(pi_root.as_os_str())),
        ]);

        assert_eq!(pi_sessions_root(), pi_root.join(".pi/sessions"));
    }

    #[test]
    fn pi_session_path_fallback_preserves_project_name() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let home_path = tmp
            .path()
            .join("sessions")
            .join("--home-alice-code-memex--")
            .join("20260703T010203Z_11111111-1111-1111-1111-111111111111.jsonl");
        let users_path = tmp
            .path()
            .join("sessions")
            .join("--Users-nico-Code-memex--")
            .join("20260703T010203Z_11111111-1111-1111-1111-111111111111.jsonl");
        let windows_path = tmp
            .path()
            .join("sessions")
            .join("--C--Users-alice-Code-memex--")
            .join("20260703T010203Z_11111111-1111-1111-1111-111111111111.jsonl");
        let nested_path = tmp
            .path()
            .join("sessions")
            .join("--home-alice-code-acme-memex--")
            .join("20260703T010203Z_11111111-1111-1111-1111-111111111111.jsonl");

        assert_eq!(project_from_pi_session_path(&home_path), "memex");
        assert_eq!(project_from_pi_session_path(&users_path), "memex");
        assert_eq!(project_from_pi_session_path(&windows_path), "memex");
        assert_eq!(project_from_pi_session_path(&nested_path), "memex");
    }

    #[test]
    fn ingest_pi_session_records_supported_message_shapes() {
        let _guard = env_lock();
        let tmp = tempfile::tempdir().expect("tempdir");
        let pi_root = tmp.path().join("pi-agent");
        let sessions_root = pi_root.join("sessions").join("--Users-nico-Code-memex--");
        fs::create_dir_all(&sessions_root).expect("create pi sessions");
        let session_file =
            sessions_root.join("20260703T010203Z_11111111-1111-1111-1111-111111111111.jsonl");
        fs::write(
            &session_file,
            r#"{"type":"session","version":3,"id":"11111111-1111-1111-1111-111111111111","timestamp":"2026-07-03T01:02:03Z","cwd":"/Users/nico/Code/memex"}
{"type":"message","id":"u1","timestamp":"2026-07-03T01:02:04Z","message":{"role":"user","content":[{"type":"text","text":"hello pi"}]}}
{"type":"message","id":"a1","parentId":"u1","timestamp":"2026-07-03T01:02:05Z","message":{"role":"assistant","content":[{"type":"thinking","thinking":"considering options"},{"type":"text","text":"I will run a command"},{"type":"toolCall","id":"tc1","name":"Read","arguments":{"file_path":"README.md"}}]}}
{"type":"message","id":"tr1","parentId":"a1","timestamp":"2026-07-03T01:02:06Z","message":{"role":"toolResult","toolCallId":"tc1","toolName":"Read","content":[{"type":"text","text":"README contents"}],"isError":false}}
{"type":"message","id":"b1","parentId":"tr1","timestamp":"2026-07-03T01:02:07Z","message":{"role":"bashExecution","command":"cargo test","output":"ok","exitCode":0,"cancelled":false,"truncated":false}}
{"type":"message","id":"bh1","parentId":"b1","timestamp":"2026-07-03T01:02:07Z","message":{"role":"bashExecution","command":"echo secret","output":"secret output","exitCode":0,"excludeFromContext":true}}
{"type":"compaction","id":"c1","parentId":"b1","timestamp":"2026-07-03T01:02:08Z","summary":"compacted top-level summary","firstKeptEntryId":"tr1","tokensBefore":50000}
{"type":"branch_summary","id":"br1","parentId":"u1","timestamp":"2026-07-03T01:02:09Z","fromId":"c1","summary":"branch top-level summary"}
{"type":"custom_message","id":"cm1","parentId":"br1","timestamp":"2026-07-03T01:02:10Z","customType":"memex","content":[{"type":"text","text":"extension context"}],"display":true}
{"type":"message","id":"mcs1","parentId":"cm1","timestamp":"2026-07-03T01:02:11Z","message":{"role":"compactionSummary","content":"summary text"}}
{"type":"message","id":"mbs1","parentId":"mcs1","timestamp":"2026-07-03T01:02:12Z","message":{"role":"branchSummary","summary":"message summary text"}}
"#,
        )
        .expect("write pi fixture");
        let _env = EnvVarGuard::set_os(&[("PI_CODING_AGENT_DIR", Some(pi_root.as_os_str()))]);

        let paths = Paths::new(Some(tmp.path().join("memex"))).expect("paths");
        paths.ensure_dirs().expect("ensure dirs");
        let index = SearchIndex::open_or_create(&paths.index).expect("index");
        let options = IngestOptions {
            claude_source: tmp.path().join("missing-claude"),
            include_agents: false,
            include_codex: false,
            include_opencode: false,
            include_cursor: false,
            include_pi: true,
            include_copilot: false,
            include_omp: false,
            omp_source: tmp.path().join("missing-omp"),
            embeddings: false,
            backfill_embeddings: false,
            model: ModelChoice::default(),
            embed_runtime: EmbedRuntimeConfig::default(),
            tool_content_limits: IndexedToolContentLimits::default(),
        };

        let report = ingest_all(&paths, &index, &options).expect("ingest");
        assert_eq!(report.records_added, 10);

        let mut records = index
            .records_by_session_id("11111111-1111-1111-1111-111111111111")
            .expect("records by session");
        records.sort_by_key(|record| record.turn_id);

        assert_eq!(records.len(), 10);
        assert!(records.iter().all(|record| record.source == SourceKind::Pi));
        assert!(records.iter().all(|record| record.project == "memex"));
        let source_path = session_file.to_string_lossy().to_string();
        assert!(
            records
                .iter()
                .all(|record| record.source_path == source_path)
        );
        assert_eq!(records[0].role, "user");
        assert_eq!(records[0].text, "hello pi");
        assert_eq!(records[0].links.event_id.as_deref(), Some("u1"));
        assert_eq!(records[0].links.conversation_kind.as_deref(), Some("main"));
        assert_eq!(records[1].role, "tool_use");
        assert_eq!(records[1].tool_name.as_deref(), Some("Read"));
        assert!(records[1].text.contains("README.md"));
        assert_eq!(records[1].links.event_id.as_deref(), Some("tc1"));
        assert_eq!(records[1].links.parent_event_id.as_deref(), Some("a1"));
        assert_eq!(records[2].role, "assistant");
        assert!(records[2].text.contains("I will run a command"));
        assert!(records[2].text.contains("Thinking:"));
        assert_eq!(records[2].links.event_id.as_deref(), Some("a1"));
        assert_eq!(records[2].links.parent_event_id.as_deref(), Some("u1"));
        assert_eq!(records[3].role, "tool_result");
        assert_eq!(records[3].tool_name.as_deref(), Some("Read"));
        assert_eq!(records[3].text, "README contents");
        assert_eq!(records[3].links.event_id.as_deref(), Some("tr1"));
        assert_eq!(records[3].links.parent_event_id.as_deref(), Some("a1"));
        assert_eq!(records[3].links.parent_tool_use_id.as_deref(), Some("tc1"));
        assert_eq!(records[4].role, "tool_result");
        assert_eq!(records[4].tool_name.as_deref(), Some("Bash"));
        assert!(records[4].text.contains("$ cargo test"));
        assert!(records[4].text.contains("exit code: 0"));
        assert_eq!(records[5].role, "assistant");
        assert_eq!(records[5].links.event_id.as_deref(), Some("c1"));
        assert_eq!(
            records[5].links.thread_source.as_deref(),
            Some("compaction")
        );
        assert_eq!(
            records[5].links.conversation_kind.as_deref(),
            Some("compaction")
        );
        assert_eq!(records[5].text, "compaction: compacted top-level summary");
        assert_eq!(records[6].role, "assistant");
        assert_eq!(records[6].text, "branch_summary: branch top-level summary");
        assert_eq!(records[6].links.event_id.as_deref(), Some("br1"));
        assert_eq!(records[6].links.parent_event_id.as_deref(), Some("u1"));
        assert_eq!(
            records[6].links.logical_parent_event_id.as_deref(),
            Some("c1")
        );
        assert_eq!(records[6].links.thread_source.as_deref(), Some("branch"));
        assert_eq!(
            records[6].links.conversation_kind.as_deref(),
            Some("branch")
        );
        assert_eq!(records[7].role, "assistant");
        assert_eq!(records[7].text, "custom_message(memex): extension context");
        assert_eq!(records[8].role, "assistant");
        assert_eq!(records[8].text, "compactionSummary: summary text");
        assert_eq!(
            records[8].links.conversation_kind.as_deref(),
            Some("compaction")
        );
        assert_eq!(records[9].role, "assistant");
        assert_eq!(records[9].text, "branchSummary: message summary text");
        assert_eq!(
            records[9].links.conversation_kind.as_deref(),
            Some("branch")
        );
        assert!(!records.iter().any(|record| record.text.contains("secret")));
    }

    #[test]
    fn ingest_pi_incremental_records_keep_header_project() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let sessions_root = tmp
            .path()
            .join("sessions")
            .join("--home-alice-code-my-project--");
        fs::create_dir_all(&sessions_root).expect("create pi sessions");
        let session_file =
            sessions_root.join("20260703T010203Z_11111111-1111-1111-1111-111111111111.jsonl");
        let existing = r#"{"type":"session","version":3,"id":"22222222-2222-2222-2222-222222222222","timestamp":"2026-07-03T01:02:03Z","cwd":"/home/alice/code/my-project"}
{"type":"message","id":"u1","timestamp":"2026-07-03T01:02:04Z","message":{"role":"user","content":"first"}}
"#;
        let appended = r#"{"type":"message","id":"a1","timestamp":"2026-07-03T01:02:05Z","message":{"role":"assistant","content":"second"}}
"#;
        fs::write(&session_file, format!("{existing}{appended}")).expect("write pi fixture");

        let (raw_tx_record, rx_record) = unbounded();
        let tx_record = RecordSender::new(raw_tx_record, IndexedToolContentLimits::default());
        let (tx_update, _rx_update) = unbounded();
        let task = FileTask {
            path: session_file,
            source: SourceKind::Pi,
            offset: existing.len() as u64,
            turn_id: 1,
            size: (existing.len() + appended.len()) as u64,
            mtime: 0,
            delete_first: false,
        };
        let progress = Arc::new(Progress::new([0; SOURCE_COUNT], [0; SOURCE_COUNT], false));
        let next_doc_id = AtomicU64::new(1);

        parse_pi_file(&task, &tx_record, &tx_update, &next_doc_id, &progress).expect("parse pi");
        drop(tx_record);
        let records: Vec<_> = rx_record.try_iter().collect();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].project, "my-project");
        assert_eq!(records[0].text, "second");
    }

    #[test]
    fn collect_omp_files_recurses_into_subagent_session_directories() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let project = tmp.path().join("-Users-nico-Code-memex");
        let parent = project.join("20260703T010203Z_11111111-1111-1111-1111-111111111111");
        fs::create_dir_all(&parent).expect("create omp subagent directory");
        let main = project.join("20260703T010203Z_11111111-1111-1111-1111-111111111111.jsonl");
        let child = parent.join("20260703T010204Z_22222222-2222-2222-2222-222222222222.jsonl");
        let ignored = parent.join("notes.json");
        fs::write(&main, "{}\n").expect("write main");
        fs::write(&child, "{}\n").expect("write child");
        fs::write(&ignored, "{}\n").expect("write ignored");

        let files = collect_omp_files_from_root(tmp.path()).expect("collect omp files");

        let mut expected = vec![main, child];
        expected.sort();
        assert_eq!(files, expected);
    }

    #[test]
    fn parse_omp_supports_v1_v2_and_v3_session_contracts() {
        let tmp = tempfile::tempdir().expect("tempdir");

        let v1 = tmp.path().join("v1.jsonl");
        fs::write(
            &v1,
            concat!(
                "{\"type\":\"session\",\"id\":\"v1-session\",\"timestamp\":\"2025-12-20T00:00:00Z\",\"cwd\":\"/tmp/v1-project\"}\n",
                "{\"type\":\"message\",\"timestamp\":\"2025-12-20T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":\"v1 question\"}}\n",
                "{\"type\":\"compaction\",\"timestamp\":\"2025-12-20T00:00:02Z\",\"summary\":\"v1 summary\",\"firstKeptEntryIndex\":1,\"tokensBefore\":100}\n"
            ),
        )
        .expect("write v1");
        let v1_records = parse_omp_records(&v1, 0, 0).expect("parse v1");
        assert_eq!(v1_records.len(), 2);
        assert_eq!(v1_records[0].text, "v1 question");
        assert_eq!(v1_records[0].project, "v1-project");
        assert_eq!(v1_records[0].links.event_id, None);
        assert_eq!(v1_records[0].links.parent_event_id, None);
        assert_eq!(v1_records[1].text, "compaction: v1 summary");

        let v2 = tmp.path().join("v2.jsonl");
        fs::write(
            &v2,
            concat!(
                "{\"type\":\"session\",\"version\":2,\"id\":\"v2-session\",\"timestamp\":\"2026-01-03T00:00:00Z\",\"cwd\":\"/tmp/v2-project\",\"parentSession\":\"v2-parent\"}\n",
                "{\"type\":\"message\",\"id\":\"u2\",\"parentId\":null,\"timestamp\":\"2026-01-03T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":\"v2 question\"}}\n",
                "{\"type\":\"message\",\"id\":\"h2\",\"parentId\":\"u2\",\"timestamp\":\"2026-01-03T00:00:02Z\",\"message\":{\"role\":\"hookMessage\",\"content\":\"legacy hook context\"}}\n"
            ),
        )
        .expect("write v2");
        let v2_records = parse_omp_records(&v2, 0, 0).expect("parse v2");
        assert_eq!(v2_records.len(), 2);
        assert_eq!(v2_records[1].role, "assistant");
        assert_eq!(v2_records[1].text, "hookMessage: legacy hook context");
        assert_eq!(v2_records[1].links.event_id.as_deref(), Some("h2"));
        assert_eq!(v2_records[1].links.parent_event_id.as_deref(), Some("u2"));
        assert_eq!(
            v2_records[1].links.parent_session_id.as_deref(),
            Some("v2-parent")
        );
        assert_eq!(v2_records[1].links.thread_source.as_deref(), Some("fork"));
        assert_eq!(
            v2_records[1].links.conversation_kind.as_deref(),
            Some("fork")
        );

        let v3 = tmp.path().join("v3.jsonl");
        fs::write(
            &v3,
            concat!(
                "{\"type\":\"title\",\"v\":1,\"title\":\"OMP fixture\",\"updatedAt\":\"2026-07-03T00:00:00Z\",\"pad\":\"\"}\n",
                "{\"type\":\"session\",\"version\":3,\"id\":\"v3-session\",\"timestamp\":\"2026-07-03T00:00:00Z\",\"cwd\":\"/tmp/v3-project\"}\n",
                "{\"type\":\"message\",\"id\":\"u3\",\"parentId\":null,\"timestamp\":\"2026-07-03T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"v3 question\"}]}}\n",
                "{\"type\":\"message\",\"id\":\"a3\",\"parentId\":\"u3\",\"timestamp\":\"2026-07-03T00:00:02Z\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"thinking\",\"thinking\":\"private reasoning\"},{\"type\":\"text\",\"text\":\"v3 answer\"},{\"type\":\"toolCall\",\"id\":\"tc3\",\"name\":\"Read\",\"arguments\":{\"path\":\"README.md\"}}]}}\n",
                "{\"type\":\"message\",\"id\":\"tr3\",\"parentId\":\"a3\",\"timestamp\":\"2026-07-03T00:00:03Z\",\"message\":{\"role\":\"toolResult\",\"toolCallId\":\"tc3\",\"toolName\":\"Read\",\"content\":[{\"type\":\"text\",\"text\":\"file contents\"}]}}\n",
                "{\"type\":\"custom_message\",\"id\":\"cm3\",\"parentId\":\"tr3\",\"timestamp\":\"2026-07-03T00:00:04Z\",\"customType\":\"steering\",\"content\":\"user steering\",\"display\":true,\"attribution\":\"user\"}\n",
                "{\"type\":\"compaction\",\"id\":\"c3\",\"parentId\":\"cm3\",\"timestamp\":\"2026-07-03T00:00:05Z\",\"summary\":\"compact summary\",\"firstKeptEntryId\":\"tr3\",\"tokensBefore\":100}\n",
                "{\"type\":\"branch_summary\",\"id\":\"br3\",\"parentId\":\"u3\",\"timestamp\":\"2026-07-03T00:00:06Z\",\"fromId\":\"c3\",\"summary\":\"branch summary\"}\n"
            ),
        )
        .expect("write v3");
        let v3_records = parse_omp_records(&v3, 0, 0).expect("parse v3");
        assert_eq!(v3_records.len(), 7);
        assert!(
            v3_records
                .iter()
                .all(|record| record.source == SourceKind::Omp)
        );
        assert_eq!(v3_records[1].role, "tool_use");
        assert_eq!(v3_records[1].tool_name.as_deref(), Some("Read"));
        assert_eq!(v3_records[1].links.event_id.as_deref(), Some("tc3"));
        assert_eq!(v3_records[1].links.parent_event_id.as_deref(), Some("a3"));
        assert_eq!(v3_records[2].role, "assistant");
        assert_eq!(v3_records[2].text, "v3 answer");
        assert!(!v3_records[2].text.contains("private reasoning"));
        assert_eq!(v3_records[3].role, "tool_result");
        assert_eq!(
            v3_records[3].links.parent_tool_use_id.as_deref(),
            Some("tc3")
        );
        assert_eq!(v3_records[4].role, "user");
        assert_eq!(
            v3_records[4].text,
            "custom_message(steering): user steering"
        );
        assert_eq!(
            v3_records[5].links.conversation_kind.as_deref(),
            Some("compaction")
        );
        assert_eq!(
            v3_records[6].links.logical_parent_event_id.as_deref(),
            Some("c3")
        );
        assert_eq!(
            v3_records[6].links.conversation_kind.as_deref(),
            Some("branch")
        );
    }

    #[test]
    fn parse_omp_incremental_records_reuses_title_slot_session_header() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let session_file = tmp.path().join("incremental.jsonl");
        let existing = concat!(
            "{\"type\":\"title\",\"v\":1,\"title\":\"Incremental\",\"updatedAt\":\"2026-07-03T00:00:00Z\",\"pad\":\"\"}\n",
            "{\"type\":\"session\",\"version\":3,\"id\":\"incremental-session\",\"timestamp\":\"2026-07-03T00:00:00Z\",\"cwd\":\"/home/alice/work/keel-project\"}\n",
            "{\"type\":\"message\",\"id\":\"u1\",\"parentId\":null,\"timestamp\":\"2026-07-03T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":\"first\"}}\n"
        );
        let appended = "{\"type\":\"message\",\"id\":\"a1\",\"parentId\":\"u1\",\"timestamp\":\"2026-07-03T00:00:02Z\",\"message\":{\"role\":\"assistant\",\"content\":\"second\"}}\n";
        fs::write(&session_file, format!("{existing}{appended}")).expect("write fixture");

        let records = parse_omp_records(&session_file, existing.len() as u64, 1)
            .expect("parse incremental OMP");

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].session_id, "incremental-session");
        assert_eq!(records[0].project, "keel-project");
        assert_eq!(records[0].turn_id, 1);
        assert_eq!(records[0].text, "second");
    }

    #[test]
    fn omp_incremental_guard_accepts_append_after_in_place_title_update() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let session_file = tmp.path().join("append.jsonl");
        let alpha_title = "{\"type\":\"title\",\"v\":1,\"title\":\"Alpha\",\"updatedAt\":\"2026-07-03T00:00:00Z\",\"pad\":\"\"}\n";
        let bravo_title = "{\"type\":\"title\",\"v\":1,\"title\":\"Bravo\",\"updatedAt\":\"2026-07-03T00:00:01Z\",\"pad\":\"\"}\n";
        assert_eq!(alpha_title.len(), bravo_title.len());
        let body = concat!(
            "{\"type\":\"session\",\"version\":3,\"id\":\"append-session\",\"timestamp\":\"2026-07-03T00:00:00Z\",\"cwd\":\"/tmp/project\"}\n",
            "{\"type\":\"message\",\"id\":\"u1\",\"parentId\":null,\"timestamp\":\"2026-07-03T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":\"first\"}}\n"
        );
        let existing = format!("{alpha_title}{body}");
        fs::write(&session_file, &existing).expect("write existing OMP session");
        let previous = omp_file_state(&session_file, 1);

        let appended = "{\"type\":\"message\",\"id\":\"a1\",\"parentId\":\"u1\",\"timestamp\":\"2026-07-03T00:00:02Z\",\"message\":{\"role\":\"assistant\",\"content\":\"second\"}}\n";
        fs::write(&session_file, format!("{bravo_title}{body}{appended}"))
            .expect("update title and append OMP entry");

        let mut state = IngestState::default();
        state
            .files
            .insert(session_file.to_string_lossy().to_string(), previous);
        let mut tasks = FileTaskSet::new(&state);
        tasks
            .add_paths([session_file], SourceKind::Omp)
            .expect("collect appended OMP session");

        assert_eq!(tasks.tasks.len(), 1);
        assert_eq!(tasks.tasks[0].offset, existing.len() as u64);
        assert_eq!(tasks.tasks[0].turn_id, 1);
        assert!(!tasks.tasks[0].delete_first);
    }

    #[test]
    fn omp_incremental_guard_reindexes_upstream_format_rewrite() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let session_file = tmp.path().join("migrated.jsonl");
        let legacy = concat!(
            "{\"type\":\"session\",\"version\":2,\"id\":\"migrated-session\",\"timestamp\":\"2026-01-03T00:00:00Z\",\"cwd\":\"/tmp/project\"}\n",
            "{\"type\":\"message\",\"id\":\"u1\",\"parentId\":null,\"timestamp\":\"2026-01-03T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":\"first\"}}\n"
        );
        fs::write(&session_file, legacy).expect("write legacy OMP session");
        let previous = omp_file_state(&session_file, 1);

        let migrated = concat!(
            "{\"type\":\"title\",\"v\":1,\"title\":\"Migrated\",\"updatedAt\":\"2026-07-03T00:00:00Z\",\"pad\":\"\"}\n",
            "{\"type\":\"session\",\"version\":3,\"id\":\"migrated-session\",\"timestamp\":\"2026-01-03T00:00:00Z\",\"cwd\":\"/tmp/project\"}\n",
            "{\"type\":\"message\",\"id\":\"u1\",\"parentId\":null,\"timestamp\":\"2026-01-03T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":\"first\"}}\n",
            "{\"type\":\"message\",\"id\":\"a1\",\"parentId\":\"u1\",\"timestamp\":\"2026-07-03T00:00:02Z\",\"message\":{\"role\":\"assistant\",\"content\":\"second\"}}\n"
        );
        fs::write(&session_file, migrated).expect("rewrite migrated OMP session");

        let mut state = IngestState::default();
        state
            .files
            .insert(session_file.to_string_lossy().to_string(), previous);
        let mut tasks = FileTaskSet::new(&state);
        tasks
            .add_paths([session_file], SourceKind::Omp)
            .expect("collect migrated OMP session");

        assert_eq!(tasks.tasks.len(), 1);
        assert_eq!(tasks.tasks[0].offset, 0);
        assert_eq!(tasks.tasks[0].turn_id, 0);
        assert!(tasks.tasks[0].delete_first);
    }

    fn omp_file_state(path: &Path, turn_id: u32) -> FileState {
        let file = File::open(path).expect("open OMP state fixture");
        let mmap = unsafe { Mmap::map(&file).expect("map OMP state fixture") };
        let session = read_omp_session_header(&mmap, path).expect("read OMP state header");
        FileState {
            size: mmap.len() as u64,
            mtime: 0,
            offset: mmap.len() as u64,
            turn_id,
            source_state: Some(session.source_file_state(&mmap, mmap.len() as u64)),
        }
    }

    #[test]
    fn parse_omp_nested_subagent_links_to_parent_session_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let parent_id = "11111111-1111-1111-1111-111111111111";
        let parent_stem = format!("20260703T010203Z_{parent_id}");
        let parent_file = tmp.path().join(format!("{parent_stem}.jsonl"));
        fs::write(
            &parent_file,
            format!(
                "{{\"type\":\"session\",\"version\":3,\"id\":\"{parent_id}\",\"timestamp\":\"2026-07-03T00:00:00Z\",\"cwd\":\"/tmp/project\"}}\n"
            ),
        )
        .expect("write parent");
        let child_dir = tmp.path().join(parent_stem);
        fs::create_dir_all(&child_dir).expect("create child dir");
        let child_file =
            child_dir.join("20260703T010204Z_22222222-2222-2222-2222-222222222222.jsonl");
        fs::write(
            &child_file,
            concat!(
                "{\"type\":\"session\",\"version\":3,\"id\":\"22222222-2222-2222-2222-222222222222\",\"timestamp\":\"2026-07-03T00:00:00Z\",\"cwd\":\"/tmp/project\"}\n",
                "{\"type\":\"message\",\"id\":\"u1\",\"parentId\":null,\"timestamp\":\"2026-07-03T00:00:01Z\",\"message\":{\"role\":\"user\",\"content\":\"subagent task\"}}\n"
            ),
        )
        .expect("write child");

        let records = parse_omp_records(&child_file, 0, 0).expect("parse child");

        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].links.parent_session_id.as_deref(),
            Some(parent_id)
        );
        assert_eq!(records[0].links.thread_source.as_deref(), Some("subagent"));
        assert_eq!(
            records[0].links.conversation_kind.as_deref(),
            Some("subagent")
        );
    }

    #[test]
    fn parse_omp_rejects_unknown_future_session_versions() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let session_file = tmp.path().join("future.jsonl");
        fs::write(
            &session_file,
            "{\"type\":\"session\",\"version\":4,\"id\":\"future\",\"timestamp\":\"2026-07-03T00:00:00Z\",\"cwd\":\"/tmp/project\"}\n",
        )
        .expect("write future fixture");

        let error = parse_omp_records(&session_file, 0, 0).expect_err("reject future version");

        assert!(
            error
                .to_string()
                .contains("unsupported OMP session version 4")
        );
    }

    #[test]
    fn collect_copilot_files_finds_session_events() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let session_dir = tmp
            .path()
            .join("session-state")
            .join("11111111-1111-4111-8111-111111111111");
        fs::create_dir_all(&session_dir).expect("create session dir");

        let events = session_dir.join("events.jsonl");
        let ignored = session_dir.join("workspace.yaml");
        fs::write(&events, "{}\n").expect("write events");
        fs::write(&ignored, "cwd: /tmp/project\n").expect("write workspace");

        let files = collect_copilot_files_from_root(&tmp.path().join("session-state"))
            .expect("collect copilot sessions");

        assert_eq!(files, vec![events]);
    }

    #[test]
    fn parse_copilot_session_extracts_messages_tools_and_workspace() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let session_id = "11111111-1111-4111-8111-111111111111";
        let session_dir = tmp.path().join("session-state").join(session_id);
        fs::create_dir_all(&session_dir).expect("create session dir");
        fs::write(
            session_dir.join("workspace.yaml"),
            "cwd: /Users/nico/Code/memex\ngitRoot: /Users/nico/Code/memex\nrepository: nicosuave/memex\nbranch: main\n",
        )
        .expect("write workspace");
        let events = session_dir.join("events.jsonl");
        fs::write(
            &events,
            concat!(
                "{\"type\":\"session.start\",\"timestamp\":\"2026-06-01T12:00:00Z\",\"data\":{\"sessionId\":\"11111111-1111-4111-8111-111111111111\",\"context\":{\"cwd\":\"/Users/nico/Code/memex\",\"repository\":\"nicosuave/memex\"}}}\n",
                "{\"type\":\"user.message\",\"timestamp\":\"2026-06-01T12:00:01Z\",\"data\":{\"content\":\"Find the parser\"}}\n",
                "{\"type\":\"assistant.message\",\"timestamp\":\"2026-06-01T12:00:02Z\",\"data\":{\"content\":\"I will inspect ingestion.\"}}\n",
                "{\"type\":\"tool.execution_start\",\"timestamp\":\"2026-06-01T12:00:03Z\",\"data\":{\"toolCallId\":\"call-1\",\"toolName\":\"grep\",\"arguments\":{\"pattern\":\"parse_copilot\"}}}\n",
                "{\"type\":\"tool.execution_complete\",\"timestamp\":\"2026-06-01T12:00:04Z\",\"data\":{\"toolCallId\":\"call-1\",\"success\":true,\"result\":{\"content\":\"src/ingest.rs\"}}}\n"
            ),
        )
        .expect("write events");
        let meta = events.metadata().expect("metadata");
        let task = FileTask {
            path: events.clone(),
            source: SourceKind::Copilot,
            offset: 0,
            turn_id: 0,
            size: meta.len(),
            mtime: 0,
            delete_first: false,
        };
        let (raw_tx_record, rx_record) = unbounded();
        let tx_record = RecordSender::new(raw_tx_record, IndexedToolContentLimits::default());
        let (tx_update, rx_update) = unbounded();
        let next_doc_id = AtomicU64::new(1);
        let mut byte_totals = [0; SOURCE_COUNT];
        byte_totals[SourceKind::Copilot.idx()] = meta.len();
        let mut file_totals = [0; SOURCE_COUNT];
        file_totals[SourceKind::Copilot.idx()] = 1;
        let progress = Arc::new(Progress::new(byte_totals, file_totals, false));

        parse_copilot_session(&task, &tx_record, &tx_update, &next_doc_id, &progress)
            .expect("parse copilot session");
        drop(tx_record);
        drop(tx_update);

        let records: Vec<Record> = rx_record.try_iter().collect();
        assert_eq!(records.len(), 4);
        assert!(records.iter().all(|r| r.source == SourceKind::Copilot));
        assert!(records.iter().all(|r| r.project == "memex"));
        assert!(records.iter().all(|r| r.session_id == session_id));
        assert_eq!(records[0].role, "user");
        assert_eq!(
            records[0].links.event_id.as_deref(),
            Some("11111111-1111-4111-8111-111111111111:0")
        );
        assert_eq!(records[0].links.conversation_kind.as_deref(), Some("main"));
        assert_eq!(records[1].role, "assistant");
        assert_eq!(
            records[1].links.event_id.as_deref(),
            Some("11111111-1111-4111-8111-111111111111:1")
        );
        assert_eq!(records[2].role, "tool_use");
        assert_eq!(records[2].tool_name.as_deref(), Some("grep"));
        assert!(records[2].text.contains("parse_copilot"));
        assert_eq!(records[2].links.event_id.as_deref(), Some("call-1"));
        assert_eq!(records[3].role, "tool_result");
        assert_eq!(records[3].tool_name.as_deref(), Some("grep"));
        assert_eq!(records[3].tool_output.as_deref(), Some("src/ingest.rs"));
        assert_eq!(records[3].links.parent_event_id.as_deref(), Some("call-1"));
        assert_eq!(
            records[3].links.parent_tool_use_id.as_deref(),
            Some("call-1")
        );

        let update = rx_update.try_recv().expect("file update");
        assert_eq!(update.state.offset, meta.len());
        assert_eq!(update.state.turn_id, 4);
    }

    #[test]
    fn writer_loop_accepts_copilot_source_progress() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let index_dir = tmp.path().join("index");
        let vector_dir = tmp.path().join("vectors");
        fs::create_dir_all(&index_dir).expect("create index dir");
        fs::create_dir_all(&vector_dir).expect("create vector dir");
        let index = SearchIndex::open_or_create(&index_dir).expect("open index");
        let (tx_record, rx_record) = unbounded();
        tx_record
            .send(Record {
                source: SourceKind::Copilot,
                doc_id: 1,
                ts: 1_780_291_200_000,
                project: "memex".to_string(),
                session_id: "11111111-1111-4111-8111-111111111111".to_string(),
                turn_id: 0,
                role: "user".to_string(),
                text: "Find the parser".to_string(),
                tool_name: None,
                tool_input: None,
                tool_output: None,
                links: RecordLinks::default(),
                source_path: tmp
                    .path()
                    .join(
                        ".copilot/session-state/11111111-1111-4111-8111-111111111111/events.jsonl",
                    )
                    .to_string_lossy()
                    .to_string(),
            })
            .expect("send record");
        drop(tx_record);

        let progress = Arc::new(Progress::new([0; SOURCE_COUNT], [0; SOURCE_COUNT], false));
        let ctx = WriterContext {
            embeddings: false,
            do_backfill_embeddings: false,
            vector_dir,
            analytics_path: tmp.path().join("state").join("analytics.sqlite"),
            progress,
            model: ModelChoice::default(),
            embed_runtime: EmbedRuntimeConfig::default(),
            tool_content_limits: IndexedToolContentLimits::default(),
        };

        let (records_added, records_embedded) =
            writer_loop(index, rx_record, Vec::new(), ctx).expect("write copilot record");

        assert_eq!(records_added, 1);
        assert_eq!(records_embedded, 0);
    }
}
