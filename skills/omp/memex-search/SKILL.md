---
name: memex-search
description: Search, filter, and retrieve OMP history via memex CLI. Use for context resumption, finding past code or decisions, and self-correction based on history.
---

# Memex for OMP

Use `memex` as the primary history-retrieval tool for OMP sessions.

## Indexing

- Index the default OMP session store: `memex index`
- Skip OMP: `memex index --no-omp`
- Index a custom OMP `--session-dir`: `memex index --omp-source <path>`
- OMP's default, `OMP_PROFILE`/legacy `PI_PROFILE`, `PI_CONFIG_DIR`, and migrated XDG data layouts are resolved automatically.
- For OMP selected through the Pi-shared `PI_CODING_AGENT_DIR`, use `memex index --omp-source <path>/sessions --no-pi` to make attribution explicit.

## Searching

| Need | Command |
| --- | --- |
| Exact terms or IDs | `memex search "term" --source omp` |
| Concepts or intent | `memex search "concept" --source omp --semantic` |
| Mixed exact and fuzzy | `memex search "term concept" --source omp --hybrid` |
| One hit per session | `memex search "topic" --source omp --unique-session` |

Use `--session <session_id>` to narrow a search and `memex session <session_id>` to fetch the complete indexed transcript.

## Output

Results are JSONL by default. Important fields include:

- `doc_id`, `ts`, `session_id`, `project`, `role`, `source`, `source_path`, and `text`
- `event_id` and `parent_event_id` for OMP v2/v3 entry trees
- `logical_parent_event_id` for branch summaries
- `parent_session_id`, `thread_source`, and `conversation_kind` for forks and nested subagents
- `parent_tool_use_id` for tool results

OMP v1 has no entry-tree IDs, so those fields are absent rather than synthesized. Full-file OMP migrations are detected before incremental indexing. Thinking and image payloads are not indexed; user/assistant text, tool calls/results, custom messages, compactions, and branch summaries are indexed.

## Resume

The TUI resumes OMP sessions with `omp --resume {session_id}` when `omp` is available. Override it in `~/.memex/config.toml` with:

```toml
omp_resume_cmd = "omp --resume {session_id}"
```

Resume templates may also use `{project}`, `{source}`, `{source_path}`, `{source_dir}`, `{cwd}`, and their shell-quoted variants.
