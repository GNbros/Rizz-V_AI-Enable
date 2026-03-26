# Test Coverage

## Backend (21 tests) — `pytest tests/`

### Health Check
| Test | What it checks |
|---|---|
| `test_root_returns_200` | Server is up |
| `test_root_message` | Response says "running" |

### `/generate` endpoint
| Test | What it checks |
|---|---|
| `test_basic_suggestion_returned` | Valid request returns `generated_code` |
| `test_fim_includes_suffix` | Prompt sent to model has `<fim_prefix>`, `<fim_suffix>`, `<fim_middle>` and suffix text |
| `test_empty_prefix_returns_error` | Empty prefix → error |
| `test_prefix_too_long_returns_error` | Prefix > 512 chars → error |
| `test_zero_max_new_tokens_returns_error` | `max_new_tokens=0` → error |
| `test_missing_prefix_field_returns_422` | Missing `prefix` field → HTTP 422 |
| `test_custom_max_tokens_passed_through` | `max_new_tokens` value is forwarded to model |

### `/rating` endpoint
| Test | What it checks |
|---|---|
| `test_helpful_rating_saved` | Rating `1` saved to DB |
| `test_unhelpful_rating_saved` | Rating `0` saved to DB |
| `test_null_rating_allowed` | Rating `null` (no feedback given) saved as NULL |
| `test_invalid_rating_value_returns_error` | Rating `3` → error |
| `test_empty_prefix_returns_error` | Empty prefix → error |
| `test_empty_suggestion_returns_error` | Empty suggestion → error |
| `test_invalid_suggestion_type_returns_error` | Unknown type → error |
| `test_comment_to_code_type_accepted` | `suggestion_type: "comment-to-code"` is valid |
| `test_prefix_and_suffix_stored_in_db` | prefix and suffix both saved correctly |
| `test_missing_prefix_field_returns_422` | Missing `prefix` field → HTTP 422 |

### Database
| Test | What it checks |
|---|---|
| `test_init_db_creates_table` | `rating` table is created on fresh DB |
| `test_migration_adds_missing_columns` | Old schema DB gets new columns without crashing |

---

## Extension (7 tests) — `npm test`

### Activation
| Test | What it checks |
|---|---|
| `extension activates successfully` | Extension loads without error |

### Status Bar
| Test | What it checks |
|---|---|
| `status bar item is visible after activation` | Status bar created when riscv file opens |

### Language
| Test | What it checks |
|---|---|
| `riscv language is registered` | `riscv` appears in VS Code language list |
| `.s file is detected as riscv language` | File has correct `languageId` |

### Completion Suppression
| Test | What it checks |
|---|---|
| `comment line is detected correctly` | `#` line is a comment, code line is not |
| `assembler directive is detected correctly` | `.text` line is a directive, code line is not |
| `comment-to-code trigger detected` | `# quick sort` triggers comment-to-code mode |
| `empty comment does not trigger comment-to-code` | `#` alone does not trigger comment-to-code |

### Commands
| Test | What it checks |
|---|---|
| `suggestionAccepted command is registered` | Accept command exists and is callable |
