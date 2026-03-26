const vscode = require('vscode');
const fetch = require('node-fetch');

let debounceTimeout = null;
let debounceResolve = null;
let requestCounter = 0;
let retryTimer = null;
let extensionContext = null;
let setConnected = null;

function activate(context) {
    extensionContext = context;

    // --- Status Bar ---
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    context.subscriptions.push(statusBar);
    context.subscriptions.push({ dispose: () => { if (retryTimer) clearInterval(retryTimer); } });

    setConnected = (connected) => {
        if (connected) {
            statusBar.text = '$(circle-filled) Rizz-V Active';
            statusBar.color = new vscode.ThemeColor('charts.green');
            statusBar.tooltip = 'Rizz-V: Backend connected';
            if (retryTimer) { clearInterval(retryTimer); retryTimer = null; }
            flushRatingQueue();
        } else {
            statusBar.text = '$(circle-slash) Rizz-V Disconnected';
            statusBar.color = new vscode.ThemeColor('errorForeground');
            statusBar.tooltip = 'Rizz-V: Backend unreachable — retrying every 5s';
            if (!retryTimer) {
                retryTimer = setInterval(async () => {
                    const ok = await pingBackend();
                    if (ok) setConnected(true);
                }, 5000);
            }
        }
        statusBar.show();
    };

    setConnected(true);

    // --- Inline Completion Provider ---
    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider('riscv', {
            async provideInlineCompletionItems(document, position, ctx, token) {
                const currentLine = document.lineAt(position.line).text.trim();

                // UC-1 alt 2a: suppress on comments (#) or assembler directives (.)
                if (currentLine.startsWith('#') || currentLine.startsWith('.')) {
                    return { items: [] };
                }

                // prefix: lines above cursor (up to 40 lines back)
                const startLine = Math.max(0, position.line - 40);
                const prefix = document.getText(
                    new vscode.Range(new vscode.Position(startLine, 0), position)
                ).trimStart();

                if (!prefix) return { items: [] };

                // suffix: lines below cursor (up to 20 lines ahead)
                const endLine = Math.min(document.lineCount - 1, position.line + 20);
                const suffix = document.getText(
                    new vscode.Range(position, new vscode.Position(endLine, document.lineAt(endLine).text.length))
                ).trimEnd();

                // UC-2: detect comment-to-code — previous line is a descriptive comment
                const prevLine = position.line > 0
                    ? document.lineAt(position.line - 1).text.trim()
                    : '';
                const isCommentToCode = prevLine.startsWith('#') && prevLine.length > 1;
                const suggestionType = isCommentToCode ? 'comment-to-code' : 'realtime';
                const maxTokens = isCommentToCode ? 150 : 50;

                // Stale request detection: discard response if user kept typing
                const myId = ++requestCounter;

                try {
                    const suggestion = await debouncedSuggestion(prefix, suffix, maxTokens);

                    if (requestCounter !== myId) return { items: [] };
                    if (!suggestion || suggestion.trim() === '') return { items: [] };

                    const lineNumber = position.line + 1;
                    const item = new vscode.InlineCompletionItem(
                        suggestion,
                        new vscode.Range(position, position)
                    );
                    item.command = {
                        command: 'rizz-v.suggestionAccepted',
                        title: 'Suggestion Accepted',
                        arguments: [prefix, suffix, suggestion, lineNumber, suggestionType]
                    };

                    return { items: [item] };

                } catch (error) {
                    console.error('Rizz-V:', error);
                    setConnected(false);
                    return { items: [] };
                }
            }
        })
    );

    // --- Accept Command ---
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'rizz-v.suggestionAccepted',
            async (prefix, suffix, suggestion, lineNumber, suggestionType) => {
                const choice = await vscode.window.showInformationMessage(
                    `Rizz-V suggestion accepted on line ${lineNumber} — Was this helpful?`,
                    '👍 Helpful',
                    '👎 Not helpful'
                );

                const rating = choice === '👍 Helpful' ? 1 : choice === '👎 Not helpful' ? 0 : null;

                queueRating({
                    prefix,
                    suffix,
                    suggestion,
                    suggestion_type: suggestionType,
                    accepted: true,
                    rating,
                    timestamp: new Date().toISOString()
                });

                if (rating !== null) {
                    vscode.window.showInformationMessage('Thanks for your feedback!');
                }
            }
        )
    );
}

// --- Rating Queue (local storage + backend sync) ---

function queueRating(data) {
    sendRatingToBackend(data).catch(() => {
        // UC-3 exc 8a: backend unreachable — store locally and retry when reconnected
        const queue = extensionContext.globalState.get('ratingQueue', []);
        queue.push(data);
        extensionContext.globalState.update('ratingQueue', queue);
    });
}

async function flushRatingQueue() {
    const queue = extensionContext.globalState.get('ratingQueue', []);
    if (queue.length === 0) return;
    const failed = [];
    for (const item of queue) {
        try {
            await sendRatingToBackend(item);
        } catch {
            failed.push(item);
        }
    }
    extensionContext.globalState.update('ratingQueue', failed);
}

async function sendRatingToBackend(data) {
    const response = await fetch('http://127.0.0.1:8000/rating', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
}

async function pingBackend() {
    try {
        const res = await fetch('http://127.0.0.1:8000/');
        return res.ok;
    } catch {
        return false;
    }
}

// --- Debounce ---

function debouncedSuggestion(prefix, suffix, maxTokens) {
    // Resolve the previous hanging promise immediately so it doesn't leak
    if (debounceResolve) {
        debounceResolve(null);
        debounceResolve = null;
    }
    clearTimeout(debounceTimeout);

    return new Promise((resolve, reject) => {
        debounceResolve = resolve;
        debounceTimeout = setTimeout(async () => {
            debounceResolve = null;
            try {
                resolve(await getAssemblySuggestion(prefix, suffix, maxTokens));
            } catch (e) {
                reject(e);
            }
        }, 500);
    });
}

// --- API ---

async function getAssemblySuggestion(prefix, suffix = '', maxTokens = 50) {
    const response = await fetch('http://127.0.0.1:8000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prefix, suffix, max_new_tokens: maxTokens })
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const data = await response.json();
    const generatedCode = data.generated_code.trim();
    if (generatedCode.startsWith(prefix)) {
        return generatedCode.substring(prefix.length).trim();
    }
    return generatedCode;
}

function deactivate() {}

module.exports = { activate, deactivate };
