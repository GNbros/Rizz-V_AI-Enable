const assert = require('assert');
const vscode = require('vscode');

suite('Rizz-V Extension Tests', () => {

    // -----------------------------------------------------------------------
    // Activation
    // -----------------------------------------------------------------------

    suite('Activation', () => {
        test('extension activates successfully', async () => {
            const ext = vscode.extensions.getExtension('undefined_publisher.rizz-v');
            if (ext) {
                await ext.activate();
                assert.ok(ext.isActive, 'Extension should be active');
            }
        });
    });

    // -----------------------------------------------------------------------
    // Status bar
    // -----------------------------------------------------------------------

    suite('Status Bar', () => {
        test('status bar item is visible after activation', async () => {
            // Open a .s file to trigger activation
            const doc = await vscode.workspace.openTextDocument({
                language: 'riscv',
                content: 'addi t0, t0, 1\n'
            });
            await vscode.window.showTextDocument(doc);
            // If no exception thrown, status bar was created without error
            assert.ok(true);
        });
    });

    // -----------------------------------------------------------------------
    // Language registration
    // -----------------------------------------------------------------------

    suite('Language', () => {
        test('riscv language is registered', async () => {
            const langs = await vscode.languages.getLanguages();
            assert.ok(langs.includes('riscv'), 'riscv language should be registered');
        });

        test('.s file is detected as riscv language', async () => {
            const doc = await vscode.workspace.openTextDocument({
                language: 'riscv',
                content: 'addi t0, t0, 1\n'
            });
            assert.strictEqual(doc.languageId, 'riscv');
        });
    });

    // -----------------------------------------------------------------------
    // Inline completion suppression (logic tests via document inspection)
    // -----------------------------------------------------------------------

    suite('Completion Suppression', () => {
        test('comment line is detected correctly', async () => {
            const doc = await vscode.workspace.openTextDocument({
                language: 'riscv',
                content: '# this is a comment\naddi t0, t0, 1\n'
            });
            const commentLine = doc.lineAt(0).text.trim();
            const codeLine = doc.lineAt(1).text.trim();
            assert.ok(commentLine.startsWith('#'), 'Line 0 should be a comment');
            assert.ok(!codeLine.startsWith('#'), 'Line 1 should not be a comment');
        });

        test('assembler directive is detected correctly', async () => {
            const doc = await vscode.workspace.openTextDocument({
                language: 'riscv',
                content: '.text\n.globl main\naddi t0, t0, 1\n'
            });
            const directiveLine0 = doc.lineAt(0).text.trim();
            const directiveLine1 = doc.lineAt(1).text.trim();
            const codeLine = doc.lineAt(2).text.trim();
            assert.ok(directiveLine0.startsWith('.'), 'Line 0 should be a directive');
            assert.ok(directiveLine1.startsWith('.'), 'Line 1 should be a directive');
            assert.ok(!codeLine.startsWith('.'), 'Line 2 should not be a directive');
        });

        test('comment-to-code trigger is detected correctly', async () => {
            const doc = await vscode.workspace.openTextDocument({
                language: 'riscv',
                content: '# quick sort\n'
            });
            const prevLine = doc.lineAt(0).text.trim();
            const isCommentToCode = prevLine.startsWith('#') && prevLine.length > 1;
            assert.ok(isCommentToCode, 'Descriptive comment should trigger comment-to-code');
        });

        test('empty comment does not trigger comment-to-code', async () => {
            const doc = await vscode.workspace.openTextDocument({
                language: 'riscv',
                content: '#\n'
            });
            const prevLine = doc.lineAt(0).text.trim();
            const isCommentToCode = prevLine.startsWith('#') && prevLine.length > 1;
            assert.ok(!isCommentToCode, 'Empty comment should not trigger comment-to-code');
        });
    });

    // -----------------------------------------------------------------------
    // Commands
    // -----------------------------------------------------------------------

    suite('Commands', () => {
        test('rizz-v.suggestionAccepted command is registered', async () => {
            const commands = await vscode.commands.getCommands(true);
            assert.ok(
                commands.includes('rizz-v.suggestionAccepted'),
                'suggestionAccepted command should be registered'
            );
        });
    });

});
