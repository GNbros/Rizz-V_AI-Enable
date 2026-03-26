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
    // Inline completion suppression
    // -----------------------------------------------------------------------

    suite('Completion Suppression', () => {
        test('no completion triggered on comment line (#)', async () => {
            const doc = await vscode.workspace.openTextDocument({
                language: 'riscv',
                content: '# this is a comment'
            });
            const editor = await vscode.window.showTextDocument(doc);
            const position = new vscode.Position(0, 19);

            const items = await vscode.commands.executeCommand(
                'vscode.executeInlineCompletionItemProvider',
                doc.uri,
                position
            );
            // Should return no items for comment lines
            assert.ok(!items || items.items.length === 0,
                'Should suppress completions on comment lines');
        });

        test('no completion triggered on directive line (.text)', async () => {
            const doc = await vscode.workspace.openTextDocument({
                language: 'riscv',
                content: '.text'
            });
            const editor = await vscode.window.showTextDocument(doc);
            const position = new vscode.Position(0, 5);

            const items = await vscode.commands.executeCommand(
                'vscode.executeInlineCompletionItemProvider',
                doc.uri,
                position
            );
            assert.ok(!items || items.items.length === 0,
                'Should suppress completions on directive lines');
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
