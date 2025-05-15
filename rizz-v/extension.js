// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
const vscode = require('vscode');
const fetch = require('node-fetch');

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed

/**	
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {

    // Use the console to output diagnostic information (console.log) and errors (console.error)
    // This line of code will only be executed once when your extension is activated
    console.log('Congratulations, your extension "rizz-v" is now active!');

    // Register the "Hello World" command
    const disposable = vscode.commands.registerCommand('rizz-v.helloWorld', function () {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            const document = editor.document;
            const position = editor.selection.active;
            var textBefore = document.getText(new vscode.Range(new vscode.Position(position.line, 0), position));
        }

        // Display a message box to the user
        vscode.window.showInformationMessage('Hello World from rizz-v!' + textBefore);
    });

    // Register the completion item provider for assembly language
    vscode.languages.registerCompletionItemProvider('riscv', {
        async provideCompletionItems(document, position) {
            const textBefore = document.getText(new vscode.Range(new vscode.Position(position.line, 0), position));
            
            try {
                const suggestion = await getAssemblySuggestion(textBefore);

            const completionItem = new vscode.CompletionItem(suggestion, vscode.CompletionItemKind.Snippet);
            completionItem.insertText = suggestion;

            return [completionItem];
            } catch (error) {
                console.error('Error fetching assembly suggestion:', error);
                return []; // Return an empty array if there's an error
            }
        }
    }, ' ',);

    context.subscriptions.push(disposable);
}

// Function to get assembly code suggestions from your model API
async function getAssemblySuggestion(prompt) {
    try {
        console.log('Sending request with prompt:', prompt);
        const response = await fetch('http://127.0.0.1:8000/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                max_new_tokens: 30
            }),
        });

        const data = await response.json(); // or response.json() if structured
        // remove promt that include in generated_code
        const generatedCode = data["generated_code"];
        const promptIndex = generatedCode.indexOf(prompt);
        if (promptIndex !== -1) {
            return generatedCode.slice(promptIndex + prompt.length).trim();
        }
        return data["generated_code"];
    } catch (error) {
        console.error('Error fetching assembly suggestion:', error);
        return 'Error fetching suggestion'; // Return error message as fallback
    }
}

// This method is called when your extension is deactivated
function deactivate() {}

module.exports = {
    activate,
    deactivate
}
