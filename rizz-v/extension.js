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
    const ratingStatusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    ratingStatusBar.text = '⭐ Rate Suggestion';
    ratingStatusBar.tooltip = 'Click to rate the last suggestion';
    ratingStatusBar.command = 'rizz-v.promptRating';
    context.subscriptions.push(ratingStatusBar);

    // Register the completion item provider for assembly language
    vscode.languages.registerCompletionItemProvider('riscv', {
        async provideCompletionItems(document, position, token, context) {
            const textBefore = document.getText(new vscode.Range(new vscode.Position(position.line, 0), position));
            
            try {
                const suggestion = await getAssemblySuggestion(textBefore);

                
                const completionItem = new vscode.CompletionItem(suggestion, vscode.CompletionItemKind.Snippet);
                completionItem.insertText = suggestion;

                completionItem.command = {
                    command: 'rizz-v.rateSuggestion',
                    title: 'Rate Suggestion',
                    arguments: [textBefore, suggestion]  // Pass original prompt
                };

                return [completionItem];

            } catch (error) {
                // console.error('Error fetching assembly suggestion:', error);
                const errorItem = new vscode.CompletionItem("Rizz-V not Working", vscode.CompletionItemKind.Text);
                errorItem.insertText = '';
                return [errorItem];
            }

        }

        
    }, ' ',);

    context.subscriptions.push(
        vscode.commands.registerCommand('rizz-v.rateSuggestion', async (prompt, suggestion) => {
            // Store values in global state or context
            context.workspaceState.update('lastPrompt', prompt);
            context.workspaceState.update('lastSuggestion', suggestion);

            ratingStatusBar.show();
        })
    );
    vscode.commands.registerCommand('rizz-v.promptRating', async () => {
            const prompt = context.workspaceState.get('lastPrompt');
            const suggestion = context.workspaceState.get('lastSuggestion');

            const rating = await vscode.window.showQuickPick(['1', '2', '3', '4', '5'], {
                placeHolder: '⭐ Rate the suggestion (1-5)',
            });

            if (rating) {
                await sendRating(prompt, suggestion, parseInt(rating));
                vscode.window.showInformationMessage('🎉 Thanks for your feedback!');
                
                ratingStatusBar.hide();
            }
    });

    

}

// Function to send the rating to your model API
async function sendRating(prompt, suggestion, rating) {
    try {
        const response = await fetch('http://127.0.0.1:8000/rating', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                prompt: prompt,
                suggestion: suggestion,
                rating: rating }),
        });

        if (!response.ok) {
            throw new Error(`Failed to submit rating. Status: ${response.status}`);
        }

        console.log('Rating submitted successfully.');
    } catch (error) {
        console.error('Error submitting rating:', error);
    }
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

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        // Assuming the response is JSON and contains a field "generated_code"

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
        return 'Error fetching suggestion service'; // Return error message as fallback
    }
}

// This method is called when your extension is deactivated
function deactivate() {}

module.exports = {
    activate,
    deactivate
}
