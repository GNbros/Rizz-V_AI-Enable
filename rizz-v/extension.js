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
    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider('riscv', {
            async provideInlineCompletionItems(document, position, context, token) {

                const startLine = Math.max(0, position.line - 40);

                const textBefore = document.getText(
                    new vscode.Range(
                        new vscode.Position(startLine, 0),
                        position
                    )
                );

                if (!textBefore.trim()) {
                    return { items: [] };
                }

                try {
                    const suggestion = await debouncedSuggestion(textBefore);

                    if (!suggestion || suggestion.trim() === "") {
                        return { items: [] };
                    }

                    // Store for rating
                    await vscode.commands.executeCommand(
                        'rizz-v.rateSuggestion',
                        textBefore,
                        suggestion
                    );

                    return {
                        items: [
                            new vscode.InlineCompletionItem(
                                suggestion,
                                new vscode.Range(position, position)
                            )
                        ]
                    };

                } catch (error) {
                    console.error(error);
                    return { items: [] };
                }
            }
        })
    );


    context.subscriptions.push(
        vscode.commands.registerCommand('rizz-v.rateSuggestion', async (prompt, suggestion) => {
            // Store values in global state or context
            context.workspaceState.update('lastPrompt', prompt);
            context.workspaceState.update('lastSuggestion', suggestion);

            ratingStatusBar.show();
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('rizz-v.promptRating', async () => {
            const prompt = context.workspaceState.get('lastPrompt');
            const suggestion = context.workspaceState.get('lastSuggestion');

            if (!prompt || !suggestion) {
                vscode.window.showWarningMessage("No suggestion to rate yet.");
                return;
            }

            const rating = await vscode.window.showQuickPick(['1', '2', '3', '4', '5'], {
                placeHolder: '⭐ Rate the suggestion (1-5)',
            });

            if (rating) {
                await sendRating(prompt, suggestion, parseInt(rating));
                vscode.window.showInformationMessage('🎉 Thanks for your feedback!');
                ratingStatusBar.hide();
            }
        })
    );

    

}

let timeout = null;

async function debouncedSuggestion(prompt) {
    return new Promise((resolve) => {
        clearTimeout(timeout);
        timeout = setTimeout(async () => {
            const result = await getAssemblySuggestion(prompt);
            resolve(result);
        }, 400); // 400ms delay
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
                max_new_tokens: 50
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        // Assuming the response is JSON and contains a field "generated_code"

        const data = await response.json(); // or response.json() if structured
        // remove promt that include in generated_code
        const generatedCode = data.generated_code.trim();

        if (generatedCode.startsWith(prompt)) {
            return generatedCode.substring(prompt.length).trim();
        }

        return generatedCode;

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