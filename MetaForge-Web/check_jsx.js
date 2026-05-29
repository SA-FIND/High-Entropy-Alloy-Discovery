const fs = require('fs');
const babel = require('@babel/parser');
const html = fs.readFileSync('templates/index.html', 'utf8');
const match = html.match(/<script type="text\/babel">([\s\S]*?)<\/script>/);
if (match) {
    let code = match[1];
    code = code.replace(/{% raw %}/g, '').replace(/{% endraw %}/g, '');
    try {
        babel.parse(code, { sourceType: 'module', plugins: ['jsx'] });
        console.log('Syntax OK');
    } catch (e) {
        console.error('Syntax Error:', e.message);
        console.log('Line:', e.loc.line, 'Column:', e.loc.column);
        console.log(code.split('\n')[e.loc.line - 1]);
    }
} else {
    console.log('No script found');
}
