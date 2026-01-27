/**
 * security.js
 * Implements low-level security features and Easter eggs.
 */

// Disable Right Click
document.addEventListener('contextmenu', function (e) {
    e.preventDefault();
});

// Disable F12, Ctrl+Shift+I, Ctrl+Shift+J, Ctrl+U
document.onkeydown = function (e) {
    if (e.keyCode == 123) {
        return false;
    }
    if (e.ctrlKey && e.shiftKey && (e.keyCode == 'I'.charCodeAt(0) || e.keyCode == 'J'.charCodeAt(0))) {
        return false;
    }
    if (e.ctrlKey && e.keyCode == 'U'.charCodeAt(0)) {
        return false;
    }
}

// Easter Egg - Console Warning
console.log("%cStop snooping around!", "color: red; font-family: sans-serif; font-size: 4.5em; font-weight: bolder; text-shadow: #000 1px 1px;");
console.log("%cThis is a project by Amey Thakur & Mega Satish.", "color: #1daeff; font-family: sans-serif; font-size: 1.5em;");
