
/* ******************************************
* In http 1.1, Web Browsers enforce a limit *
* on number of concurrent SSE connections   *
* per domain (usually 5).                   *
*                                           *
* Even though retrain-pipelines' WebConsole *
* uses multiplexed SSE for each page,       *
* there's still a limit in how many         *
* WebConsole tabs can be opened.            *
*                                           *
* To circonvent that, use http 2            *
* (which requires https and certificates    *
*  signed by a trusted authority).          *
****************************************** */


// Broadcast channel to communicate between tabs
const MAX_COUNT = 5;
const channel = new BroadcastChannel('tab-counter');
let tabId = Math.random().toString(36).substr(2, 9);
let allTabs = new Set([tabId]);
let hasAlerted = false;

// Persist tab count in localStorage
function saveTabCount() {
    localStorage.setItem('tabCount', JSON.stringify([...allTabs]));
}

function loadTabCount() {
    const saved = localStorage.getItem('tabCount');
    if (saved) {
        const savedTabs = JSON.parse(saved);
        // Only keep tabs that are still alive
        allTabs = new Set([tabId]);
        savedTabs.forEach(id => allTabs.add(id));
    }
}

// Load existing count on startup
loadTabCount();

function checkTabLimit() {
    const isHttp = window.location.protocol === 'http:';

    if (isHttp && allTabs.size > MAX_COUNT && !hasAlerted) {
        hasAlerted = true;
        alert(`Too many tabs open (${allTabs.size}/${MAX_COUNT}).\n` +
              'Please close some to remain under web-browser-enforced SSE connections limit.\n\n' +
              'Note that this restriction can be lifted by switching to http2,\n' +
              'which in practice implies to host over https (eg behind a proxy).'
        );
    } else if (allTabs.size <= MAX_COUNT) {
        hasAlerted = false;
    }
}

// Heartbeat to detect dead tabs
function cleanupDeadTabs() {
    const heartbeats = JSON.parse(localStorage.getItem('tabHeartbeats') || '{}');
    const now = Date.now();
    const activeTabs = new Set([tabId]);
    
    Object.keys(heartbeats).forEach(id => {
        if (now - heartbeats[id] < 10000) { // 10 second timeout
            activeTabs.add(id);
        }
    });
    
    allTabs = activeTabs;
    saveTabCount();
}

// Update heartbeat
function updateHeartbeat() {
    const heartbeats = JSON.parse(localStorage.getItem('tabHeartbeats') || '{}');
    heartbeats[tabId] = Date.now();
    localStorage.setItem('tabHeartbeats', JSON.stringify(heartbeats));
}

// Test when tab becomes visible
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        cleanupDeadTabs();
        checkTabLimit();
    }
});

// Test when window gets focus
window.addEventListener('focus', () => {
    cleanupDeadTabs();
    checkTabLimit();
});

// Register this tab
channel.postMessage({type: 'register', tabId: tabId});

// Listen for messages from other tabs
channel.onmessage = function(event) {
    switch(event.data.type) {
        case 'register':
            allTabs.add(event.data.tabId);
            channel.postMessage({type: 'response', tabId: tabId});
            //console.log(`Tabs open: ${allTabs.size}`);
            saveTabCount();
            if (!document.hidden) checkTabLimit();
            break;
            
        case 'response':
            allTabs.add(event.data.tabId);
            //console.log(`Tabs open: ${allTabs.size}`);
            saveTabCount();
            if (!document.hidden) checkTabLimit();
            break;
            
        case 'unload':
            allTabs.delete(event.data.tabId);
            //console.log(`Tabs open: ${allTabs.size}`);
            saveTabCount();
            checkTabLimit();
            break;
    }
};

// Remove tab when closed/refreshed
window.addEventListener('beforeunload', () => {
    channel.postMessage({type: 'unload', tabId: tabId});
    const heartbeats = JSON.parse(localStorage.getItem('tabHeartbeats') || '{}');
    delete heartbeats[tabId];
    localStorage.setItem('tabHeartbeats', JSON.stringify(heartbeats));
});

// Periodic cleanup of dead tabs
setInterval(() => {
    updateHeartbeat();
    cleanupDeadTabs();
    channel.postMessage({type: 'ping', tabId: tabId});
}, 5000);

// Initial count request
setTimeout(() => {
    cleanupDeadTabs();
    //console.log(`Initial tabs open: ${allTabs.size}`);
    if (!document.hidden) checkTabLimit();
}, 500);

