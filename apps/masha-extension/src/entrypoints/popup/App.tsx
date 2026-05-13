import { useState, useEffect } from 'react';

function App() {
  const [serverUrl, setServerUrl] = useState('http://localhost:13305/v1');
  const [status, setStatus] = useState<'idle' | 'saving' | 'saved'>('idle');
  const [imageMode, setImageMode] = useState(false);

  useEffect(() => {
    chrome.storage.local.get(['serverUrl']).then((result) => {
      if (result.serverUrl) {
        setServerUrl(result.serverUrl);
      }
    });
  }, []);

  const handleSave = () => {
    setStatus('saving');
    chrome.storage.local.set({ serverUrl }).then(() => {
      setStatus('saved');
      setTimeout(() => setStatus('idle'), 2000);
    });
  };

  const handleFullPageTranslate = async () => {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tabs[0]?.id) {
      chrome.tabs.sendMessage(tabs[0].id, { type: 'START_PAGE_TRANSLATION' });
    }
  };

  const handleToggleImageMode = async () => {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tabs[0]?.id) {
      chrome.tabs.sendMessage(tabs[0].id, { type: 'TOGGLE_IMAGE_MODE' }, (response) => {
        if (response && response.success) {
          setImageMode(response.active);
        }
      });
    }
  };

  const handleExportToLore = async () => {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tabs[0]?.id) {
      chrome.tabs.sendMessage(tabs[0].id, { type: 'GATHER_LORE_DATA' }, (response) => {
        if (response && response.success) {
          const blob = new Blob([JSON.stringify(response.data, null, 2)], { type: 'application/json' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `lore_export_${response.data.title.replace(/[^\w\s]/gi, '').replace(/\s+/g, '_')}.json`;
          a.click();
          URL.revokeObjectURL(url);
        }
      });
    }
  };

  return (
    <div style={{ width: '320px', padding: '20px', fontFamily: 'sans-serif', backgroundColor: '#f9fafb' }}>
      <h1 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '8px', color: '#1f2937' }}>MASHA Extension</h1>
      <p style={{ fontSize: '14px', color: '#6b7280', marginBottom: '20px' }}>
        Status: <span style={{ color: '#10b981', fontWeight: 'bold' }}>Active</span>
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <label style={{ fontSize: '13px', fontWeight: 'bold', color: '#374151' }}>Lemonade Server URL</label>
        <input 
          type="text" 
          value={serverUrl}
          onChange={(e) => setServerUrl(e.target.value)}
          placeholder="http://localhost:13305/v1"
          style={{ 
            width: '100%', 
            padding: '10px', 
            border: '1px solid #d1d5db', 
            borderRadius: '6px',
            boxSizing: 'border-box'
          }}
        />
      </div>

      <button 
        onClick={handleSave}
        style={{ 
          marginTop: '20px', 
          width: '100%', 
          padding: '10px', 
          backgroundColor: '#3b82f6', 
          color: 'white', 
          border: 'none', 
          borderRadius: '6px',
          fontWeight: 'bold',
          cursor: 'pointer',
          transition: 'background-color 0.2s'
        }}
        onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#2563eb'}
        onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#3b82f6'}
      >
        {status === 'saving' ? 'Saving...' : status === 'saved' ? 'Saved!' : 'Save Settings'}
      </button>

      <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px', borderTop: '1px solid #e5e7eb', paddingTop: '16px' }}>
        <button 
          onClick={handleFullPageTranslate}
          style={{ 
            width: '100%', 
            padding: '10px', 
            backgroundColor: '#10b981', 
            color: 'white', 
            border: 'none', 
            borderRadius: '6px',
            fontWeight: 'bold',
            cursor: 'pointer',
            transition: 'background-color 0.2s'
          }}
          onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#059669'}
          onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#10b981'}
        >
          ✨ Translate Full Page
        </button>

        <button 
          onClick={handleToggleImageMode}
          style={{ 
            width: '100%', 
            padding: '10px', 
            backgroundColor: imageMode ? '#f59e0b' : '#8b5cf6', 
            color: 'white', 
            border: 'none', 
            borderRadius: '6px',
            fontWeight: 'bold',
            cursor: 'pointer',
            transition: 'background-color 0.2s'
          }}
          onMouseOver={(e) => e.currentTarget.style.backgroundColor = imageMode ? '#d97706' : '#7c3aed'}
          onMouseOut={(e) => e.currentTarget.style.backgroundColor = imageMode ? '#f59e0b' : '#8b5cf6'}
        >
          {imageMode ? '🛑 Stop Image Selection' : '🖼️ Select Images to Translate'}
        </button>

        <button 
          onClick={handleExportToLore}
          style={{ 
            width: '100%', 
            padding: '10px', 
            backgroundColor: '#6366f1', 
            color: 'white', 
            border: 'none', 
            borderRadius: '6px',
            fontWeight: 'bold',
            cursor: 'pointer',
            transition: 'background-color 0.2s'
          }}
          onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#4f46e5'}
          onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#6366f1'}
        >
          📚 Export to LORE
        </button>
      </div>
    </div>
  );
}

export default App;
