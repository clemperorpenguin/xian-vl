import { useState, useEffect } from 'react';
import { normalizeLemonadeBaseUrl, shouldWarnHttpToNonLoopback } from '../../utils/lemonadeUrl';

function App() {
  const [serverUrl, setServerUrl] = useState('http://localhost:13305/v1');
  const [status, setStatus] = useState<'idle' | 'saving' | 'saved'>('idle');

  useEffect(() => {
    chrome.storage.local.get(['serverUrl']).then((result: { [key: string]: any }) => {
      if (result.serverUrl) {
        setServerUrl(normalizeLemonadeBaseUrl(result.serverUrl));
      }
    });
  }, []);

  const handleSave = () => {
    setStatus('saving');
    const normalized = normalizeLemonadeBaseUrl(serverUrl);
    
    if (shouldWarnHttpToNonLoopback(normalized)) {
      const ok = window.confirm(
        'Warning: You are saving an HTTP Lemonade URL that is not localhost. Traffic on untrusted networks can be intercepted.\n\nClick OK to save or Cancel to edit.',
      );
      if (!ok) {
        setStatus('idle');
        return;
      }
    }
    
    setServerUrl(normalized);
    chrome.storage.local.set({ serverUrl: normalized }).then(() => {
      setStatus('saved');
      setTimeout(() => setStatus('idle'), 2000);
    });
  };

  return (
    <div style={{
      width: '320px',
      padding: '24px',
      fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      backgroundColor: '#0f172a',
      color: '#f8fafc',
      boxSizing: 'border-box',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
        <div style={{
          width: '36px',
          height: '36px',
          borderRadius: '10px',
          background: 'linear-gradient(135deg, #6366f1, #a855f7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '18px',
          boxShadow: '0 0 15px rgba(99, 102, 241, 0.5)',
        }}>
          ✨
        </div>
        <div>
          <h1 style={{ fontSize: '18px', fontWeight: '800', margin: 0, letterSpacing: '-0.025em', background: 'linear-gradient(to right, #818cf8, #c084fc)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            MASHA
          </h1>
          <p style={{ fontSize: '11px', color: '#94a3b8', margin: 0 }}>Translate & Replace Assistant</p>
        </div>
      </div>

      {/* Description */}
      <p style={{ fontSize: '13px', lineHeight: '1.5', color: '#cbd5e1', marginTop: 0, marginBottom: '20px' }}>
        Select text on any page and click the floating button to instantly replace it with its translation.
      </p>

      {/* Settings section */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <label style={{ fontSize: '12px', fontWeight: '600', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Lemonade Node URL
        </label>
        <input
          type="text"
          value={serverUrl}
          onChange={(e) => setServerUrl(e.target.value)}
          placeholder="http://localhost:13305/v1"
          style={{
            width: '100%',
            padding: '10px 12px',
            backgroundColor: '#1e293b',
            border: '1px solid #334155',
            borderRadius: '8px',
            color: '#f8fafc',
            fontSize: '13px',
            boxSizing: 'border-box',
            outline: 'none',
            transition: 'border-color 0.2s',
          }}
          onFocus={(e) => e.target.style.borderColor = '#6366f1'}
          onBlur={(e) => e.target.style.borderColor = '#334155'}
        />
        <p style={{ fontSize: '11px', color: '#64748b', margin: 0, lineHeight: '1.4' }}>
          Must be an OpenAI-compatible endpoint (ends in <code style={{ color: '#94a3b8', fontSize: '11px' }}>/v1</code>).
        </p>
      </div>

      {/* Action button */}
      <button
        onClick={handleSave}
        disabled={status === 'saving'}
        style={{
          marginTop: '24px',
          width: '100%',
          padding: '11px',
          background: status === 'saved'
            ? '#10b981'
            : 'linear-gradient(135deg, #6366f1, #a855f7)',
          color: '#ffffff',
          border: 'none',
          borderRadius: '8px',
          fontWeight: '600',
          fontSize: '13px',
          cursor: 'pointer',
          boxShadow: status === 'saved'
            ? '0 4px 12px rgba(16, 185, 129, 0.2)'
            : '0 4px 12px rgba(99, 102, 241, 0.3)',
          transition: 'all 0.2s ease',
        }}
        onMouseOver={(e) => {
          if (status !== 'saved') {
            e.currentTarget.style.background = 'linear-gradient(135deg, #4f46e5, #9333ea)';
            e.currentTarget.style.transform = 'translateY(-1px)';
          }
        }}
        onMouseOut={(e) => {
          if (status !== 'saved') {
            e.currentTarget.style.background = 'linear-gradient(135deg, #6366f1, #a855f7)';
            e.currentTarget.style.transform = 'translateY(0)';
          }
        }}
      >
        {status === 'saving' ? 'Saving...' : status === 'saved' ? 'Saved Successfully!' : 'Save Connection'}
      </button>

      {/* Footer Info */}
      <div style={{ marginTop: '20px', paddingTop: '12px', borderTop: '1px solid #1e293b', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '11px', color: '#475569' }}>v1.0.0</span>
        <span style={{ fontSize: '11px', color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#10b981', display: 'inline-block' }}></span>
          Active
        </span>
      </div>
    </div>
  );
}

export default App;
