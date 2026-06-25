/*
 * Masha — Browser extension selection translator.
 * Copyright (C) 2026  Clementine Pendragon <clem@pendragon.systems>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Contact: clem@pendragon.systems (Clementine Pendragon, c/o Xian Project Development)
 */

import { useState, useEffect, CSSProperties } from 'react';
import { shouldWarnHttpToNonLoopback } from '../../utils/lemonadeUrl';
import { getConfig, setConfig } from '../../utils/config';
import { DEFAULT_CONFIG } from '../../platform/bridge';

const SOURCE_LANGS = ['Auto', 'Chinese', 'Japanese', 'Korean', 'English', 'Spanish', 'French', 'German', 'Russian', 'Arabic', 'Hindi', 'Vietnamese'];
const TARGET_LANGS = SOURCE_LANGS.filter((l) => l !== 'Auto');

function App() {
  const [serverUrl, setServerUrl] = useState(DEFAULT_CONFIG.serverUrl);
  const [sourceLang, setSourceLang] = useState(DEFAULT_CONFIG.sourceLang);
  const [targetLang, setTargetLang] = useState(DEFAULT_CONFIG.targetLang);
  const [styles, setStyles] = useState('');
  const [status, setStatus] = useState<'idle' | 'saving' | 'saved'>('idle');

  useEffect(() => {
    getConfig().then((cfg) => {
      setServerUrl(cfg.serverUrl);
      setSourceLang(cfg.sourceLang);
      setTargetLang(cfg.targetLang);
      setStyles(cfg.styles.join(', '));
    });
  }, []);

  const handleSave = async () => {
    setStatus('saving');
    if (shouldWarnHttpToNonLoopback(serverUrl)) {
      const ok = window.confirm(
        'Warning: You are saving an HTTP Lemonade URL that is not localhost. Traffic on untrusted networks can be intercepted.\n\nClick OK to save or Cancel to edit.',
      );
      if (!ok) {
        setStatus('idle');
        return;
      }
    }
    await setConfig({
      serverUrl,
      sourceLang,
      targetLang,
      styles: styles.split(',').map((s) => s.trim()).filter(Boolean),
    });
    const cfg = await getConfig();
    setServerUrl(cfg.serverUrl);
    setStatus('saved');
    setTimeout(() => setStatus('idle'), 2000);
  };

  const label: CSSProperties = { fontSize: '12px', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' };
  const field: CSSProperties = {
    width: '100%', padding: '10px 12px', backgroundColor: '#1e293b', border: '1px solid #334155',
    borderRadius: '8px', color: '#f8fafc', fontSize: '13px', boxSizing: 'border-box', outline: 'none',
  };

  return (
    <div style={{ width: '320px', padding: '24px', fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', backgroundColor: '#0f172a', color: '#f8fafc', boxSizing: 'border-box' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
        <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'linear-gradient(135deg, #6366f1, #a855f7)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '18px', boxShadow: '0 0 15px rgba(99, 102, 241, 0.5)' }}>✨</div>
        <div>
          <h1 style={{ fontSize: '18px', fontWeight: 800, margin: 0, letterSpacing: '-0.025em', background: 'linear-gradient(to right, #818cf8, #c084fc)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>MASHA</h1>
          <p style={{ fontSize: '11px', color: '#94a3b8', margin: 0 }}>Context-Aware Translator</p>
        </div>
      </div>

      <p style={{ fontSize: '13px', lineHeight: 1.5, color: '#cbd5e1', marginTop: 0, marginBottom: '18px' }}>
        Select text on any page, right-click, and choose <strong>Translate with MASHA</strong>.
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <label style={label}>Lemonade Node URL</label>
          <input type="text" value={serverUrl} onChange={(e) => setServerUrl(e.target.value)} placeholder="http://localhost:13305/v1" style={field} />
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={label}>From</label>
            <select value={sourceLang} onChange={(e) => setSourceLang(e.target.value)} style={field}>
              {SOURCE_LANGS.map((l) => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={label}>To</label>
            <select value={targetLang} onChange={(e) => setTargetLang(e.target.value)} style={field}>
              {TARGET_LANGS.map((l) => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <label style={label}>Style terms (optional)</label>
          <input type="text" value={styles} onChange={(e) => setStyles(e.target.value)} placeholder="e.g. formal, gaming" style={field} />
        </div>
      </div>

      <button
        onClick={handleSave}
        disabled={status === 'saving'}
        style={{ marginTop: '20px', width: '100%', padding: '11px', background: status === 'saved' ? '#10b981' : 'linear-gradient(135deg, #6366f1, #a855f7)', color: '#ffffff', border: 'none', borderRadius: '8px', fontWeight: 600, fontSize: '13px', cursor: 'pointer', transition: 'all 0.2s ease' }}
      >
        {status === 'saving' ? 'Saving...' : status === 'saved' ? 'Saved!' : 'Save Settings'}
      </button>

      <div style={{ marginTop: '16px', paddingTop: '12px', borderTop: '1px solid #1e293b', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '11px', color: '#475569' }}>v1.0.0</span>
        <span style={{ fontSize: '11px', color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#10b981', display: 'inline-block' }}></span> Active
        </span>
      </div>
    </div>
  );
}

export default App;
