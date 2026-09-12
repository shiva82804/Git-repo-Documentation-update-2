import React, { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';

mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  themeVariables: {
    primaryColor: '#00d4ff',
    primaryTextColor: '#fff',
    primaryBorderColor: '#00d4ff',
    lineColor: '#3b82f6',
    secondaryColor: '#131724',
    tertiaryColor: '#1e2a45',
    fontSize: '14px',
  },
});

const sanitizeMermaidCode = (tabName, rawCode) => {
  if (!rawCode) return '';
  let code = rawCode.trim();
  code = code.replace(/^```(?:mermaid)?/i, '').replace(/```$/i, '').trim();

  // Class diagram cleaning
  if (tabName === 'Class_Diagram' || code.startsWith('classDiagram')) {
    // Replace generics <T> with ~T~
    code = code.replace(/<([a-zA-Z0-9_, ]+)>/g, '~$1~');

    // Split multiple statements if on a single line
    const arrowSyms = '(?:<\\|--|--\\|>|\\*--|--\\*|o--|--o|-->|<--|\\.\\.>|<\\.\\.|\\.\\.\\|>|<\\|\\.\\.|--|\\.\\.)';
    code = code.replace(new RegExp('("|[a-zA-Z0-9_]+)\\s*([a-zA-Z0-9_]+\\s*' + arrowSyms + ')', 'g'), '$1\n$2');

    const arrowRegex = /(\s*(?:<\|--|--\|>|\*--|--\*|o--|--o|-->|<--|\.\.>|<\.\.|\.\.\|>|<\|\.\.|--|\.\.)\s*)/;
    const lines = code.split('\n');
    const cleanedLines = [];

    for (let line of lines) {
      let trimmed = line.trim();
      if (!trimmed) continue;
      if (arrowRegex.test(trimmed)) {
        let [relPart, ...labelParts] = trimmed.split(':');
        let labelSuffix = labelParts.length > 0 ? ` : ${labelParts.join(':').trim()}` : '';

        const match = relPart.match(arrowRegex);
        if (match) {
          const arrow = match[1].trim();
          let left = relPart.substring(0, match.index).trim();
          let right = relPart.substring(match.index + match[0].length).trim();

          // If right is quoted identifier e.g. "uvicorn" -> uvicorn
          const rightMatch = right.match(/^["']([^"']+)["']$/);
          if (rightMatch) {
            right = rightMatch[1].replace(/[^a-zA-Z0-9_]/g, '_');
          }

          // If left is quoted identifier e.g. "Agent" -> Agent
          const leftMatch = left.match(/^["']([^"']+)["']$/);
          if (leftMatch) {
            left = leftMatch[1].replace(/[^a-zA-Z0-9_]/g, '_');
          }

          cleanedLines.push(`    ${left} ${arrow} ${right}${labelSuffix}`);
          continue;
        }
      }
      cleanedLines.push(line);
    }
    code = cleanedLines.join('\n');
  }

  return code;
};

const DiagramViewer = ({ diagrams }) => {
  const [activeTab, setActiveTab] = useState(null);
  const [showCode, setShowCode] = useState(false);
  const containerRef = useRef(null);
  const codeContainerRef = useRef(null);

  useEffect(() => {
    if (!diagrams || Object.keys(diagrams).length === 0) return;
    const keys = Object.keys(diagrams);
    if (!activeTab || !keys.includes(activeTab)) setActiveTab(keys[0]);
  }, [diagrams, activeTab]);

  // Render diagram when tab changes or showCode toggles (but code display doesn't affect rendering)
  useEffect(() => {
    if (!activeTab || !diagrams[activeTab] || !containerRef.current) return;
    const render = async () => {
      try {
        containerRef.current.innerHTML = '';
        const cleanId = `mermaid-${activeTab.replace(/[^a-zA-Z0-9]/g, '')}-${Date.now()}`;
        const cleanCode = sanitizeMermaidCode(activeTab, diagrams[activeTab]);
        const { svg } = await mermaid.render(
          cleanId,
          cleanCode
        );
        containerRef.current.innerHTML = svg;
      } catch (error) {
        console.error('Mermaid render error:', error);
        containerRef.current.innerHTML = `<pre style="color: #ff6b6b;">⚠️ Error rendering diagram:\n${error.message}</pre>`;
        // Also show the raw code automatically if there's an error
        setShowCode(true);
      }
    };
    render();
  }, [activeTab, diagrams]);

  if (!diagrams || Object.keys(diagrams).length === 0) {
    return (
      <div className="card">
        <h2>📊 Diagrams</h2>
        <div style={{ color: 'var(--text-secondary)', padding: '20px' }}>
          No diagrams generated yet.
        </div>
      </div>
    );
  }

  const tabs = Object.keys(diagrams);
  const currentCode = activeTab ? sanitizeMermaidCode(activeTab, diagrams[activeTab]) : '';

  return (
    <div className="card" style={{ marginBottom: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h2 style={{ margin: 0 }}>📊 Diagrams</h2>
        <button
          className="btn-outline"
          onClick={() => setShowCode(!showCode)}
          style={{ padding: '4px 12px', fontSize: '0.85rem' }}
        >
          {showCode ? 'Hide Code' : 'Show Code'}
        </button>
      </div>

      <div className="tab-bar">
        {tabs.map((tab) => (
          <button
            key={tab}
            className={`tab-btn ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab.replace(/_/g, ' ')}
          </button>
        ))}
      </div>

      <div className="diagram-container" ref={containerRef} />

      {showCode && (
        <div style={{ marginTop: '16px' }}>
          <div style={{ 
            background: '#0a0e1a', 
            padding: '16px', 
            borderRadius: '8px',
            border: '1px solid rgba(255,255,255,0.08)',
            overflow: 'auto',
            maxHeight: '300px',
          }}>
            <pre style={{ 
              margin: 0, 
              fontFamily: 'JetBrains Mono, monospace', 
              fontSize: '12px',
              color: '#e8edf5',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
            }}>
              {currentCode}
            </pre>
          </div>
          <div style={{ marginTop: '8px', display: 'flex', gap: '8px' }}>
            <button
              className="btn-outline"
              onClick={() => {
                navigator.clipboard.writeText(currentCode);
                alert('Copied to clipboard!');
              }}
              style={{ padding: '4px 12px', fontSize: '0.8rem' }}
            >
              📋 Copy
            </button>
            <button
              className="btn-outline"
              onClick={() => {
                // Open Mermaid Live with encoded code
                const encoded = encodeURIComponent(currentCode);
                window.open(`https://mermaid.live/edit#pako:${btoa(currentCode)}`, '_blank');
              }}
              style={{ padding: '4px 12px', fontSize: '0.8rem' }}
            >
              🔗 Open in Mermaid Live
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default DiagramViewer;