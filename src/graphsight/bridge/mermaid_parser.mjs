import { JSDOM } from 'jsdom';

// 1. Setup JSDOM environment BEFORE importing mermaid
const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>');

// Global pollution required for Mermaid to think it's in a browser
global.window = dom.window;
global.document = dom.window.document;
global.Element = dom.window.Element;
global.SVGElement = dom.window.SVGElement;
global.HTMLElement = dom.window.HTMLElement;
global.Node = dom.window.Node;

// Polyfill for some browser specific APIs
global.window.matchMedia = global.window.matchMedia || function() {
    return {
        matches: false,
        addListener: function() {},
        removeListener: function() {}
    };
};

async function main() {
  try {
    // 2. Dynamically import mermaid AFTER globals are set
    // This ensures mermaid sees the 'window' object upon initialization
    const mermaidModule = await import('mermaid');
    const mermaid = mermaidModule.default;

    // Initialize Mermaid
    mermaid.initialize({
      startOnLoad: false,
      logLevel: 'error',
      securityLevel: 'loose', // Helps avoid some strict DOMPurify checks in node
    });

    // Read from Stdin
    const chunks = [];
    process.stdin.on('data', chunk => chunks.push(chunk));
    process.stdin.on('end', async () => {
      const code = Buffer.concat(chunks).toString('utf-8');
      if (!code.trim()) return;
      
      try {
        const result = await parseMermaid(mermaid, code);
        console.log(JSON.stringify(result));
      } catch (err) {
        console.error("Mermaid Parsing Logic Error:", err.message);
        process.exit(1);
      }
    });

  } catch (error) {
    console.error("Script Initialization Error:", error.message);
    process.exit(1);
  }
}

async function parseMermaid(mermaid, code) {
  // mermaidAPI.getDiagramFromText parses and returns the diagram object
  const diagram = await mermaid.mermaidAPI.getDiagramFromText(code);
  
  // Access internal database (db)
  const db = diagram.db;
  
  // Extract Nodes (Vertices)
  // v10/v11 handles db.getVertices() differently depending on diagram type,
  // but for flowchart, it usually returns an object map.
  const vertices = typeof db.getVertices === 'function' ? db.getVertices() : {};
  const nodes = Object.entries(vertices).map(([id, node]) => ({
    id: id,
    label: node.text || id,
    shape: normalizeShape(node.type),
  }));

  // Extract Edges
  const edgesRaw = typeof db.getEdges === 'function' ? db.getEdges() : [];
  const edges = edgesRaw.map(edge => ({
    src: edge.start,
    dst: edge.end,
    label: edge.text || "",
    style: "-->" // Simplify style for now
  }));

  const direction = typeof db.getDirection === 'function' ? db.getDirection() : 'TD';

  return {
    direction,
    nodes,
    edges
  };
}

function normalizeShape(type) {
  const map = {
    'round': 'round',
    'square': 'rect',
    'rect': 'rect',
    'diamond': 'diamond',
    'stadium': 'stadium',
    'circle': 'circle',
    'hexagon': 'hex',
    'double-circle': 'circle',
  };
  return map[type] || 'rect';
}

// Start the script
main();

